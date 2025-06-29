from transformers import ViTConfig
from transformers import ViTImageProcessor
from transformers import TFViTForImageClassification
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_name = "google/vit-base-patch16-224"
config = ViTConfig.from_pretrained(model_name)

config.num_labels = 2

processor = ViTImageProcessor.from_pretrained(model_name)

expected_image_size = processor.size['height']
print(f"Hugging Face model expects input images of size: {expected_image_size}x{expected_image_size}")


try:
    hf_model = TFViTForImageClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )


except FileNotFoundError:
    print(f"Error: Model files for {model_name} not found. Check model name and internet connection.")
except Exception as e:
    print(f"An error occurred during model loading: {e}")
    hf_model = None

# Define the size variable, e.g.
path_100 = '/scratch/s3333302/drive_thesis/models/100_checkpoint_2.weights.h5'

path = path_100

# Load the saved weights into the Hugging Face model
if hf_model:
    print(f"Loading weights from: {path}")
    try:
        hf_model.load_weights(path)
        print("Successfully loaded model weights.")
    except tf.errors.NotFoundError:
        print(f"Error: Weights file not found at {path}")
    except Exception as e:
        print(f"An error occurred during weight loading: {e}")
else:
    print("Hugging Face model (hf_model) is not loaded. Cannot load weights.")

def preprocess_data(sizes, X_train_preprocessed_list=[], y_train_preprocessed_list=[], X_test_preprocessed_list=[], y_test_preprocessed_list=[]):

    # ViT takes inputs of size 224x224
    expected_image_size = 224

    for i in range(len(sizes)):
        size = sizes[i]

        folder = '/scratch/s3333302/drive_thesis/thesis/npy/cropped/n_1/'
        h5_split_file = os.path.join(folder, 'train_test_split.h5')

        X_train_processed_list = []
        X_test_processed_list = []
        y_train_list = []
        y_test_list = []

        batch_size = 32

        print(f"Loading and preprocessing data from HDF5 file in batches: {h5_split_file}...")
        with h5py.File(h5_split_file, 'r') as hf:
            if 'X_train' in hf and 'y_train' in hf and 'X_test' in hf and 'y_test' in hf:
                train_images_dataset = hf['X_train']
                train_labels_dataset = hf['y_train']
                test_images_dataset = hf['X_test']
                test_labels_dataset = hf['y_test']

                print(f"Successfully accessed train/test datasets from HDF5.")
                print(f"  X_train dataset shape: {train_images_dataset.shape}, Dtype: {train_images_dataset.dtype}")
                print(f"  y_train dataset shape: {train_labels_dataset.shape}, Dtype: {train_labels_dataset.dtype}")
                print(f"  X_test dataset shape: {test_images_dataset.shape}, Dtype: {test_images_dataset.dtype}")
                print(f"  y_test dataset shape: {test_labels_dataset.shape}, Dtype: {test_labels_dataset.dtype}")

                target_height = target_width = expected_image_size # Use the expected size from the processor


                print("\nPreprocessing training images in batches...")
                for i in tqdm(range(0, train_images_dataset.shape[0], batch_size), desc="Preprocessing Train"):
                    batch_images = train_images_dataset[i : i + batch_size]
                    batch_labels = train_labels_dataset[i : i + batch_size]

                    processed_inputs = processor(images=list(batch_images), return_tensors="tf")
                    X_train_processed_list.append(processed_inputs['pixel_values'])
                    y_train_list.append(batch_labels)

                X_train_processed = tf.concat(X_train_processed_list, axis=0)
                y_train_tf = tf.constant(np.concatenate(y_train_list, axis=0), dtype=tf.int64)


                print("\nPreprocessing testing images in batches...")
                for i in tqdm(range(0, test_images_dataset.shape[0], batch_size), desc="Preprocessing Test"):
                    batch_images = test_images_dataset[i : i + batch_size]
                    batch_labels = test_labels_dataset[i : i + batch_size]

                    processed_inputs = processor(images=list(batch_images), return_tensors="tf")
                    X_test_processed_list.append(processed_inputs['pixel_values'])
                    y_test_list.append(batch_labels)

                X_test_processed = tf.concat(X_test_processed_list, axis=0)
                y_test_tf = tf.constant(np.concatenate(y_test_list, axis=0), dtype=tf.int64)


                print("\nChecking shape for transposition...")
                if len(X_train_processed.shape) == 4 and X_train_processed.shape[2] == 3 and X_train_processed.shape[3] == expected_image_size: # Explicitly check for (Batch, H, C, W)
                     print(f"Detected (Batch, H, C, W) format: {X_train_processed.shape}. Transposing to (Batch, H, W, C)...")
                     X_train_processed = tf.transpose(X_train_processed, perm=[0, 1, 3, 2])
                     X_test_processed = tf.transpose(X_test_processed, perm=[0, 1, 3, 2])
                     print(f"Shape after transposing: {X_train_processed.shape}")
                elif len(X_train_processed.shape) == 4 and X_train_processed.shape[3] == 3: # Check if already NHWC (Batch, H, W, C)
                     print(f"Detected NHWC format: {X_train_processed.shape}. No transposition needed.")
                else:
                     print("Warning: Processed image shape is not in expected (Batch, H, C, W) or (Batch, H, W, C) format for transposition check:", X_train_processed.shape)


                X_train_processed = tf.cast(X_train_processed, tf.float32)
                X_test_processed = tf.cast(X_test_processed, tf.float32)

                X_train_preprocessed_list.append(X_train_processed)
                X_test_preprocessed_list.append(X_test_processed)
                y_train_preprocessed_list.append(y_train_tf)
                y_test_preprocessed_list.append(y_test_tf)

                print("\nPreprocessing complete.")
                print(f"Final shape of X_train_processed: {X_train_processed.shape}, Dtype: {X_train_processed.dtype}")
                print(f"Final shape of y_train_tf: {y_train_tf.shape}, Dtype: {y_train_tf.dtype}")
                print(f"Final shape of X_test_processed: {X_test_processed.shape}, Dtype: {X_test_processed.dtype}")
                print(f"Final shape of y_test_tf: {y_test_tf.shape}, Dtype: {y_test_tf.dtype}")

                return X_train_preprocessed_list, y_train_preprocessed_list, X_test_preprocessed_list, y_test_preprocessed_list


            else:
                print("Error: Required datasets (X_train, y_train, X_test, y_test) not found in HDF5 file.")
                return


def predict(models, predict_output, X_test_data, y_test_data):
    for i in range(len(y_test_data)):
        hf_model = models[i]
        X_test_processed = X_test_data[i]
        y_test_tf = y_test_data[i]

        if hf_model:
            print("Making predictions on the test set...")
            try:
                predictions = hf_model.predict(X_test_processed)

                predicted_classes = tf.argmax(predictions.logits, axis=-1).numpy()

                print("Predictions complete.")
                print("Shape of predicted_classes:", predicted_classes.shape)
                print("First 10 predicted classes:", predicted_classes[:50])
                print("First 10 true labels:", y_test_tf.numpy()[:50])

                predict_output.append(predicted_classes)

                cm = confusion_matrix(y_test_tf.numpy(), predicted_classes)

                print(cm)

                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')

                plt.savefig(f"/scratch/s3333302/drive_thesis/results/confusion_matrix_{i}.png")

            except NameError:
                print("Error: X_test_processed or hf_model is not defined. Please run the previous cells to load and preprocess the data and load the model.")
            except Exception as e:
                print(f"An error occurred during prediction: {e}")
        else:
            print("Hugging Face model (hf_model) is not loaded. Cannot make predictions.")

    return predict_output


def save_predictions(models, X_test_data, y_test_data):
    predictions = []
    predictions = predict(models, predictions, X_test_data, y_test_data)

    for i in range(len(predictions)):
        d = {'real label': y_test_data[i].numpy(),'predicted': predictions[i]}

        results = pd.DataFrame(data=d)

        output_path = f"/scratch/s3333302/drive_thesis/results/labels/predictions_{i}.csv"
        results.to_csv(output_path)

        f1 = f1_score(y_test_data[i].numpy(), predictions[i])
        classification = classification_report(y_test_data[i].numpy(), predictions[i], digits=4)
        print(f"Saved real/pred labels to {output_path}.\n F1 Score: {f1:.4f}\n Classification Report: {classification}\n")


def predict_and_save(models):
    X_train_data = []
    y_train_data = []
    X_test_data = []
    y_test_data = []

    data_sizes = [1200]

    X_train_data, y_train_data, X_test_data, y_test_data = preprocess_data(data_sizes, X_train_data, y_train_data, X_test_data, y_test_data)

    save_predictions(models, X_test_data, y_test_data)



models = [hf_model]
predict_and_save(models)