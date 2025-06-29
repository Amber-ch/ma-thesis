from transformers import ViTConfig
from transformers import ViTImageProcessor
from transformers import TFViTForImageClassification
from transformers import set_seed
import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import numpy as np
import os
from tqdm import tqdm
import h5py

set_seed(42)

size = 300
crop = 'n5'
folder = f'/scratch/s3333302/drive_thesis/thesis/npy/cropped/n_5'
h5_split_file = os.path.join(folder, 'train_test_split.h5')  # Path to your HDF5 file

model_name = "google/vit-base-patch16-224"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

config = ViTConfig.from_pretrained(model_name)

config.num_labels = 2


processor = ViTImageProcessor.from_pretrained(model_name)

expected_image_size = processor.size['height']
print(f"Hugging Face model expects input images of size: {expected_image_size}x{expected_image_size}")


print("model loading:")
hf_model = TFViTForImageClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True
)

print(f"Successfully loaded Hugging Face model: {model_name} with modified config.")
hf_model.summary()

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


        target_height = target_width = expected_image_size

        print("\nPreprocessing training images in batches...")
        for i in tqdm(range(0, train_images_dataset.shape[0], batch_size), desc="Preprocessing Train"):
            batch_images = train_images_dataset[i: i + batch_size]
            batch_labels = train_labels_dataset[i: i + batch_size]

            processed_inputs = processor(images=list(batch_images), return_tensors="tf")
            X_train_processed_list.append(processed_inputs['pixel_values'])
            y_train_list.append(batch_labels)

        X_train_processed = tf.concat(X_train_processed_list, axis=0)
        y_train_tf = tf.constant(np.concatenate(y_train_list, axis=0),
                                 dtype=tf.int64)

        print("\nPreprocessing testing images in batches...")
        for i in tqdm(range(0, test_images_dataset.shape[0], batch_size), desc="Preprocessing Test"):
            batch_images = test_images_dataset[i: i + batch_size]
            batch_labels = test_labels_dataset[i: i + batch_size]

            processed_inputs = processor(images=list(batch_images), return_tensors="tf")
            X_test_processed_list.append(processed_inputs['pixel_values'])
            y_test_list.append(batch_labels)

        X_test_processed = tf.concat(X_test_processed_list, axis=0)
        y_test_tf = tf.constant(np.concatenate(y_test_list, axis=0),
                                dtype=tf.int64)


        print("\nChecking shape for transposition...")
        if len(X_train_processed.shape) == 4 and X_train_processed.shape[2] == 3 and X_train_processed.shape[
            3] == expected_image_size:  # Explicitly check for (Batch, H, C, W)
            print(
                f"Detected (Batch, H, C, W) format: {X_train_processed.shape}. Transposing to (Batch, H, W, C)...")
            X_train_processed = tf.transpose(X_train_processed, perm=[0, 1, 3, 2])
            X_test_processed = tf.transpose(X_test_processed, perm=[0, 1, 3, 2])
            print(f"Shape after transposing: {X_train_processed.shape}")
        elif len(X_train_processed.shape) == 4 and X_train_processed.shape[
            3] == 3:  # Check if already NHWC (Batch, H, W, C)
            print(f"Detected NHWC format: {X_train_processed.shape}. No transposition needed.")
        else:
            print(
                "Warning: Processed image shape is not in expected (Batch, H, C, W) or (Batch, H, W, C) format for transposition check:",
                X_train_processed.shape)

        X_train_processed = tf.cast(X_train_processed, tf.float32)
        X_test_processed = tf.cast(X_test_processed, tf.float32)

        print("\nPreprocessing complete.")
        print(f"Final shape of X_train_processed: {X_train_processed.shape}, Dtype: {X_train_processed.dtype}")
        print(f"Final shape of y_train_tf: {y_train_tf.shape}, Dtype: {y_train_tf.dtype}")
        print(f"Final shape of X_test_processed: {X_test_processed.shape}, Dtype: {X_test_processed.dtype}")
        print(f"Final shape of y_test_tf: {y_test_tf.shape}, Dtype: {y_test_tf.dtype}")


    else:
        print("Error: Required datasets (X_train, y_train, X_test, y_test) not found in HDF5 file.")



if hf_model:

    print("Compiling Hugging Face model...")
    optimizer_instance = 'adam'

    hf_model.compile(
        optimizer=optimizer_instance,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    print("Model compilation complete.")


    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)


    lr_scheduler_callback = LearningRateScheduler(scheduler)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    checkpoint_filepath = f'/scratch/s3333302/drive_thesis/models/{size}_checkpoint_crop_{crop}.weights.h5'

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    callbacks_list = [
        lr_scheduler_callback,
        early_stopping_callback,
        model_checkpoint_callback
    ]

    print("\n--- Shapes and Dtypes before Training ---")
    print("X_train_processed shape:", X_train_processed.shape)
    print("X_train_processed dtype:", X_train_processed.dtype)
    print("y_train_tf shape:", y_train_tf.shape)
    print("y_train_tf dtype:", y_train_tf.dtype)
    print("X_test_processed shape:", X_test_processed.shape)
    print("X_test_processed dtype:", X_test_processed.dtype)
    print("y_test_tf shape:", y_test_tf.shape)
    print("y_test_tf dtype:", y_test_tf.dtype)
    print("------------------------------------------\n")

    print("\nStarting fine-tuning of Hugging Face model with Early Stopping and LR Scheduling...")
    history = hf_model.fit(
        X_train_processed,
        y_train_tf,
        epochs=100,
        batch_size=8,
        validation_data=(X_test_processed, y_test_tf),
        callbacks=callbacks_list
    )
    print("Fine-tuning finished.")

    print("\nEvaluating Hugging Face model...")
    loss, accuracy = hf_model.evaluate(X_test_processed, y_test_tf, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

else:
    print("Hugging Face model (hf_model) is not loaded. Please run the model loading cell first.")