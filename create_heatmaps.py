from transformers import TFViTForImageClassification, ViTConfig, ViTImageProcessor
import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import pandas as pd

model_name = "google/vit-base-patch16-224"


config = ViTConfig.from_pretrained(model_name)


config.num_labels = 2


processor = ViTImageProcessor.from_pretrained(model_name)

expected_image_size = processor.size['height']  # Assuming height == width
print(f"Hugging Face model expects input images of size: {expected_image_size}x{expected_image_size}")


try:
    hf_model = TFViTForImageClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True  # Ignore shape mismatches, specifically for the classification head
    )



except FileNotFoundError:
    print(f"Error: Model files for {model_name} not found. Check model name and internet connection.")
except Exception as e:
    print(f"An error occurred during model loading: {e}")
    hf_model = None

# Define the size variable
# CHANGEHERE
path_300 = '/scratch/s3333302/drive_thesis/models/300_checkpoint_100.weights.h5'
path_300_crop_n2 = '/scratch/s3333302/drive_thesis/models/crop_img_only_300_checkpoint_crop_n2.weights.h5'

path = path_300

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


def preprocess_data(sizes, X_train_preprocessed_list=[], y_train_preprocessed_list=[], X_test_preprocessed_list=[],
                    y_test_preprocessed_list=[]):
    # ViT takes inputs of size 224x224
    expected_image_size = 224

    for i in range(len(sizes)):
        size = sizes[i]

        h5_split_file = f'/scratch/s3333302/drive_thesis/thesis/npy/cropped/n_2/train_test_split.h5'

        X_train_processed_list = []
        X_test_processed_list = []
        y_train_list = []
        y_test_list = []

        batch_size = 32

        print(f"Loading and preprocessing data from HDF5 file in batches: {h5_split_file}...")
        # try:
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


                target_height = target_width = expected_image_size  # Use the expected size from the processor

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
#%%
#CHANGEHERE

sizes = [300]

X_train_data_300_crop, y_train_data_300_crop, X_test_data_300_crop, y_test_data_300_crop = [], [], [], []

X_train_data_300_crop, y_train_data_300_crop, X_test_data_300_crop, y_test_data_300_crop = preprocess_data(sizes, X_train_data_300_crop, y_train_data_300_crop, X_test_data_300_crop, y_test_data_300_crop)

n_sample = 22
real_predreal_300_crop = []
synthetic_predreal_300_crop = []
real_predsynthetic_300_crop = []
synthetic_predsynthetic_300_crop = []

# select random samples for each category from the csv file with predictions
path_300 = '/scratch/s3333302/drive_thesis/results/labels/predictions_300.csv'
path_300_crop = '/scratch/s3333302/drive_thesis/results/labels/predictions_300_crop_only_n2.csv'

predictions = pd.read_csv(path_300)
real_real_index = predictions.loc[(predictions['real label']==0) & (predictions['predicted']==0)]
synthetic_real_index = predictions.loc[(predictions['real label']==1) & (predictions['predicted']==0)]
real_synthetic_index = predictions.loc[(predictions['real label']==0) & (predictions['predicted']==1)]
synthetic_synthetic_index = predictions.loc[(predictions['real label']==1) & (predictions['predicted']==1)]

real_real_index = real_real_index.rename(columns={'Unnamed: 0':'index'})
synthetic_real_index = synthetic_real_index.rename(columns={'Unnamed: 0':'index'})
real_synthetic_index = real_synthetic_index.rename(columns={'Unnamed: 0':'index'})
synthetic_synthetic_index = synthetic_synthetic_index.rename(columns={'Unnamed: 0':'index'})

random_state = 42
real_real_sample = real_real_index.sample(n_sample, random_state=random_state)
synthetic_real_sample = synthetic_real_index.sample(n_sample, random_state=random_state)
real_synthetic_sample = real_synthetic_index.sample(n_sample, random_state=random_state)
synthetic_synthetic_sample = synthetic_synthetic_index.sample(n_sample, random_state=random_state)

real_real_sample_index = real_real_sample['index'].tolist()
synthetic_real_sample_index = synthetic_real_sample['index'].tolist()
real_synthetic_sample_index = real_synthetic_sample['index'].tolist()
synthetic_synthetic_sample_index = synthetic_synthetic_sample['index'].tolist()

def get_images_from_index(images, indexes):
    image_array = []
    for index in indexes:
        image_array.append(images[0][index])

    return image_array

real_real_images = get_images_from_index(X_train_data_100, real_real_sample_index)
synthetic_real_images = get_images_from_index(X_train_data_100, synthetic_real_sample_index)
real_synthetic_images = get_images_from_index(X_train_data_100, real_synthetic_sample_index)
synthetic_synthetic_images = get_images_from_index(X_train_data_100, synthetic_synthetic_sample_index)

all_images = [real_real_images, synthetic_real_images, real_synthetic_images, synthetic_synthetic_images]
types = ["real_real", "synthetic_real", "real_synthetic", "synthetic_synthetic"]




def numpy_to_image_channels_first(numpy_array):

    if numpy_array.ndim != 3 or numpy_array.shape[0] != 3:
        print(f"Error: Expected input shape (3, H, W), but received {numpy_array.shape}")
        return None

    if numpy_array.dtype == np.float32 or numpy_array.dtype == np.float64:
        min_val = np.min(numpy_array)
        max_val = np.max(numpy_array)

        if np.abs(max_val - min_val) < 1e-6: # Handle case where all values are the same
            image_data_uint8 = np.full(numpy_array.shape, 128, dtype=np.uint8) # Set to a mid-gray value
            print("Warning: Float input data has constant values. Converted to mid-gray uint8.")
        else:
            clipped_array = np.clip(numpy_array, min_val, max_val)
            scaled_array = 255.0 * (clipped_array - min_val) / (max_val - min_val)
            image_data_uint8 = scaled_array.astype(np.uint8)
            print(f"Scaled float input data from [{min_val:.4f}, {max_val:.4f}] to [0, 255] for uint8 conversion.")


    elif numpy_array.dtype == np.uint8:
        image_data_uint8 = numpy_array
    else:
        print(f"Error: Unsupported input dtype: {numpy_array.dtype}. Expected uint8 or float.")
        return None

    image_data_pil_format = np.transpose(image_data_uint8, (1, 2, 0))

    try:
        img = Image.fromarray(image_data_pil_format, 'RGB')
        return img
    except Exception as e:
        print(f"Error creating PIL Image from array: {e}")
        return None


def extract_attention_weights(img_list_tensors, model):

    attention_weights_extracted = []

    model_name = 'google/vit-base-patch16-224'
    model_input_size = (224, 224) # ViT base expects 224x224 input
    print(f"Defined model input size: {model_input_size}")

    hf_model = model

    # Assume hf_model is already loaded and compiled in a previous cell
    if 'hf_model' not in locals() or hf_model is None:
        print("Error: Hugging Face model (hf_model) is not loaded. Please ensure the model loading cell ran successfully.")
        return []

    else:
        print("Hugging Face model is already loaded. Proceeding with tensor preprocessing and attention extraction.")


    for i, image_tensor_input in enumerate(img_list_tensors):

        if not tf.is_tensor(image_tensor_input) or image_tensor_input.ndim != 3:
            print(f"Warning: Skipping input at index {i} of unexpected type {type(image_tensor_input)} or number of dimensions ({image_tensor_input.ndim}). Expected a 3D TensorFlow Tensor.")
            continue #

        input_shape = image_tensor_input.shape
        is_channels_last = input_shape[-1] == 3 and input_shape[0] != 3
        is_channels_first = input_shape[0] == 3 and input_shape[-1] != 3

        if is_channels_first:
            print(f"Input tensor at index {i} detected as Channels First: {input_shape}. Transposing to Channels Last for resizing.")
            image_tensor_processed = tf.transpose(image_tensor_input, perm=(1, 2, 0))
        elif is_channels_last:
            print(f"Input tensor at index {i} detected as Channels Last: {input_shape}.")
            image_tensor_processed = image_tensor_input
        else:
            print(f"Warning: Skipping input at index {i} with ambiguous shape {input_shape}. Expected (H, W, 3) or (3, H, W).")
            continue


        try:

            image_tensor_with_batch = tf.expand_dims(image_tensor_processed, axis=0)


            image_resized_tensor_with_batch = tf.image.resize(
                image_tensor_with_batch,
                size=model_input_size,
                method=tf.image.ResizeMethod.BILINEAR
            )

            image_resized_tensor = tf.squeeze(image_resized_tensor_with_batch, axis=0)
            print(f"Tensor at index {i} resized to model input size: {image_resized_tensor.shape[:2]}")


            if image_resized_tensor.dtype == tf.uint8:
                 image_processed_float = tf.image.convert_image_dtype(image_resized_tensor, dtype=tf.float32) # Scales uint8 to [0, 1]
                 print(f"Tensor at index {i} converted to float32 and scaled to [0, 1].")
            elif image_resized_tensor.dtype == tf.float32 or image_resized_tensor.dtype == tf.float64:

                 if tf.reduce_max(image_resized_tensor) > 1.01:
                      image_processed_float = image_resized_tensor / 255.0
                      print(f"Tensor at index {i} (float > 1) scaled to [0, 1].")
                 else:
                      image_processed_float = image_resized_tensor
                      print(f"Tensor at index {i} (float <= 1) used directly (assuming correct range).")
            else:
                 print(f"Warning: Unsupported dtype for tensor at index {i}: {image_resized_tensor.dtype}. Skipping.")
                 example_image_processed_tf = None
                 continue


            image_processed_with_batch = tf.expand_dims(image_processed_float, axis=0)

            example_image_processed_tf = tf.transpose(image_processed_with_batch, perm=(0, 3, 1, 2))
            print(f"Tensor at index {i} transposed to Channels First for model input. Shape: {example_image_processed_tf.shape}")

            print(f"Tensor at index {i} preprocessed. Final Shape: {example_image_processed_tf.shape}, Dtype: {example_image_processed_tf.dtype}")

        except Exception as e:
            print(f"An error occurred while preprocessing tensor at index {i}: {e}. Skipping.")
            example_image_processed_tf = None


        if example_image_processed_tf is not None:
            try:

                outputs = hf_model(example_image_processed_tf, output_attentions=True, return_dict=True)

                # Attention from the last layer
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    attention_weights = outputs.attentions[-1]
                    print(f"Successfully extracted attention weights for index {i}. Shape: {attention_weights.shape}")
                    print(f"Data type: {attention_weights.dtype}")
                    attention_weights_extracted.append(attention_weights)

                    if attention_weights.ndim == 4 and attention_weights.shape[0] == 1:
                        print("Attention weights shape is as expected (Batch, Num_Heads, Seq_Length, Seq_Length).")
                        num_heads = attention_weights.shape[1]
                        seq_length = attention_weights.shape[2]
                        print(f"  Number of attention heads: {num_heads}")
                        print(f"  Sequence length (including CLS token): {seq_length}")
                    else:
                        print(f"Warning: Attention weights shape for index {i} is unexpected: {attention_weights.shape}")

                else:
                    print(f"Error: Could not access 'attentions' from model outputs for tensor at index {i}.")

            except Exception as e:
                print(f"An error occurred while extracting attention weights for tensor at index {i}: {e}. Skipping.")

        else:
            print(f"\nSkipping attention weight extraction for tensor at index {i} as preprocessing failed.")

    return attention_weights_extracted


def create_heatmap(attention_weights_array, image_array):
    attention_heatmap_array = []

    for index in range(len(attention_weights_array)):
        attention_weights = attention_weights_array[index]
        example_image_np = image_array[index]

        if 'attention_weights' not in locals() or attention_weights is None:
            print("Error: 'attention_weights' not available. Please ensure the previous step to extract attention weights was successful.")
        elif 'example_image_np' not in locals() or example_image_np is None:
            print("Error: 'example_image_np' not available. Please ensure the example image was loaded and processed.")
        else:
            print("Proceeding to process attention weights for heatmap.")

            if not isinstance(attention_weights, tf.Tensor):
                attention_weights_tf = tf.constant(attention_weights)
            else:
                attention_weights_tf = attention_weights

            print(f"Initial attention weights shape: {attention_weights_tf.shape}")

            averaged_attention = tf.reduce_mean(attention_weights_tf, axis=1)
            print(f"Averaged attention shape (after reducing heads): {averaged_attention.shape}")

            attention_no_batch = averaged_attention[0, :, :]
            print(f"Attention shape (after removing batch): {attention_no_batch.shape}")

            cls_attention_to_all = attention_no_batch[0, :]
            print(f"CLS attention to all tokens shape: {cls_attention_to_all.shape}")

            cls_attention_to_patches = cls_attention_to_all[1:]
            print(f"CLS attention to patches shape: {cls_attention_to_patches.shape}")

            actual_seq_length = attention_no_batch.shape[0]
            actual_num_patches = actual_seq_length - 1

            patch_size = 16
            expected_patches_height = 224 // patch_size
            expected_patches_width = 224 // patch_size
            expected_num_patches = expected_patches_height * expected_patches_width

            if actual_num_patches == expected_num_patches:
                 num_patches_height = expected_patches_height
                 num_patches_width = expected_patches_width
                 print(f"Actual number of patches ({actual_num_patches}) matches expected for 224x224 input. Reshaping to ({num_patches_height}, {num_patches_width}).")
            else:
                 side_length = int(np.sqrt(actual_num_patches))
                 if side_length * side_length == actual_num_patches:
                      num_patches_height = side_length
                      num_patches_width = side_length
                      print(f"Actual number of patches ({actual_num_patches}) is a perfect square. Assuming square grid ({num_patches_height}, {num_patches_width}).")
                 else:
                      print(f"Error: Actual number of patches ({actual_num_patches}) does not match expected for 224x224 and is not a perfect square. Cannot confidently reshape to a spatial grid.")
                      num_patches_height = None
                      num_patches_width = None
                      attention_heatmap = None


            if num_patches_height is not None and num_patches_width is not None:
                attention_grid = tf.reshape(cls_attention_to_patches, (num_patches_height, num_patches_width))
                print(f"Attention grid shape: {attention_grid.shape}")

                original_height, original_width, _ = example_image_np.shape
                print(f"Original image size: ({original_height}, {original_width})")

                attention_grid_with_batch = tf.expand_dims(attention_grid, axis=0)
                attention_grid_with_batch = tf.expand_dims(attention_grid_with_batch, axis=-1)
                print(f"Attention grid shape before resize (with batch and channel): {attention_grid_with_batch.shape}")

                resized_heatmap = tf.image.resize(
                    attention_grid_with_batch,
                    size=(original_height, original_width),
                    method=tf.image.ResizeMethod.BILINEAR
                )
                print(f"Resized heatmap shape (with batch and channel): {resized_heatmap.shape}")

                resized_heatmap = tf.squeeze(resized_heatmap, axis=[0, 3])
                print(f"Resized heatmap shape (after removing batch and channel): {resized_heatmap.shape}")

                min_val = tf.reduce_min(resized_heatmap)
                max_val = tf.reduce_max(resized_heatmap)

                if tf.abs(max_val - min_val) < 1e-6:
                    normalized_heatmap = tf.zeros_like(resized_heatmap)
                    print("Warning: Heatmap values are constant. Normalized heatmap set to zeros.")
                else:
                    normalized_heatmap = (resized_heatmap - min_val) / (max_val - min_val)

                print(f"Normalized heatmap shape: {normalized_heatmap.shape}")
                print(f"Normalized heatmap min: {tf.reduce_min(normalized_heatmap).numpy():.4f}, max: {tf.reduce_max(normalized_heatmap).numpy():.4f}")


                attention_heatmap = normalized_heatmap.numpy()
                print("Normalized heatmap created and stored in 'attention_heatmap'.")

            else:
                 print("Skipping heatmap creation due to inability to reshape attention weights.")
                 attention_heatmap = None

        if 'attention_heatmap' in locals() and attention_heatmap is not None:
            print(f"\nFinal 'attention_heatmap' shape: {attention_heatmap.shape}")
            print(f"Final 'attention_heatmap' dtype: {attention_heatmap.dtype}")

            attention_heatmap_array.append(attention_heatmap)

    return attention_heatmap_array


def overlay_heatmap(attention_heatmap_array, original_image_array, type_name):

    # Combine and save heatmap visual

    images_with_heatmap = []

    if not attention_heatmap_array or not original_image_array or len(attention_heatmap_array) != len(original_image_array):
        print("Error: Invalid input for overlay_heatmap. Ensure attention_heatmap_array and original_image_array are non-empty and have the same length.")
        return []

    for index in range(len(attention_heatmap_array)):
        attention_heatmap_np = attention_heatmap_array[index]
        original_image_np_input = original_image_array[index]

        name = f"{type_name}_{index}"

        if attention_heatmap_np is None or not isinstance(attention_heatmap_np, np.ndarray):
            print(f"Warning: Skipping overlay for index {index} ('{name}') due to invalid heatmap array.")
            continue

        if original_image_np_input is None or not isinstance(original_image_np_input, np.ndarray):
            print(f"Warning: Skipping overlay for index {index} ('{name}') due to invalid original image array.")
            continue

        original_shape = original_image_np_input.shape
        is_channels_last = original_shape[-1] == 3 and original_shape[0] != 3
        is_channels_first = original_shape[0] == 3 and original_shape[-1] != 3

        if is_channels_first:
            print(f"Original image array at index {index} ('{name}') detected as Channels First: {original_shape}. Transposing to Channels Last.")
            original_image_np = np.transpose(original_image_np_input, (1, 2, 0))
        elif is_channels_last:
            print(f"Original image array at index {index} ('{name}') detected as Channels Last: {original_shape}.")
            original_image_np = original_image_np_input # Use directly
        else:
            print(f"Warning: Skipping overlay for index {index} ('{name}') due to original image array having ambiguous shape {original_shape}. Expected (H, W, 3) or (3, H, W).")
            continue

        if original_image_np.ndim != 3 or original_image_np.shape[-1] != 3:
            print(f"Warning: Skipping overlay for index {index} ('{name}') due to original image array having unexpected shape {original_image_np.shape} after potential transposition. Expected (H, W, 3).")
            continue # Skip this image


        print(f"Proceeding to overlay heatmap onto original image at index {index} ('{name}').")

        try:
            if original_image_np.dtype == np.float32 or original_image_np.dtype == np.float64:
                 min_val = np.min(original_image_np)
                 max_val = np.max(original_image_np)
                 if np.abs(max_val - min_val) < 1e-6:
                      original_image_uint8 = np.full(original_image_np.shape, 128, dtype=np.uint8) # Mid-gray
                      print(f"Warning: Original image array for index {index} ('{name}') has constant float values. Converted to mid-gray uint8.")
                 else:
                      clipped_array = np.clip(original_image_np, min_val, max_val)
                      scaled_array = 255.0 * (clipped_array - min_val) / (max_val - min_val)
                      original_image_uint8 = scaled_array.astype(np.uint8)
                      print(f"Scaled original image float data for index {index} ('{name}') from [{min_val:.4f}, {max_val:.4f}] to [0, 255] for uint8 conversion.")
            elif original_image_np.dtype == np.uint8:
                 original_image_uint8 = original_image_np
            else:
                 print(f"Warning: Unsupported dtype for original image array at index {index} ('{name}'): {original_image_np.dtype}. Skipping overlay.")
                 continue

            try:
                 original_image_pil = Image.fromarray(original_image_uint8, 'RGB') # Use uint8 array
            except Exception as e:
                 print(f"Error converting original image NumPy array for index {index} ('{name}') to PIL Image: {e}. Skipping overlay.")
                 continue


            heatmap_uint8 = (attention_heatmap_np * 255).astype(np.uint8)
            heatmap_pil_gray = Image.fromarray(heatmap_uint8, 'L')


            colormap = plt.get_cmap('hot')
            heatmap_colored_np = colormap(heatmap_uint8)
            heatmap_colored_np_uint8 = (heatmap_colored_np * 255).astype(np.uint8)
            heatmap_colored_pil = Image.fromarray(heatmap_colored_np_uint8, 'RGBA')


            original_width, original_height = original_image_pil.size
            resized_heatmap_colored_pil = heatmap_colored_pil.resize((original_width, original_height), resample=Image.Resampling.BILINEAR)


            alpha = 0.5 # Opacity of the heatmap layer
            original_image_rgba = original_image_pil.convert('RGBA')
            image_with_heatmap = Image.blend(original_image_rgba, resized_heatmap_colored_pil, alpha)

            images_with_heatmap.append(image_with_heatmap)
            print(f"Overlaying heatmap onto original image at index {index} ('{name}') completed.")


            plt.figure(figsize=(8, 8))
            plt.imshow(image_with_heatmap)
            plt.title(f"Attention Heatmap Overlay: {name}")
            plt.axis('off') # Turn off axis
            plt.show()

            try:
                save_path_overlay = f'/scratch/s3333302/drive_thesis/results/discussion_heatmap/300/heatmap_{name}.png'
                image_with_heatmap.save(save_path_overlay)

                save_path_original = f'/scratch/s3333302/drive_thesis/results/discussion_heatmap/300/original_{name}.png'
                original_image_rgba.save(save_path_original)

                print(f"Saved image with heatmap to: {save_path_overlay}\n Saved original image to: {save_path_original}\n")
            except Exception as e:
                 print(f"Error saving image with heatmap for index {index} ('{name}'): {e}")


        except Exception as e:
            print(f"An error occurred while overlaying heatmap for index {index} ('{name}'): {e}. Skipping.")

    return images_with_heatmap


def create_and_save_heatmap(img_list_original_data, model, type_name):

    original_image_np_list = []
    image_tensor_list_for_extraction = []

    for i, img_data in enumerate(img_list_original_data):
        if isinstance(img_data, np.ndarray):
            original_image_np_list.append(img_data)
            if img_data.ndim == 3 and img_data.shape[-1] == 3:
                 image_tensor_list_for_extraction.append(tf.constant(img_data, dtype=tf.float32))
            else:
                 print(f"Warning: Skipping conversion to tensor for extraction for index {i} due to unexpected NumPy array shape {img_data.shape}.")
        elif tf.is_tensor(img_data):
            try:
                original_image_np_list.append(img_data.numpy())
                image_tensor_list_for_extraction.append(img_data)
            except Exception as e:
                print(f"Error converting TensorFlow tensor at index {i} to NumPy array: {e}. Skipping.")
                original_image_np_list.append(None)
                image_tensor_list_for_extraction.append(None)
        else:
            print(f"Warning: Skipping input at index {i} of unexpected type {type(img_data)}. Expected NumPy array or TensorFlow Tensor.")
            original_image_np_list.append(None)
            image_tensor_list_for_extraction.append(None)

    valid_original_image_np_list = [img_np for img_np in original_image_np_list if img_np is not None]
    valid_image_tensor_list_for_extraction = [img_tensor for img_tensor in image_tensor_list_for_extraction if img_tensor is not None]

    if not valid_image_tensor_list_for_extraction:
         print("Error: No valid image tensors available for attention extraction after processing input list.")
         return

    attention_weights = extract_attention_weights(valid_image_tensor_list_for_extraction, model)

    if not attention_weights or len(attention_weights) != len(valid_original_image_np_list):
        print("Error: Attention weights extraction failed or number of extracted weights does not match number of valid original images.")
        return

    heatmaps = create_heatmap(attention_weights, valid_original_image_np_list)

    if not heatmaps or len(heatmaps) != len(valid_original_image_np_list):
        print("Error: Heatmap creation failed or number of created heatmaps does not match number of valid original images.")
        return

    overlay_heatmap(heatmaps, valid_original_image_np_list, type_name)



for img_type in range(len(types)):
    create_and_save_heatmap(all_images[img_type], hf_model, types[img_type])