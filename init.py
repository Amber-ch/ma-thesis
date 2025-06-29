import numpy as np
import sys
import tensorflow as tf
from tqdm import tqdm
import datetime
import keras
from keras.utils import to_categorical
from keras.models import  Model
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
import h5py

def printn(string):
    sys.stdout.write(string)
    sys.stdout.flush()


def Create_Model():
    img_rows, img_cols = 100, 100
    input_shape = (img_rows, img_cols, 3)

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(2, activation="sigmoid"))
    model.summary()

    return model

def euclidean_distance(vects):
    eps = 1e-08
    x, y = vects
    return sqrt(maximum(sum(square(x - y), axis=1, keepdims=True), eps))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return mean(y_true * square(y_pred) + (1 - y_true) * square(maximum(margin - y_pred, 0)))

def sqrt(x):
    zero = tf.convert_to_tensor(0.0, x.dtype)
    x = tf.maximum(x, zero)
    return tf.sqrt(x)

def maximum(x, y):
    return tf.maximum(x, y)

def sum(x, axis=None, keepdims=False):
    return tf.reduce_sum(x, axis, keepdims)

def square(x):
    return tf.square(x)

def mean(x, axis=None, keepdims=False):
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, backend.floatx())
    return tf.reduce_mean(x, axis, keepdims)


def training_the_model_with_debugging(model, domain_adaptation_task, contrastive_loss_weight=1.0, classification_loss_weight=1.0, epochs=1): # Added loss weight arguments
    nb_classes = 2
    UM = domain_adaptation_task
    batch_size = 64

    if UM == 'high_to_low':
        folder = 'sourcemed_targetlow'
        target = 'lowres'
    else:
        folder = 'sourcelow_targetmed'
        target = 'medres'

    # Define the path to the HDF5 file
    hdf5_file_path = '/Volumes/T7/drive_thesis/thesis/npy/pairs/pairs_optimized/image_pairs.h5'

    def load_batch_from_hdf5(file_path, start_idx, end_idx):
        with h5py.File(file_path, 'r') as hf:
            X1_batch = hf['X1'][start_idx:end_idx]
            X2_batch = hf['X2'][start_idx:end_idx]
            y1_batch = hf['y1'][start_idx:end_idx]
            y2_batch = hf['y2'][start_idx:end_idx]
            yc_batch = hf['yc'][start_idx:end_idx]
        return X1_batch, X2_batch, y1_batch, y2_batch, yc_batch

    def data_generator(file_path, dataset_size, batch_size):

        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)

            X1, X2, y1, y2, yc = load_batch_from_hdf5(file_path, start_idx, end_idx)

            X1 = X1.astype(np.float32) / 255.0
            X2 = X2.astype(np.float32) / 255.0

            if X2.shape[1:3] != (100, 100): # Assuming target size is 100x100
                 X2 = tf.image.resize(X2, (100, 100)).numpy() # Resize using tf.image


            # One-hot encode y1 and y2 if needed for classification loss
            nb_classes = 2
            y1_categorical = tf.keras.utils.to_categorical(y1, nb_classes)
            yc = yc.astype(np.float32)

            yield (X1, X2), (yc, y1_categorical) # Yield inputs and outputs (labels)


    # Get the total number of samples in the dataset
    try:
        with h5py.File(hdf5_file_path, 'r') as hf:
            dataset_size = hf['X1'].shape[0]
        print(f"Dataset size found in HDF5: {dataset_size}")
    except FileNotFoundError:
        print(f"Error: HDF5 file not found at {hdf5_file_path}. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred while getting dataset size: {e}")
        return None

    if dataset_size == 0:
        print("No data found in the HDF5 file. Exiting training.")
        return None


    # Determine output shapes and types for the tf.data.Dataset
    # Load a single batch to infer shapes and types after preprocessing
    try:
        sample_X1, sample_X2, sample_y1, sample_y2, sample_yc = load_batch_from_hdf5(hdf5_file_path, 0, min(batch_size, dataset_size))
        sample_X1 = sample_X1.astype(np.float32) / 255.0
        sample_X2 = sample_X2.astype(np.float32) / 255.0
        if sample_X2.shape[1:3] != (100, 100):
             sample_X2 = tf.image.resize(sample_X2, (100, 100)).numpy()
        nb_classes = 2
        sample_y1_categorical = tf.keras.utils.to_categorical(sample_y1, nb_classes)
        sample_yc = sample_yc.astype(np.float32)

        output_shapes = ((tf.TensorShape([None] + list(sample_X1.shape[1:])),
                          tf.TensorShape([None] + list(sample_X2.shape[1:]))),
                         (tf.TensorShape([None] + list(sample_yc.shape[1:])),
                          tf.TensorShape([None] + list(sample_y1_categorical.shape[1:]))))

        output_types = ((tf.float32, tf.float32),
                        (tf.float32, tf.float32))

        print("\nInferred Dataset output structure:")
        print("output_shapes:", output_shapes)
        print("output_types:", output_types)


    except Exception as e:
        print(f"Error inferring dataset output structure: {e}")
        print("Falling back to manual shape/type definition (adjust if needed).")
        img_rows, img_cols, img_channels = 100, 100, 3
        nb_classes = 2
        output_shapes = ((tf.TensorShape([None, img_rows, img_cols, img_channels]),
                          tf.TensorShape([None, img_rows, img_cols, img_channels])),
                         (tf.TensorShape([None, 1]),
                          tf.TensorShape([None, nb_classes])))

        output_types = ((tf.float32, tf.float32),
                        (tf.float32, tf.float32))


    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(hdf5_file_path, dataset_size, batch_size),
        output_types=output_types,
        output_shapes=output_shapes
    )

    dataset = dataset.shuffle(buffer_size=min(dataset_size, 10 * batch_size)) # Shuffle buffer size
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # Automatically tune the prefetch buffer size

    print("\nTensorFlow Dataset created successfully for training.")
    print("Dataset element spec:", dataset.element_spec)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Or your chosen optimizer

    @tf.function
    def train_step(inputs, outputs):
        x1, x2 = inputs
        yc, y1_cat = outputs

        x1 = tf.cast(x1, tf.float32)
        x2 = tf.cast(x2, tf.float32)
        yc = tf.cast(yc, tf.float32)
        y1_cat = tf.cast(y1_cat, tf.float32)

        with tf.GradientTape() as tape:
            predictions1 = model([x1, x2], training=True)
            distance_pred1 = predictions1[0]
            classification_pred1 = predictions1[1]

            predictions2 = model([x2, x1], training=True)
            distance_pred2 = predictions2[0]
            classification_pred2 = predictions2[1]

            # Calculate mean losses per batch and apply weights
            # Use the loss weights defined outside the train_step
            total_loss = (contrastive_loss_weight * (tf.reduce_mean(contrastive_loss(y_true=yc, y_pred=distance_pred1)) + tf.reduce_mean(contrastive_loss(y_true=yc, y_pred=distance_pred2))) +
                          classification_loss_weight * (tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=y1_cat, y_pred=classification_pred1)) + tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=y2_cat, y_pred=classification_pred2)))) / 2.0


        gradients = tape.gradient(total_loss, model.trainable_variables)

        gradients = [(tf.clip_by_value(grad, clip_value_min=-0.5, clip_value_max=0.5) if grad is not None else None) for grad in gradients]

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return total_loss


    # epochs = 100 # Set a reasonable number of epochs for training
    batch_size = 256

    initial_contrastive_loss_weight = 1.0
    initial_classification_loss_weight = 1.0

    print('Training the model with debugging - Epoch '+str(epochs)) # Updated print statement


    model.compile(
        optimizer=optimizer,
        loss={'distance': contrastive_loss, 'classification': 'categorical_crossentropy'},
        loss_weights={'distance': initial_contrastive_loss_weight, 'classification': initial_classification_loss_weight},
        metrics={'classification': 'accuracy'}
    )


    steps_per_epoch = dataset_size // batch_size
    if dataset_size % batch_size != 0:
        steps_per_epoch += 1

    print(f"Steps per epoch: {steps_per_epoch}")


    for e in tqdm(range(epochs), desc="Training Epochs"):
        epoch_loss = 0
        num_batches = 0

        for step, (inputs, outputs) in enumerate(dataset):
            batch_loss = train_step(inputs, outputs)

            batch_loss_np = batch_loss.numpy()
            scalar_batch_loss = batch_loss_np.item() if batch_loss_np.ndim > 0 else batch_loss_np
            epoch_loss += scalar_batch_loss
            num_batches += 1


        if num_batches > 0:
            average_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {e} - Average Training Loss: {average_epoch_loss:.4f}")
        else:
            print(f"Epoch {e} - No batches processed in this epoch.")


    print(str(epochs) + " epochs completed.")

    # Save the trained model weights
    save_path = f'./trained_model_weights_{UM}.weights.h5'
    try:
        model.save_weights(save_path)
        print(f"\nModel weights saved successfully to {save_path}")
    except Exception as e:
        print(f"\nError saving model weights: {e}")

    return model



def Create_Shared_Branch():
    img_rows, img_cols = 100, 100
    input_shape = (img_rows, img_cols, 3)

    input_layer = Input(shape=input_shape, name='shared_input')

    # Add Convolutional Layer
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same', name='conv1')(input_layer)
    # conv1 output shape: (None, 100, 100, 32)

    # Add another Convolutional Layer
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv2')(conv1)
    # conv2 output shape: (None, 100, 100, 64)

    # Add MaxPooling Layer
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv2)
    # pool1 output shape: (None, 50, 50, 64)

    # Add more Convolutional/Pooling blocks
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv3')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv3)
    # pool2 output shape: (None, 25, 25, 128)

    global_pool = GlobalAveragePooling2D(name='global_avg_pool')(pool2)
    # global_pool output shape: (None, 128)

    dense_embedding = Dense(128, activation='relu', name='dense_embedding')(global_pool)
    # dense_embedding output shape: (None, 128)

    dropout_layer = Dropout(0.5, name='dropout')(dense_embedding)

    final_embedding = dropout_layer

    shared_branch_model = Model(inputs=input_layer, outputs=final_embedding, name='shared_branch_model')

    shared_branch_model.summary()

    return shared_branch_model

def Create_Shared_Branch_CCSA():
    img_rows, img_cols = 100, 100
    input_shape = (img_rows, img_cols, 3)
    nb_filters = 32
    kernel_size = (3, 3)

    input_layer = Input(shape=input_shape, name='shared_input')

    # Add Convolutional layer
    conv1 = Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='valid', name='conv1')(input_layer)

    # Add Activation layer
    activation1 = Activation('relu')(conv1)

    # Add Pooling layer
    maxpool1 = MaxPooling2D(pool_size=(2, 2), name='maxpool1')(activation1)

    # Add Dropout layer
    dropout1 = Dropout(0.25, name='dropout1')(maxpool1)

    # Add another Convolutional layer
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='valid', name='conv2')(dropout1) # Added layer
    # Add another Pooling layer
    maxpool2 = MaxPooling2D(pool_size=(2, 2), name='maxpool2')(conv2) # Added layer
    # Add another Dropout layer
    dropout2 = Dropout(0.25, name='dropout2')(maxpool2) # Added layer


    # Add Flatten layer
    flatten1 = Flatten(name='flatten1')(dropout2) # Changed input to dropout2

    # Add Dense layer
    dense_embedding = Dense(120, name='dense_embedding')(flatten1)

    # Add Activation layer
    activation2 = Activation('relu')(dense_embedding)

    # Add Dense layer
    dense_embedding2 = Dense(84, name='dense_embedding2')(activation2)

    # Add Activation layer
    activation3 = Activation('relu')(dense_embedding2)

    final_embedding = activation3

    shared_branch_model_CCSA = Model(inputs=input_layer, outputs=final_embedding, name='shared_branch_model_CCSA')

    shared_branch_model_CCSA.summary()
    return shared_branch_model_CCSA


def create_classification_model(input_shape, nb_classes):

    # Define the input layer for a single image
    input_img = Input(shape=input_shape, name='input_image')

    # Use the shared branch as the feature extractor
    feature_extractor = Create_Shared_Branch_CCSA()

    # Process the input image through the feature extractor
    extracted_features = feature_extractor(input_img)

    # Add a classification head on top of the extracted features
    classification_output = Dense(nb_classes, activation='sigmoid', name='classification_output')(
        extracted_features)

    # Create the final classification model
    classification_model = Model(inputs=input_img, outputs=classification_output, name='classification_model')

    return classification_model

    img_rows, img_cols = 100, 100
    input_channels = 3
    input_shape = (img_rows, img_cols, input_channels)
    nb_classes = 2

    # Create the classification model
    classification_model = create_classification_model(input_shape, nb_classes)

    classification_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])

    classification_model.summary()

    return classification_model

def train_classification_model(model):
    img_rows, img_cols = 100, 100
    input_channels = 3
    nb_classes = 2

    # Load your training data
    X_train = np.load('/Volumes/T7/drive_thesis//thesis/npy/X_train_300.npy')
    y_train = np.load('/Volumes/T7/drive_thesis/thesis/npy/y_train_300.npy')
    X_test = np.load('/Volumes/T7/drive_thesis//thesis/npy/X_test_300.npy')
    y_test = np.load('/Volumes/T7/drive_thesis//thesis/npy/y_test_300.npy')

    # Preprocess images (normalize and reshape)
    # Apply float32 conversion as discussed earlier for efficiency
    X_train = X_train / 255.0
    X_train = X_train.astype(np.float32)
    # Ensure correct reshaping based on your data dimensions after loading
    # If your raw data is already 4D (samples, height, width, channels), no reshape needed unless dimensions are wrong
    if X_train.ndim == 3:
         X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1) # Assuming grayscale if 3D
    elif X_train.ndim == 4 and X_train.shape[-1] != input_channels:
         X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, input_channels)


    # One-hot encode labels
    y_train_categorical = to_categorical(y_train, nb_classes).astype(np.float32)

    # Check shapes and dtypes
    print("X_train shape:", X_train.shape)
    print("X_train dtype:", X_train.dtype)
    print("y_train_categorical shape:", y_train_categorical.shape)
    print("y_train_categorical dtype:", y_train_categorical.dtype)

    # Preprocess test data
    X_test = X_test / 255.0
    X_test = X_test.astype(np.float32)
    # Ensure correct reshaping based on your data dimensions after loading
    if X_test.ndim == 3:
         X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1) # Assuming grayscale if 3D
    elif X_test.ndim == 4 and X_test.shape[-1] != input_channels:
         X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, input_channels)

    y_test_categorical = to_categorical(y_test, nb_classes).astype(np.float32)

    print("X_test shape:", X_test.shape)
    print("X_test dtype:", X_test.dtype)
    print("y_test_categorical shape:", y_test_categorical.shape)
    print("y_test_categorical dtype:", y_test_categorical.dtype)


    epochs = 1 # Set a reasonable number of epochs for training
    batch_size = 256
    nn = batch_size

    # --- Callbacks Setup ---
    callbacks_list = [] # Initialize an empty list for callbacks

    # TensorBoard Callback (Optional but recommended)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0) # histogram_freq=0 for potentially large images
    callbacks_list.append(tensorboard_callback)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # ModelCheckpoint Callback (Optional but recommended)
    checkpoint_filepath = '/tmp/best_model_checkpoint.weights.h5' # Define where to save the best model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True, # Save only weights to save space
        monitor='val_accuracy', # Monitor validation accuracy
        mode='max', # Save when val_accuracy is maximized
        save_best_only=True) # Only save the best model seen so far
    callbacks_list.append(model_checkpoint_callback)

    # EarlyStopping Callback (Optional but recommended)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', # Monitor validation loss
        patience=5, # Stop after 5 epochs with no improvement in validation loss
        restore_best_weights=True) # Restore the best weights found
    callbacks_list.append(early_stopping_callback)

    # --- Learning Rate Scheduler Callback ---
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', # Metric to monitor (usually validation loss)
        factor=0.1, # Factor by which the learning rate will be reduced. new_lr = old_lr * factor
        patience=3, # Number of epochs with no improvement after which learning rate will be reduced.
        min_lr=0.00001, # Lower bound on the learning rate.
        verbose=1) # Print a message when the learning rate is reduced.
    callbacks_list.append(reduce_lr)
    # --- End Learning Rate Scheduler Setup ---


    # Compile the model
    # Assuming a single classification output model
    model.compile(optimizer=tf.keras.optimizers.Lion(learning_rate=0.001), # Start with an initial learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    print("Starting model training...")
    train = model.fit(X_train, y_train_categorical,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     validation_data=(X_test, y_test_categorical), # Essential for monitoring val_* metrics
                                     callbacks=callbacks_list, # Pass the list of callbacks
                                     verbose=1)

    print("Training finished.")

    # Evaluate the model on the test set (the best weights if using restore_best_weights)
    loss, accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)

    print(f"Test Loss (Best Model): {loss:.4f}")
    print(f"Test Accuracy (Best Model): {accuracy:.4f}")

    # You can optionally load the best weights back manually if not using restore_best_weights=True
    # model.load_weights(checkpoint_filepath)

    return train