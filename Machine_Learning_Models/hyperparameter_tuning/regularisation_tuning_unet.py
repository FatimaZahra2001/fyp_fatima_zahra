import os
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, structural_similarity
from tensorflow.image import ssim
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers, regularizers

class unet_sinogram(tf.keras.Model):
    def __init__(self, width=128, depth=5, dropout_rate=0.1, l2_lambda=0.001):
        super(unet_sinogram, self).__init__()
        self.width = width
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        # Define U-Net model architecture for sinograms
        self.inputs = tf.keras.Input(shape=(128, 128, 1))
        x = self.inputs

        # Encoder
        for _ in range(depth):
            x = layers.Conv2D(width, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Bottleneck
        x = layers.Conv2D(width, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(x)

        # Decoder
        for _ in range(depth):
            x = layers.Conv2DTranspose(width, 2, strides=(2, 2), padding='same')(x)
            x = layers.Conv2D(width, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        self.outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)

        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name=f"sinogram_unet_{width}_{depth}")

    def call(self, inputs):
        return self.model(inputs)

    def compile_model(self, optimizer, loss_function, metrics):
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    def train_model(self, train_input_images, train_output_images, epochs, batch_size, validation_split):
        history = self.model.fit(train_input_images, train_output_images, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
        self.history['loss'].extend(history.history['loss'])
        self.history['accuracy'].extend(history.history['accuracy'])
        self.history['val_loss'].extend(history.history['val_loss'])
        self.history['val_accuracy'].extend(history.history['val_accuracy'])

        self.save_metrics()

    def evaluate_model(self, test_input_images, test_output_images):
        return self.model.evaluate(test_input_images, test_output_images, verbose=0)

    def save_metrics(self):
        with open("training_metrics.txt", "w") as file:
            file.write("Epoch\tLoss\tAccuracy\tVal_Loss\tVal_Accuracy\n")
            for i in range(len(self.history['loss'])):
                file.write(f"{i+1}\t{self.history['loss'][i]}\t{self.history['accuracy'][i]}\t{self.history['val_loss'][i]}\t{self.history['val_accuracy'][i]}\n")

    def calculate_metrics(self, test_input_images, test_output_images):
        mse_values = []
        ssim_values = []
        psnr_values = []

        for i in range(len(test_input_images)):
            input_image = test_input_images[i]
            output_image = test_output_images[i]
            predicted_image = self.model.predict(np.expand_dims(input_image, axis=0))[0]

            mse = mean_squared_error(output_image, predicted_image)
            ssim_score = structural_similarity(output_image, predicted_image, data_range=predicted_image.max() - predicted_image.min())
            psnr = tf.image.psnr(output_image, predicted_image, max_val=1.0)

            mse_values.append(mse)
            ssim_values.append(ssim_score)
            psnr_values.append(psnr)

        return mse_values, ssim_values, psnr_values

# custom loss functions
def ssim_loss(y_true, y_pred):
    return 1 - K.mean(ssim(y_true, y_pred, max_val=1.0))

def psnr_loss(y_true, y_pred):
    return -K.mean(10.0 * K.log(K.square(1.0) / (K.square(y_pred - y_true) + K.epsilon())) / K.log(10.0))
def load_and_preprocess_sinograms(input_folder, output_folder):
    input_images = []
    output_images = []
    
    input_filenames = sorted(os.listdir(input_folder))
    output_filenames = sorted(os.listdir(output_folder))
    
    for input_filename, output_filename in zip(input_filenames, output_filenames):
        input_img_path = os.path.join(input_folder, input_filename)
        output_img_path = os.path.join(output_folder, output_filename)
        
        input_img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
        output_img = cv2.imread(output_img_path, cv2.IMREAD_GRAYSCALE)
        
        if input_img is not None and output_img is not None:
            input_img = cv2.resize(input_img, (128, 128))  
            output_img = cv2.resize(output_img, (128, 128)) 
            
            input_img = input_img.astype('float32') / 255.0 
            output_img = output_img.astype('float32') / 255.0  
            
            input_images.append(input_img)
            output_images.append(output_img)
    
    return np.array(input_images), np.array(output_images)

input_folder = '/jupyter/work/fyp/data/sinograms/3rd_set/4'
output_folder = '/jupyter/work/fyp/data/sinograms/3rd_set/16'

input_images, output_images = load_and_preprocess_sinograms(input_folder, output_folder)

train_input_images, test_input_images, train_output_images, test_output_images = train_test_split(
    input_images, output_images, test_size=0.2, random_state=42
)

# unet_model = unet_sinogram()
# unet_model.compile_model(optimizer=Adam(learning_rate=0.001), loss_function=psnr_loss, metrics=['accuracy'])

# unet_model.train_model(train_input_images, train_output_images, epochs=200, batch_size=32, validation_split=0.2)

# unet_model.model.save("unet_sinogram_model.h5")

dropout_rates = [0.1, 0.2, 0.3]  
l2_lambda_values = [0.001, 0.01, 0.1]  
best_val_loss = float('inf')
best_dropout_rate = None
best_l2_lambda = None

for dropout_rate in dropout_rates:
    for l2_lambda in l2_lambda_values:
        # compile the model with current hyperparameters
        model = unet_sinogram(dropout_rate=dropout_rate, l2_lambda=l2_lambda)
        model.compile_model(optimizer=Adam(learning_rate=0.001), loss_function=psnr_loss, metrics=['accuracy'])  # Compile the model
        
        # Train the model
        model.train_model(train_input_images, train_output_images, epochs=20, batch_size=32, validation_split=0.2)
        
        val_loss = model.evaluate_model(train_input_images, train_output_images)[0]
        
        # Check if current hyperparameters result in better validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dropout_rate = dropout_rate
            best_l2_lambda = l2_lambda


print(f"Best Dropout Rate: {best_dropout_rate}, Best L2 Lambda: {best_l2_lambda}, Best Validation Loss: {best_val_loss}")