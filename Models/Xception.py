# Model Setup and Training for Xception with pre-trained weights
# REQUIRED TENSORFLOW VERSION: 2.9.0 all other libraries are the latest versions as of 9/29/2023
# pip install tensorflow==2.9.0

# Import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
import os

# Extract training data from ZIP file
with ZipFile("/root/Train.zip", 'r') as zf:
    zf.extractall("/root/Zipped")

# Extract test data from ZIP file
with ZipFile("/root/test.zip", 'r') as zf:
    zf.extractall("/root/Zipped")

# Display the number of images in each label folder
augmented_images_directory = r"/root/Zipped/Train"
label_folders = os.listdir(augmented_images_directory)
for label_folder in label_folders:
    folder_path = os.path.join(augmented_images_directory, label_folder)
    num_images = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
    print(f"Label Folder: {label_folder} | Number of Images: {num_images}")

# Set the image dimensions and the batch size
image_size = (299, 299)  # Xception's original input size
batch_size = 32

# Load the pre-trained Xception model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Modify the model architecture
x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
prediction = Dense(69, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=prediction)

# Setup data generators for training and validation
train_datagen = ImageDataGenerator(validation_split=0.1)
data_dir = r"/root/Zipped/Train"
train_generator = train_datagen.flow_from_directory(data_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training', shuffle=True)
valid_generator = train_datagen.flow_from_directory(data_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation')

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size, validation_data=valid_generator, validation_steps=valid_generator.samples // batch_size, epochs=120, workers=4, verbose=1)

# Setup data generator for testing and evaluate the model
test_dir = r"/root/Zipped/test"
test_datagen = ImageDataGenerator()
eval_generator = test_datagen.flow_from_directory(test_dir, target_size=image_size, class_mode='categorical')
evaluation = model.evaluate(eval_generator, verbose=1, workers=1)
print('Test loss:', evaluation[0])
print('Test accuracy:', evaluation[1])

# Save the model with the specified name
model.save('Xception.h5')
