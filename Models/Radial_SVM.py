from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import h5py

# Load pretrained model
custom_model = load_model('ResNet50.h5')

# Exclude the output layer to get features
feature_model = Model(inputs=custom_model.input, outputs=custom_model.get_layer('conv5_block3_out').output)

# Initialize your data generator
data_dir = "your_data_directory_here"
image_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator()  # Initialize as needed

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)


# Extract features to disk
def extract_features_to_disk(generator, model, output_file):
    with h5py.File(output_file, 'w') as f:
        features_shape = (generator.samples,) + model.output_shape[1:]
        labels_shape = (generator.samples, generator.num_classes)

        f.create_dataset('features', shape=features_shape, dtype='float32')
        f.create_dataset('labels', shape=labels_shape, dtype='float32')

        for i, (img_batch, label_batch) in enumerate(generator):
            feature_batch = model.predict(img_batch)
            start = i * generator.batch_size
            end = start + feature_batch.shape[0]

            f['features'][start:end] = feature_batch
            f['labels'][start:end] = label_batch
            if i >= len(generator) - 1:
                break


# Extract features
extract_features_to_disk(train_generator, feature_model, 'train_features.h5')

# Load features and labels
with h5py.File('train_features.h5', 'r') as f:
    train_features = np.array(f['features'])
    train_labels = np.array(f['labels'])

train_labels_svm = np.argmax(train_labels, axis=1)
train_features_flat = train_features.reshape((train_features.shape[0], -1))

# Train SVM
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf.fit(train_features_flat, train_labels_svm)

predictions = svm_rbf.predict(train_features_flat)
accuracy = accuracy_score(train_labels_svm, predictions)

print(f"SVM Training Accuracy: {accuracy}")