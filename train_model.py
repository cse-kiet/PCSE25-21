import tensorflow as tf
from tensorflow.keras import layers, models

# Paths to your dataset folders
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Load datasets from folders
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    shuffle=True,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    shuffle=False,
)

# Get class names (emotion labels)
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Normalize pixel values to [0,1]
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Build the CNN model
model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax'),
])

model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
epochs = 15
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs,
)

# Save the trained model
model.export('fer2013_emotion_model')

print("Training complete! Model saved as fer2013_emotion_model.h5")
