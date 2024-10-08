import keras
from keras import datasets, models, layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1, ))

x_train = x_train / 255.
x_test = x_test / 255.

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Define model
model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
    layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

# Train model and capture history
history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Evaluate model on test set
loss, accuracy = model.evaluate(x_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Plot Loss and Accuracy curves
plt.figure(figsize=(12, 4))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Generate Confusion Matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))

# Plot Confusion Matrix
plt.figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Save the model
# model.save("CNN-DigitRecog-V5.h5")
