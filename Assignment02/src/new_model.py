import tensorflow as tf
import cleanup_preprocessing as p
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# --------------------------------- Build model ---------------------------------
input = tf.keras.layers.Input(shape=(68, 68, 3))

# Create the model
model = tf.keras.models.Sequential()

# --------------------------------- Adding layers ---------------------------------
n_layers = 4
n_units = [32, 64, 128, 258, 512]  # 32, 64, 128, 256, 512

model.add(tf.keras.layers.Rescaling(1. / 127.5, offset=-1))

for i in range(n_layers):
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=n_units[i],
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    # model.add(tf.keras.layers.Dropout(rate=0.0))

# Add a global average pooling layer
model.add(tf.keras.layers.Flatten())

# Add a final dense layer with the number of classes
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(7, activation='softmax'))

# --------------------------------- Train model ---------------------------------
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.build(input_shape=(None, 68, 68, 3))
model.summary()

history2 = model.fit(p.x_train_augmented,
                     validation_data=[p.val_images, p.val_labels],
                     epochs=100,
                     verbose=1,
                     class_weight=p.class_weights_dict)

# ---------------------------------- Evaluate model ---------------------------------
results = model.evaluate(x=p.val_images, y=p.val_labels)

for i, metric in enumerate(model.metrics_names):
    print('Final validation {}: {}'.format(metric, results[i]))

# ---------------------------------- Plot confusion matrix ---------------------------------
print("------------------ Confusion matrix for training set ------------------")
train_preds = model.predict(p.train_images)

train_pred = np.argmax(train_preds, axis=1)
train_true = np.argmax(p.train_labels, axis=1)

cm = confusion_matrix(train_true, train_pred)
cm = cm / cm.astype(float).sum(axis=1)[:, np.newaxis]

figure = plt.figure(figsize=(8, 8))
sns.heatmap(cm,
            annot=True,
            cmap=plt.cm.Blues,
            xticklabels=p.label_dict.keys(),
            yticklabels=p.label_dict.keys())
plt.tight_layout(pad=2)
plt.title('Confusion Matrix for Training Set')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("------------------ Confusion matrix for validation set ------------------")
val_preds = model.predict(p.val_images)

val_pred = np.argmax(val_preds, axis=1)
val_true = np.argmax(p.val_labels, axis=1)

cm = confusion_matrix(val_true, val_pred)
cm = cm / cm.astype(float).sum(axis=1)[:, np.newaxis]

figure = plt.figure(figsize=(8, 8))
sns.heatmap(cm,
            annot=True,
            cmap=plt.cm.Blues,
            xticklabels=p.label_dict.keys(),
            yticklabels=p.label_dict.keys())
plt.tight_layout(pad=2)
plt.title('Confusion Matrix for Validation Set')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("------------------ Confusion matrix for test set ------------------")
test_preds = model.predict(p.test_images)

test_pred = np.argmax(test_preds, axis=1)
test_true = np.argmax(p.test_labels, axis=1)

cm = confusion_matrix(test_true, test_pred)
cm = cm / cm.astype(float).sum(axis=1)[:, np.newaxis]

figure = plt.figure(figsize=(8, 8))
sns.heatmap(cm,
            annot=True,
            cmap=plt.cm.Blues,
            xticklabels=p.label_dict.keys(),
            yticklabels=p.label_dict.keys())
plt.tight_layout(pad=2)
plt.title('Confusion Matrix for Test Set')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
# ---------------------------------- Loss history ---------------------------------
# Training loss
print("Training loss")
plt.plot(history2.history['loss'])
plt.title('Loss History for Training Set')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Validation loss
print("Validation loss")
plt.plot(history2.history['val_loss'])
plt.title('Loss History for Validation Set')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# ---------------------------------- Accuracy history ---------------------------------
# Training accuracy
print("Training accuracy")
plt.plot(history2.history['accuracy'])
plt.title('Accuracy History for Training Set')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Val'], loc='upper left')
plt.show()

# Validation accuracy
plt.plot(history2.history['val_accuracy'])
plt.title('Accuracy History for Validation Set')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Val'], loc='upper left')
plt.show()
