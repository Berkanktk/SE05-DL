import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

import model as m
import cleanup_preprocessing as p

# --------------------------------- Build model ---------------------------------
model = m.cnn_model(input_shape=(68, 68, 3))
model.summary()

# --------------------------------- Train model ---------------------------------
print("\nTraining model...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # OG: 1e-4
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

val_preds = model.predict(p.train_images)

history = model.fit(p.x_train_augmented,
                    validation_data=[p.val_images, p.val_labels],
                    epochs=20,
                    verbose=1,
                    class_weight=p.class_weights_dict
                    )
print("Training done!\n")

# --------------------------------- Evaluate model ---------------------------------
print("Performing test..")
results = model.evaluate(x=p.val_images, y=p.val_labels)

for i, metric in enumerate(model.metrics_names):
    print('Final validation {}: {}'.format(metric, results[i]))

# --------------------------------- Plot confusion matrix ---------------------------------
val_pred = np.argmax(val_preds, axis=1)
val_true = np.argmax(p.train_labels, axis=1)

cm = confusion_matrix(val_true, val_pred)
cm = cm / cm.astype(float).sum(axis=1)[:, np.newaxis]

figure = plt.figure(figsize=(8, 8))
sns.heatmap(cm,
            annot=True,
            cmap=plt.cm.Blues,
            xticklabels=p.label_dict.keys(),
            yticklabels=p.label_dict.keys())
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
