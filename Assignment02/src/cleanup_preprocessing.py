import os
import csv
import numpy as np
import tensorflow as tf
import random
import tqdm
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split


"""Eksempel på import af data og konvertering til numpy arrays som kan opereres på i tensorflow/keras efterfulgt af 
træning af et lille CNN som inspiration til brug i assignment 2."""

path = f'{os.getcwd()}/'
print(path)

"""
Definer funktion til at læse data (billeder + labels), konvertere billede til numpy arrays og gemme disse i numpys .npz 
format hvis ikke data.npz filen allerede eksisterer.

I dette eksempel har jeg lagt alle billederne i en samlet mappe, og de er preprocesseret (crop+resize) således de har en 
størrelse på 68x68x3. Billedet kobles til den korrekte klasse vha. af meta-data filen.
"""


def cleanup(wanted=True):
    files = ['train.npz', 'val.npz', 'test.npz']

    if wanted:
        for file in files:
            if os.path.exists('../data/temp/' + file):
                os.remove('../data/temp/' + file)

        print('Cleanup finished!\n')
    else:
        print('Cleanup aborted!\n')


def create_and_load_data(img_path, meta_file, return_dict=False):
    random.seed(420)

    with open(meta_file, newline='') as csv_file:
        meta_list = list(csv.reader(csv_file, delimiter=',', quotechar='|'))[1:]

    random.shuffle(meta_list)

    img_names = [meta[1] for meta in meta_list]
    lesion_types = [meta[2] for meta in meta_list]

    label_dict = {}

    for i, lesion in enumerate(np.unique(lesion_types)):
        label_dict[lesion] = i

    # img_path er stien til mappen med det samlede HAM_10000 datasæt
    # meta_file er metadata-filen i .csv format.
    if os.path.exists('../data/temp/train.npz') and os.path.exists('../data/temp/val.npz') and os.path.exists(
            '../data/temp/test.npz'):
        train_data = np.load('../data/temp/train.npz')
        val_data = np.load('../data/temp/val.npz')
        test_data = np.load('../data/temp/test.npz')
        data = {'train': train_data, 'val': val_data, 'test': test_data}

        if return_dict == False:
            return data
        else:
            return data, label_dict

    arrays = []
    labels = []

    # Load images and labels
    for i, name in tqdm.tqdm(enumerate(img_names)):
        img_array = np.array(tf.keras.utils.load_img(img_path + name + '.jpg', target_size=(68, 68)))
        arrays.append(img_array)
        labels.append(label_dict[lesion_types[i]])

    n_data = len(labels)
    print(f'Number of images: {n_data}')

    arrays = np.array(arrays, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Count number of images in each class
    unique, counts = np.unique(labels, return_counts=True)
    print('\nImages by label:\t', "   ".join(list(label_dict.keys())))
    print("Images by class:\t", counts, "\n")

    # Split data into train, validation and test
    X_train, X_test, Y_train, Y_test = train_test_split(
        arrays, labels, shuffle=True, stratify=labels, test_size=0.1
    )

    X_train, val_images, Y_train, val_labels = train_test_split(
        X_train, Y_train, stratify=Y_train, test_size=0.1
    )

    np.savez('../data/temp/train.npz', arrays=X_train, labels=Y_train)
    np.savez('../data/temp/val.npz', arrays=val_images, labels=val_labels)
    np.savez('../data/temp/test.npz', arrays=X_test, labels=Y_test)

    train_data = np.load('../data/temp/train.npz')
    val_data = np.load('../data/temp/val.npz')
    test_data = np.load('../data/temp/test.npz')

    data = {'train': train_data, 'val': val_data, 'test': test_data}

    if return_dict == False:
        return data
    else:
        return data, label_dict


# --------------------------------- Load or create data ---------------------------------
cleanup(wanted=False)

data, label_dict = create_and_load_data('../data/images/', '../data/HAM10000_metadata.csv', return_dict=True)

train_images, Y_train = data['train']['arrays'], data['train']['labels']
val_images, Y_val = data['val']['arrays'], data['val']['labels']
test_images, Y_test = data['test']['arrays'], data['test']['labels']

train_labels = tf.keras.utils.to_categorical(Y_train, num_classes=7)
val_labels = tf.keras.utils.to_categorical(Y_val, num_classes=7)
test_labels = tf.keras.utils.to_categorical(Y_test, num_classes=7)

print("Total Training:\t\t", train_images.shape, train_labels.shape)
print("Total Validation:\t", val_images.shape, val_labels.shape)
print("Total Test:\t\t\t", test_images.shape, test_labels.shape)

print('\nImages by label:\t', "   ".join(list(label_dict.keys())))
print("Training set:\t\t", np.sum(train_labels, axis=0))
print("Validation set:\t\t", np.sum(val_labels, axis=0))
print("Test set:\t\t\t", np.sum(test_labels, axis=0))

# --------------------------------- Data augmentation ---------------------------------
# Define transformations to apply to the images
transformations = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Use the ImageDataGenerator to generate additional training data
transformations.fit(train_images)

# Apply the transformations to the training data during training randomly
x_train_augmented = transformations.flow(train_images, train_labels, batch_size=32)

# --------------------------------- Class weights ---------------------------------
class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
class_weights_dict = dict(zip(np.unique(Y_train), class_weights))
