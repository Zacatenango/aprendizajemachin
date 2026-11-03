# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 16:59:24 2025

@author: zaratejo
"""

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense



from keras.preprocessing import image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

path = '/home/usuario/programacion/ITESO/semestre3/datasets/doggos/images/Images'

#RECOMENDATION!  create a "test" folder to store two folders from Images: n02085620-Chihuahua  and n02085782-Japanese_spaniel as an example.
path = '/home/usuario/programacion/ITESO/semestre3/datasets/doggos_test/images/Images'  

images = [] # Images of all breeds
labels = [] # Labels to indetify the breed
breed_to_index = {}
index = 0
directory=path
for breed in os.listdir(directory):
    breed_to_index[breed] = index
    breed_path = os.path.join(directory, breed)
    for img_file in os.listdir(breed_path):
        img_path = os.path.join(breed_path, img_file)
         # Resize and normalize the images
        img = image.load_img(img_path, target_size=(100, 100)) 
        img = image.img_to_array(img)
        img = img / 255.0
        # Append the images using the index of the breed
        images.append(img)
        labels.append(breed)
    index += 1
#%%
# Process images from the breed directory
def preprocessing(directory):
    images = [] # Images of all breeds
    labels = [] # Labels to indetify the breed
    breed_to_index = {}
    index = 0

    for breed in os.listdir(directory):
        breed_to_index[breed] = index
        breed_path = os.path.join(directory, breed)
        for img_file in os.listdir(breed_path):
            img_path = os.path.join(breed_path, img_file)
             # Resize and normalize the images
            img = image.load_img(img_path, target_size=(100, 100)) 
            img = image.img_to_array(img)
            img = img / 255.0
            # Append the images using the index of the breed
            images.append(img)
            labels.append(breed)
        index += 1

    return np.array(images), np.array(labels)

images, labels = preprocessing(path)
#%%
def plot_top_breeds_distribution(labels, num_breeds=25):
    unique_breeds, counts = np.unique(labels, return_counts=True)
    # Sort breeds by frequency in descending order
    sorted_indices = np.argsort(-counts) 
    top_breeds = unique_breeds[sorted_indices][:num_breeds]
    top_counts = counts[sorted_indices][:num_breeds]

    plt.figure(figsize=(12, 6))
    plt.bar(top_breeds, top_counts)
    plt.title('Distribution of Top 25 Dog Breeds')
    plt.xlabel('Dog Breed')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=90)
    plt.show()

# Plot breed distribution
plot_top_breeds_distribution(labels)
#%%
def plot_sample_images_top_breeds(images, labels, num_samples=3, num_breeds=3):
    unique_breeds, counts = np.unique(labels, return_counts=True)
    # Sort breeds by frequency in descending order
    sorted_indices = np.argsort(-counts)
    top_breeds = unique_breeds[sorted_indices][:num_breeds]

    plt.figure(figsize=(15, 8))
    for i, breed in enumerate(top_breeds):
        breed_images = images[labels == breed]
        for j in range(num_samples):
            plt.subplot(num_samples, num_breeds, j * num_breeds + i + 1)
            plt.imshow(breed_images[j])
            plt.title(breed)
            plt.axis('off')
    plt.suptitle('Sample Images from Top 3 Dog Breeds')
    plt.show()

plot_sample_images_top_breeds(images, labels)
#%%
# Shuffle images
indexes = np.arange(len(images))
np.random.shuffle(indexes)
images = images[indexes]
labels = labels[indexes]

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)
#%%

# Assuming 'labels' is a list or array of labels for the dataset
labels = [0, 1, 2]  # Example labels, replace with actual labels

# Define the model
model = Sequential([
    # Input layer: Specify the shape of the input data (100x100 RGB image)
    Input(shape=(100, 100, 3)),  # input_shape specifies the shape of the input data (100x100 RGB image)
    #HIGHLY RECOMMENDED TO ADD CONVOLUTION HIDDEN LAYER ( UNCOMMENTLINES BELOW)
    #Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)), 
    Flatten(),  # Flatten layer to convert 2D image to 1D array. Flatten layer: Convert the 2D image to a 1D array
        
    # Hidden layer: Fully connected layer with 128 neurons and ReLU activation function
    Dense(10, activation='relu'),  # 100 is the number of neurons in the hidden layer, 'relu' is the activation function
    
    # Output layer: Fully connected layer with number of neurons equal to the number of unique labels and softmax activation function
    Dense(len(np.unique(labels)), activation='softmax')  # len(np.unique(labels)) gives the number of unique classes, 'softmax' is used for multi-class classification
])

# Summary of the model
model.summary()


# Compile the model
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

test_p = model.predict(X_test)
y_pred = np.argmax(test_p, axis=1)

# Measure the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Test: {accuracy:.2f}')



#%%

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)), Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),  Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)), Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(labels)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

test_p = model.predict(X_test)
y_pred = np.argmax(test_p, axis=1)

# Measure the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Test: {accuracy:.2f}')

#%%
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(16, activation='linear'),
    Dense(len(np.unique(labels)), activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

test_p = model.predict(X_test)
y_pred = np.argmax(test_p, axis=1)

# Measure the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Test: {accuracy:.2f}')

#%%
model = Sequential([
    Conv2D(32, (3, 3), activation='tanh', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='tanh'),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='tanh'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(len(np.unique(labels)), activation='relu')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

test_p = model.predict(X_test)
y_pred = np.argmax(test_p, axis=1)

# Measure the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Test: {accuracy:.2f}')


#%%
# Ask vision-language models (VLMs) hosted by NVIDIA for the breed of a dog.
#
# The old per-model endpoints (https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b
# and .../microsoft/kosmos-2) are deprecated. NVIDIA now serves its models through
# one OpenAI-compatible endpoint: you pick the model with the "model" field and
# send the image as an "image_url" content part instead of an <img> HTML tag.
#
# Get a key at https://build.nvidia.com and set it before running, e.g.:
#   export NVIDIA_API_KEY="nvapi-..."
import os
import base64
import requests

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# Replacements for NeVA-22B and Kosmos-2. Any vision model listed at
# https://integrate.api.nvidia.com/v1/models works here.
MODEL_A = "meta/llama-3.2-11b-vision-instruct"
MODEL_B = "meta/llama-3.2-90b-vision-instruct"

DATASETS_DIR = '/home/usuario/programacion/ITESO/semestre3/datasets'
BLOODHOUND_IMG = os.path.join(DATASETS_DIR, 'doggos_test/images/Images/n02088466-bloodhound/n02088466_1262.jpg')
# The Yorkshire terrier folder is only in the full dataset, not in doggos_test
YORKSHIRE_IMG = os.path.join(DATASETS_DIR, 'doggos/images/Images/n02094433-Yorkshire_terrier/n02094433_2401.jpg')

BLOODHOUND_BREEDS = ["bloodhound", "bloodhounds"]
YORKSHIRE_BREEDS = ["yorkshire", "yorkshires", "yorkie", "terrier"]


def ask_breed(model, image_path, breeds):
    """Ask `model` for the dog's breed and check whether the answer names one of `breeds`."""
    if not NVIDIA_API_KEY:
        raise RuntimeError("Set the NVIDIA_API_KEY environment variable first.")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }

    # We will ask the breed of the picture to the model
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the breed of this dog?"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.70,
        "seed": 0,
        "stream": False,
    }

    response = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=120)

    # Figure out if we found the correct breed of the dog
    if response.status_code != 200:
        print(f"[{model}] POST failed with code {response.status_code}: {response.text}")
        return None

    result = response.json()['choices'][0]['message']['content']
    normalized = result.lower().replace(" ", "")
    if any(breed in normalized for breed in breeds):
        print(f"[{model}] {result}")
    else:
        print(f"[{model}] We couldn't determine that the dog was a {breeds[0]}. Answer: {result}")
    return result


#%%
# Model A on the bloodhound
ask_breed(MODEL_A, BLOODHOUND_IMG, BLOODHOUND_BREEDS)

#%%
# Model A on the Yorkshire terrier
ask_breed(MODEL_A, YORKSHIRE_IMG, YORKSHIRE_BREEDS)

#%%
# Model B on the bloodhound
ask_breed(MODEL_B, BLOODHOUND_IMG, BLOODHOUND_BREEDS)

#%%
# Model B on the Yorkshire terrier
ask_breed(MODEL_B, YORKSHIRE_IMG, YORKSHIRE_BREEDS)
