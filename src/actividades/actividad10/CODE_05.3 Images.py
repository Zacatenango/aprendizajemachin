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

path = '/home/usuario/programacion/ITESO/semestre3/datasets/doggos/'

#RECOMENDATION!  create a "test" folder to store two folders from Images: n02085620-Chihuahua  and n02085782-Japanese_spaniel as an example.
path = '/home/usuario/programacion/ITESO/semestre3/datasets/doggos_test/'  

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
import requests, base64
from PIL import Image
import json

invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
stream = False

im = Image.open('dog_images/Images/n02088466-bloodhound/n02088466_1262.jpg')
im.save('Bloodhunt.png')

with open("Bloodhunt.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

headers = {
  "Authorization": "Bearer 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",
  "Accept": "text/event-stream" if stream else "application/json"
}

# We will ask the breed of the picture to the model

payload = {
  "messages": [
    {
      "role": "user",
      "content": f'What is the breed of this dog. <img src="data:image/png;base64,{image_b64}" />'
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.20,
  "top_p": 0.70,
  "seed": 0,
  "stream": stream
}

# Use Neva 22b to determine a dog breed
response = requests.post(invoke_url, headers=headers, json=payload)
result = ""
breeds = ["bloodhunt", "bloodhunts"]

# Figure out if we found the correct breed of the dog
if response.status_code == 200:
    json_data = response.json()
    result = json_data['choices'][0]['message']['content'].lower().replace(" ", "")
    
    if any(breed in result.lower().replace(" ", "") for breed in breeds):
        print(result)
    else:
        print("We couldn't determine that the dog was a " + breeds[0])
else:
    print("POST Error failed with code: ", response.status_code)
    
    
#%%
invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
stream = False

im = Image.open('dog_images/Images/n02094433-Yorkshire_terrier/n02094433_2401.jpg')
im.save('Yorkshire.png')

with open("Yorkshire.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

headers = {
  "Authorization": "Bearer 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",
  "Accept": "text/event-stream" if stream else "application/json"
}

# We will ask the breed of the picture to the model
payload = {
  "messages": [
    {
      "role": "user",
      "content": f'What is the breed of this dog. <img src="data:image/png;base64,{image_b64}" />'
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.20,
  "top_p": 0.70,
  "seed": 0,
  "stream": stream
}

# Use Neva 22b to determine a dog breed
response = requests.post(invoke_url, headers=headers, json=payload)
result = ""
breeds = ["yorkshire", "yorkshires", "yorkie", "terrier"]

# Figure out if we found the correct breed of the dog
if response.status_code == 200:
    json_data = response.json()
    result = json_data['choices'][0]['message']['content'].lower().replace(" ", "")
    
    if any(breed in result.lower().replace(" ", "") for breed in breeds):
        print(result)
    else:
        print("We couldn't determine that the dog was a " + breeds[0])
else:
    print("POST Error failed with code: ", response.status_code)
    
    
#%%
import requests, base64
from PIL import Image

invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2"

with open("Bloodhunt.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

headers = {
  "Authorization": "Bearer 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",
  "Accept": "application/json"
}

# We will ask the breed of the picture to the model
payload = {
  "messages": [
    {
      "role": "user",
      "content": f'What is the breed of this dog. <img src="data:image/png;base64,{image_b64}" />'
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.20,
  "top_p": 0.20,
}

# Use Neva 22b to determine a dog breed
response = requests.post(invoke_url, headers=headers, json=payload)
result = ""
breeds = ["bloodhount", "bloodhounds"]

# Figure out if we found the correct breed of the dog
if response.status_code == 200:
    json_data = response.json()
    result = json_data['choices'][0]['message']['content']

    if any(breed in result.lower().replace(" ", "") for breed in breeds):
        print(result)
    else:
        print("We couldn't determine that the dog was a " + breeds[0])
else:
    print("POST Error failed with code: ", response.status_code)
    
#%%
import requests, base64
from PIL import Image

invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2"

with open("Yorkshire.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

headers = {
  "Authorization": "Bearer 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",
  "Accept": "application/json"
}

# We will ask the breed of the picture to the model
payload = {
  "messages": [
    {
      "role": "user",
      "content": f'What is the breed of this dog. <img src="data:image/png;base64,{image_b64}" />'
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.20,
  "top_p": 0.20,
}

# Use Neva 22b to determine a dog breed
response = requests.post(invoke_url, headers=headers, json=payload)
result = ""
breeds = ["yorkshire", "yorkshires", "yorkie", "terrier"]

# Figure out if we found the correct breed of the dog
if response.status_code == 200:
    json_data = response.json()
    result = json_data['choices'][0]['message']['content']

    if any(breed in result.lower().replace(" ", "") for breed in breeds):
        print(result)
    else:
        print("We couldn't determine that the dog was a " + breeds[0])
else:
    print("POST Error failed with code: ", response.status_code)