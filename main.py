import os

#path,dirs,files = next(os.walk("C:\\Users\\MIHIR\\Downloads\\dogs-vs-cats\\train\\train"))
#fname = os.listdir("C:\\Users\\MIHIR\\Downloads\\dogs-vs-cats\\train\\train")
#print(fname)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from sklearn.model_selection import train_test_split
from PIL import Image

#img = mimage.imread("C:\\Users\\MIHIR\\Downloads\\dogs-vs-cats\\train\\train")
 
img = mimage.imread("C:\\Users\\MIHIR\\Desktop\\cat and dog DL\\dogs-vs-cats\\train\\train\\cat.4.jpg")
imgplt = plt.imshow(img)
plt.show()

fname = os.listdir("C:\\Users\\MIHIR\\Desktop\\cat and dog DL\\dogs-vs-cats\\train\\train")
for i in range(5):
    print(fname[i])

# Resizing all the images

original_folder = "C:\\Users\\MIHIR\\Desktop\\cat and dog DL\\dogs-vs-cats\\train\\train\\"
resized_folder = "C:\\Users\\MIHIR\\Desktop\\cat and dog DL\\dogs-vs-cats\\resized train\\"

# for i in range(2000):
#     filename = os.listdir(original_folder)[i]
#     img_path = original_folder+filename

#     img = Image.open(img_path)
#     img = img.resize((224, 224))
#     img = img.convert('RGB')

#     newImgPath = resized_folder+filename
#     img.save(newImgPath)

#   # display resized dog image
# img = mimage.imread("C:\\Users\\MIHIR\\Desktop\\cat and dog DL\\dogs-vs-cats\\resized train\\cat.0.jpg")
# imgplt = plt.imshow(img)
# plt.show()


################################################################

# Creating labels for resized images of dogs and cats

# creaing a for loop to assign labels
filenames = os.listdir("C:\\Users\\MIHIR\\Desktop\\cat and dog DL\\dogs-vs-cats\\resized train\\")

labels = []

for i in range(2000):

  file_name = filenames[i]
  label = file_name[0:3]

  if label == 'dog':
    labels.append(1)

  else:
    labels.append(0)

#print(filenames[0:5])
#print(len(filenames))
#
#print(labels[0:5])
#print(len(labels))

# counting the images of dogs and cats out of 2000 images
values, counts = np.unique(labels, return_counts=True)
#print(values)
#print(counts)

#########################################

# Converting all the resized images to numpy arrays


import cv2
import glob

image_directory = "C:\\Users\\MIHIR\\Desktop\\cat and dog DL\\dogs-vs-cats\\resized train\\"
image_extension = ['png', 'jpg']

files = []

[files.extend(glob.glob(image_directory + '*.' + e)) for e in image_extension]

dog_cat_images = np.asarray([cv2.imread(file) for file in files])

# print(dog_cat_images)

#print(dog_cat_images.shape)

X = dog_cat_images
Y = np.asarray(labels)

# Train Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#print(X.shape, X_train.shape, X_test.shape)

# 1600 --> training images
# 400 --> test images
# as we can see that theere is a large  diff between traning and testing data,
# therefore we are scaling the data 

# scaling the data
X_train_scaled = X_train/255

X_test_scaled = X_test/255

#print(X_train_scaled)

# Building the Neural Network

import tensorflow as tf
import tensorflow_hub as hub

mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224,224,3), trainable=False)

num_of_classes = 2

model = tf.keras.Sequential([
    
    pretrained_model,
    tf.keras.layers.Dense(num_of_classes)

])

model.summary()

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['acc']
)

model.fit(X_train_scaled, Y_train, epochs=5)

score, acc = model.evaluate(X_test_scaled, Y_test)
#print('Test Loss =', score)
#print('Test Accuracy =', acc)

# Predictive System
input_image_path = input('Path of the image to be predicted: ')

input_image = cv2.imread(input_image_path)

cv2.imshow(input_image)

input_image_resize = cv2.resize(input_image, (224,224))

input_image_scaled = input_image_resize/255

image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])

input_prediction = model.predict(image_reshaped)

print(input_prediction)

input_pred_label = np.argmax(input_prediction)

print(input_pred_label)

if input_pred_label == 0:
  print('The image represents a Cat')

else:
  print('The image represents a Dog')







