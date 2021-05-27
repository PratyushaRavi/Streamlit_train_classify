import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.header(" Teachable Image classifier")

st.sidebar.markdown(" ## Upload the data here  ")

class_no = 2

@st.cache
def prep_data(images,cls):
    train_img = []
    train_label = []
    for index, image in enumerate(images):
        image = Image.open(image)
        img_array = np.array(image.getdata())
        image = np.resize(img_array, (224, 224, 3))
        train_img.append(image)
        train_label.append(cls)

    train_img = np.array(train_img).reshape(-1, 224, 224, 3)
    train_label = np.array(train_label)

    train_img = train_img / 255
    return train_img, train_label

@st.cache
def shuffle_data(train1,label1,train0,label0):
    X = np.append(train0, train1, axis=0)
    y = np.append(label0, label1, axis=0)
    # shuffle data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    return X,y



uploaded_img0 = st.sidebar.file_uploader("Class 0", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
uploaded_img1 = st.sidebar.file_uploader("Class 1", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)



@st.cache
def create_train_model(train_img,train_label):
    base_model = VGG16(input_shape=(224, 224, 3),  # Shape of our images
                       include_top=False,  # Leave out the last fully connected layer
                       weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(base_model.input, x)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])

    model.fit(train_img, train_label, epochs=10)
    return model


st.markdown("Want to train an image classifier without usinf any code and instantenously use it to classify unseen images? We got you covered!")
st.markdown("You can upload the images in the sidebar")



prep = st.checkbox("Prep And Train")
if prep:
    X0, y0 = prep_data(uploaded_img0, 0)
    X1, y1 = prep_data(uploaded_img1, 1)
    X, y = shuffle_data(X1, y1, X0, y0)
    model = create_train_model(X, y)
    st.success("Done")


img_uploaded = st.file_uploader("", type=['jpg', 'jpeg','png'], accept_multiple_files=False)
if img_uploaded is not None:
    to_predict = Image.open(img_uploaded)
    st.image(to_predict)

@st.cache
def predict(to_pred):
    img_array = np.array(to_pred.getdata())
    image = np.resize(img_array, (224, 224, 3))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image / 255
    yhat = model.predict(image)

    return yhat


out = st.checkbox("Predict")
if out:
    ans = predict(to_predict)
    if ans > 0.5:
         st.write(" Predicted class is : Class 1")
    else:
        st.write("Predicted class is : Class 0")
