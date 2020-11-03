import streamlit as st
import numpy as np
from PIL import Image
import random
import tensorflow as tf
from tensorflow.keras import layers
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16

st.header(" My teachable Machine clone")
st.sidebar.markdown(" Upload your datasets here",)

X = []
y = []


for x in range(2):
    images = st.sidebar.file_uploader(str(x), type=['jpg', 'jpeg','png'], accept_multiple_files=True)
    if images is not None:
        for index, image in enumerate(images):
            image = Image.open(image)
            img_array = np.array(image.getdata())
            image = np.resize(img_array, (224, 224, 3))
            X.append(image)
            y.append(x)


@st.cache
def prep_data(train_img,train_label):
    c = list(zip(train_img, train_label))
    random.shuffle(c)
    train_img, train_label = zip(*c)

    train_img = np.array(X).reshape(-1, 224, 224, 3)
    train_label = np.array(train_label)

    train_img = train_img / 255

    return train_img, train_label


prep = st.checkbox("Start preparing data")
if prep:
    X, y = prep_data(X, y)


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

    (model.fit(train_img, train_label, epochs=10))
    return model



train = st.checkbox("Start Training model")
if train:
    model = create_train_model(X,y)

st.markdown("Upload images to get prediction")

img_uploaded = st.file_uploader("", type=['jpg', 'jpeg','png'], accept_multiple_files=False)
if img_uploaded is not None:
    to_predict = Image.open(img_uploaded)
    st.image(to_predict)

def predict(image1):
    image = load_img(image1, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = image / 255
    # predict the probability across all output classes
    yhat = model.predict(image)

    return yhat



out = st.checkbox("Predict")
if out:
    ans = predict(img_uploaded)
    if ans > 0.5:
        st.write("Dog")
    else:
        st.write("Cat")

