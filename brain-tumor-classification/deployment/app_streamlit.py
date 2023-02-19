import streamlit as st
from tensorflow import keras
import numpy  as np
import PIL
import joblib

# load target class
with open('../models/class_names.pkl','rb') as file_1:
    target_classes = joblib.load(file_1)

@st.cache_resource
def load_model(path):
    model = keras.models.load_model(path)
    return model

# load CNN model
model = load_model('../models/brain_tumor_baseline.h5')

# preprocess function
def preprocess(img):
    img = np.asarray(img) # convert to numpy array
    preprocessed_img = np.expand_dims(img,axis=0) # add batch dimension
    return preprocessed_img

# inference function
def infer(img):
    # preprocess input
    preprocessed_img = preprocess(img) # resize to 256x256 (model only accept those size)

    # predict result with best model
    pred_proba = model.predict(preprocessed_img)
    idx_class = np.argmax(pred_proba,axis=1)
    return idx_class



# header deployment
st.header("Brain Tumor Classification")
st.write("Predicts if the MRI scanned brain image belongs to one of these classes:")
st.markdown("**Glioma Tumor**, **Meningioma Tumor**, **Pituitary Tumor**, or **No Tumor**")

# upload image
uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False)

c1, c2 = st.columns(2)
c1.header("Input Image:")
c2.header(f"Prediction:")

if uploaded_file is not None:
    img = PIL.Image.open(uploaded_file)
    img = img.resize((256,256))
    c1.image(uploaded_file)
    
if st.button("Predict"):
    # call inference function
    pred_class = infer(img)

    # show prediction results
    for idx in pred_class:
        c2.write(target_classes[idx])

