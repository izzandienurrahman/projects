import gradio as gr
from tensorflow import keras
import numpy as np
import joblib

# load target class
with open('../models/class_names.pkl','rb') as file_1:
    target_classes = joblib.load(file_1)

# load CNN model
model = keras.models.load_model('../models/brain_tumor_model_2.h5')

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
    confidences = {target_classes[i]: float(pred_proba[0][i]) for i in range(len(target_classes))}
    return confidences


img_inputs = gr.inputs.Image(shape=(256, 256), label="Upload Image:")

interface = gr.Interface(
    fn=infer,
    inputs=img_inputs,
    outputs=gr.Label(num_top_classes=4),
    title="Brain Tumor Classification",
    description="Predicts if the MRI scanned brain image belongs to one of these classes: Glioma Tumor, Meningioma Tumor, Pituitary Tumor, or No Tumor"
)

interface.launch(share=False)
