# import needed libraries
import streamlit as st
import pandas as pd
import numpy  as np
import joblib

# load best model
with open('gbr.pkl','rb') as file_1:
    gbr_model = joblib.load(file_1)

# define inference function
def infer(data_infer):
    # predict result with pre-trained model
    pred = gbr_model.predict(data_infer)
    return pred

# header
st.header("Predicting Plant Nutritional Value")

# artificial data infer
sample_type = st.selectbox("Please specify where you obtained this sample: ", ["lab 1", "lab 2"])
v1          = st.slider("Please input v1 component values: ",0,1000, key='v1')
v2          = st.slider("Please input v2 component values: ",0,1000, key='v2')    
v3          = st.slider("Please input v3 component values: ",0,1000, key='v3')    
v4          = st.slider("Please input v4 component values: ",0,1000, key='v4')    
v5          = st.slider("Please input v5 component values: ",0,1000, key='v5')
v6          = st.slider("Please input v6 component values: ",0,1000, key='v6')    
v7          = st.slider("Please input v7 component values: ",0,1000, key='v7')    
v8          = st.slider("Please input v8 component values: ",0,1000, key='v8')  

if st.button("Submit"):
# assign data inference
    D = {
        'v1':v1,
        'v2':v2,
        'v3':v3,
        'v4':v4,
        'v5':v5,
        'v6':v6,
        'v7':v7,
        'v8':v8,
        'sample_type':sample_type
    }
    
    # construct data inference dalam dataframe
    data_infer = pd.DataFrame(data=D,index=[0])

    #panggil fungsi inference
    pred = infer(data_infer)

    st.header(f"Plant Nutrition Prediction:")
    st.write(pred[0])