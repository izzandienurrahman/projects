# import library yang dibutuhkan
import streamlit as st
import pandas as pd
import numpy  as np
import joblib


# load best model
with open('catb_randcv.pkl','rb') as file_1:
    catb_pipe = joblib.load(file_1)

# Construct Data Infer
# define semua fitur/kolom
features = ['Gender','Customer Type','Age','Type of Travel','Class','Flight Distance',\
    'Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking',\
        'Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment',\
            'On-board service','Leg room service','Baggage handling','Checkin service','Inflight service',\
                'Cleanliness','Departure Delay in Minutes','Arrival Delay in Minutes']

def infer(data_infer):
    # predict result with pre-trained model
    pred = catb_pipe.predict(data_infer)
    return pred

# header deployment
st.header("Predicting Passenger Flight Satisfaction")

# artificial data infer
gender_options              = ["Male", "Female"]
gender                      = st.selectbox("Please input your gender: ", gender_options)
customer_type_options       = ['Loyal Customer', 'disloyal Customer']
customer_type               = st.selectbox("Which type of customer are you? ", customer_type_options)               
type_of_travel_options      = ['Personal Travel', 'Business travel']
type_of_travel              = st.selectbox("Please input your type of travel: ", type_of_travel_options)
class_options               = ['Eco', 'Eco Plus', 'Business']
class_                      = st.selectbox("Please input your flight class: ", class_options)
age                         = st.slider("Please input your age: ",0,100)    
flight_distance             = st.slider("Please input your flight distance (in Miles): ",0,5000)
wifi_service                = st.slider("Please input your wifi experience (0\:lowest 5\:highest) ",0,5)
departure_arrival_conv      = st.slider("Please input your departure or arrival time convenience experience (0\:lowest 5\:highest)",0,5)
online_booking_exp          = st.slider("Please input your online booking experience (0\:lowest 5\:highest)",0,5)
gate_loc_exp                = st.slider("Please input your gate location  experience (0\:lowest 5\:highest)",0,5)
food_drinks_exp             = st.slider("Please input your food & drinks experience (0\:lowest 5\:highest)",0,5)
online_boarding_exp         = st.slider("Please input your online boarding experience (0\:lowest 5\:highest)",0,5)
seat_comfort_exp            = st.slider("Please input your seat comfort experience (0\:lowest 5\:highest)",0,5)
inflight_entertainment_exp  = st.slider("Please input your inflight entertainment experience (0\:lowest 5\:highest)",0,5)
on_board_svc_exp            = st.slider("Please input your on-board service experience (0\:lowest 5\:highest)",0,5)
leg_room_svc_exp            = st.slider("Please input your leg room service experience (0\:lowest 5\:highest)",0,5)    
baggage_handling_exp        = st.slider("Please input your baggage handling experience (0\:lowest 5\:highest)",0,5)    
checkin_svc_exp             = st.slider("Please input your check-in service experience (0\:lowest 5\:highest)",0,5)    
inflight_svc_exp            = st.slider("Please input your inflight service experience (0\:lowest 5\:highest)",0,5)    
cleanliness                 = st.slider("How do you rate our cleanliness? (0\:lowest 5\:highest)\: ",0,5)    
depart_delay                = st.slider("Did you experience delay in your departure? if so please specify (in minutes): ",0,1500)
arriv_delay                 = st.slider("Did you experience delay in your arrival? if so please specify (in minutes): ",0,1500)


if st.button("Submit"):
    D = {
            'Gender':gender,
            'Customer Type':customer_type,
            'Age':age,
            'Type of Travel':type_of_travel,
            'Class':class_,
            'Flight Distance':flight_distance,
            'Inflight wifi service':wifi_service,
            'Departure/Arrival time convenient':departure_arrival_conv,
            'Ease of Online booking':online_booking_exp,
            'Gate location':gate_loc_exp,
            'Food and drink':food_drinks_exp,
            'Online boarding':online_boarding_exp,
            'Seat comfort':seat_comfort_exp,
            'Inflight entertainment':inflight_entertainment_exp,
            'On-board service':on_board_svc_exp,
            'Leg room service':leg_room_svc_exp,
            'Baggage handling':baggage_handling_exp,
            'Checkin service':checkin_svc_exp,
            'Inflight service':inflight_svc_exp,
            'Cleanliness':cleanliness,
            'Departure Delay in Minutes':depart_delay,
            'Arrival Delay in Minutes':arriv_delay,  
    }
    
    # construct data inference dalam dataframe
    data_infer = pd.DataFrame(data=D,columns=features,index=[0])

    #panggil fungsi inference
    pred = infer(data_infer)


    st.header(f"Prediction Result: ")
    st.write("You are most likely " + pred[0] + " with your flight experience")

