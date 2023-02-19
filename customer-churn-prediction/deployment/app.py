# import library yang dibutuhkan
import streamlit as st
import pandas as pd
import numpy  as np
import joblib
from tensorflow import keras

# load preprocessor
with open('preprocessor.pkl','rb') as file_1:
    preprocessor = joblib.load(file_1)

# load ANN model
model = keras.models.load_model('func_model_tuned.hdf5')

# Construct Data Infer
# define semua fitur/kolom
features = [
    'region_category', 'membership_category', 'joined_through_referral',
        'preferred_offer_types', 'medium_of_operation', 'days_since_last_login',
            'avg_time_spent', 'avg_transaction_value', 'avg_frequency_login_days',
                'points_in_wallet', 'used_special_discount',
                    'offer_application_preference', 'feedback']

def infer(data_infer):
    # preprocess input
    preprocessed_data = preprocessor.transform(data_infer)
    # predict result with best model
    pred = model.predict(preprocessed_data)
    pred = np.where(pred>0.5,1,0)
    return pred

# header deployment
st.header("Customer Churn Prediction")


# artificial data infer
region_category_ = ['City', 'Town', 'Village']
region_category = st.selectbox("Where are you from?", region_category_)
membership_category_ = ['No Membership','Basic Membership','Premium Membership','Silver Membership','Gold Membership','Platinum Membership']
membership_category = st.selectbox("What is your membership status?", membership_category_)
joined_through_referral = st.radio("Did you joined through referral?",('Yes','No'))
preferred_offer_types_ = ['Gift Vouchers/Coupons', 'Without Offers','Credit/Debit Card Offers'] 
preferred_offer_types = st.selectbox("Which one is your preferred offer types?", preferred_offer_types_)
medium_of_operation_ = ['Smartphone', 'Desktop', 'Both']               
medium_of_operation = st.selectbox("What kind of device do you use to browse our website?", medium_of_operation_)
days_since_last_login = st.slider("How many days since you last logged in to our website?",0,90)
avg_time_spent = st.slider("Approximately, how long did you browse for product from our website? (in minutes)",0,300)
avg_transaction_value = st.slider("How much did you spend from our website? (give us the average)",0,100000)
avg_frequency_login_days  = st.slider("How frequent did you visits our website in a day?",0,100)    
points_in_wallet = st.slider("How many points do you have in your wallet?",0,10000)
used_special_discount = st.radio("Did you used special discount?",('Yes','No'))
offer_application_preference = st.radio("Did you prefer offers from us?",('Yes','No'))
feedback_ = ['Quality Customer Care', 'Too many ads', 'User Friendly Website',
       'Poor Website', 'No reason specified', 'Poor Customer Service',
       'Poor Product Quality', 'Reasonable Price',
       'Products always in Stock']
feedback = st.selectbox("From the following choice, please give us your feedback",feedback_)



if st.button("Submit"):
    D ={ 
    'region_category'              :region_category,              
    'membership_category'          :membership_category,       
    'joined_through_referral'      :joined_through_referral,     
    'preferred_offer_types'        :preferred_offer_types,        
    'medium_of_operation'          :medium_of_operation,     
    'days_since_last_login'        :days_since_last_login,        
    'avg_time_spent'               :avg_time_spent*60, # convert minutes to seconds             
    'avg_transaction_value'        :avg_transaction_value,        
    'avg_frequency_login_days'     :avg_frequency_login_days,     
    'points_in_wallet'             :points_in_wallet,             
    'used_special_discount'        :used_special_discount,        
    'offer_application_preference' :offer_application_preference, 
    'feedback'                     :feedback  
    }
    
    # construct data inference dalam dataframe
    data_infer = pd.DataFrame(data=D,columns=features,index=[0])


    #panggil fungsi inference
    pred = infer(data_infer)
    pred_string = ''
    if pred==1:
        pred_string = "churn"
    else:
        pred_string = "not churn"

    st.header(f"Prediction Result: ")
    st.write("This customer will " + pred_string)

