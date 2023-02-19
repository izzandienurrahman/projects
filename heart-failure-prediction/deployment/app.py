# import library yang dibutuhkan
import streamlit as st
import pandas as pd
import numpy  as np
import joblib


# load best model
with open('gbc_randcv.pkl','rb') as file_1:
    rf_randcv = joblib.load(file_1)

# Construct Data Infer
# define semua fitur/kolom
features = ['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction',\
    'high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']

def infer(data_infer):
    # predict result random forest model
    y_pred_rf = rf_randcv.predict(data_infer)
    return y_pred_rf 

st.header("Prediksi Pasien Penderita Gagal Jantung")

# artificial data infer
age                         = st.slider("Masukkan Umur:",0,100)                 
anaemia                     = st.radio("Apakah anda penderita anemia?",("Ya","Tidak"))
anaemia_                    = lambda anaemia: 1 if 'Ya' in anaemia else 0
creatinine_phosphokinase    = st.slider("Masukkan level enzim CPK dalam darah (mcg/L):",0,10000)
diabetes                    = st.radio("Apakah anda penderita diabetes?",("Ya","Tidak"))
diabetes_                   = lambda diabetes: 1 if 'Ya' in diabetes else 0
ejection_fraction           = st.slider("Masukkan persentase darah yang keluar pada tiap kontraksi jantung (%):",0,100)
high_blood_pressure         = st.radio("Apakah anda penderita hipertensi?",("Ya","Tidak"))
high_blood_pressure_        = lambda high_blood_pressure: 1 if 'Ya' in high_blood_pressure else 0
platelets                   = st.slider("Masukkan trombosit dalam darah (kiloplatelets/mL):",2e4,1e6)
serum_creatinine            = st.slider("Masukkan level serum creatinine dalam darah (mg/mL):",0.0,10.0)
serum_sodium                = st.slider("Masukkan level serum sodium dalam darah (mEq/mL):",100,200)
sex                         = st.radio("Masukkan jenis kelamin?",("Perempuan","Laki-laki"))
sex_                        = lambda sex: 1 if "Ya" in sex else 0
smoking                     = st.radio("Apakah anda merokok?",("Ya","Tidak"))
smoking_                    = lambda smoking: 1 if "Ya" in smoking else 0
time                        = st.slider("Masukkan periode follow-up (hari):",0,300)
   



if st.button("Submit"):
    D = {
    'age':age,
    'anaemia':anaemia_(anaemia),                    
    'creatinine_phosphokinase':creatinine_phosphokinase,
    'diabetes':diabetes_(diabetes),                    
    'ejection_fraction':ejection_fraction,           
    'high_blood_pressure':high_blood_pressure_(high_blood_pressure),
    'platelets':platelets,                   
    'serum_creatinine':serum_creatinine,            
    'serum_sodium':serum_sodium,                
    'sex':sex_(sex),                         
    'smoking':smoking_(smoking),                     
    'time':time,  
    }
    
    # construct data inference dalam dataframe
    data_infer = pd.DataFrame(data=D,columns=features,index=[0])

    #panggil fungsi inference
    rf_pred = infer(data_infer)
    res_rf = ''

    # interpretasi hasil prediksi
    if rf_pred[0] == 0:
        res_rf = "Kemungkinan tidak akan gagal jantung"
    else:
        res_rf = "Kemungkinan akan gagal jantung"


    st.header(f"Hasil Prediksi: ")
    st.write(res_rf)

