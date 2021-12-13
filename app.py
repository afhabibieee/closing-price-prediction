import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def feature_input(data, scaler):
  timeSteps = 10
#  end = data.index.max()
#  start = end - timedelta(days=timeSteps+1)
#  data_input = data.loc[start:end]
  data_input = data.iloc[-timeSteps:]
  scaled = scaler.fit_transform(data_input)

  n_samples=1
  n_features=scaled.shape[1]
  
  # Reshaping data sebagai 3D input
  scaled=scaled.reshape(n_samples,timeSteps,n_features)
  return scaled

def pred_target(X, model, scaler):
  # Buat prediksi Close value untuk 5 hari kedepan
  next5DaysClose = model.predict(X)
  
  # Kembalikan ke skala original
  next5DaysClose = scaler.inverse_transform(next5DaysClose)
  next5DaysClose = next5DaysClose.reshape(5)
  return next5DaysClose

def df_close_plus_pred(y_close, y_pred):
  day_names = ['Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']

  for i, d in enumerate(day_names, 3):
    if y_close.index.max().day_name() == d:
      next_date = y_close.index.max() + timedelta(days=i)
      next_5dates = [next_date + timedelta(days=j) for j in range(5)]

  df_pred = pd.DataFrame(y_pred, index=next_5dates)
  df_conclose = pd.concat([y_close, df_pred], axis=0)
  df_conclose.rename(columns={0: "Close \n+ \nPredicted"}, inplace=True)
  return df_conclose

def content_ticker_selected(ticker, data, y_pred):
  st.title(ticker)
  st.subheader('\nData {} dari tahun ke tahun\n'.format(ticker))
  st.dataframe(data)
  
  candle = go.Figure(data=[go.Candlestick(x=data.index,
              open=data['Open'],
              high=data['High'],
              low=data['Low'],
              close=data['Close'])])
  st.subheader('\nCandle chart dari stock price [Open, High, Low, Close]\n')
  st.plotly_chart(candle, use_container_width=True)
  
  st.subheader('\nClosing price aktual dan prediksi 5 hari kedepan')
  close_df = df_close_plus_pred(data.Close, y_pred)
  st.line_chart(close_df)
  with st.expander("Penjelasan lebih lajut"):
     st.write("""
         Line chart diatas menampilkan data historis dan data prediktif.
         Data prediktif adalah prediksi nilai-nilai closing price untuk
         lima hari kedepan. Perlu dizoom lebih dekat untuk melihat hasil
         prediksi secara grafis. Sedangkan untuk nilai prediksi sebagai 
         berikut ... 
     """)
     st.table(close_df[-5:])

def pilih_models():
  pilih_model = st.radio(
      "Pilih model yang akan digunakan : [LSTM 2 disarankan/default]",
      ('LSTM 2', 'LSTM 1', 'BiLSTM')
  )
  if pilih_model == 'LSTM 2':
    model = load_model('model_lstm2.h5')
  elif pilih_model == 'LSTM 1':
    model = load_model('model_lstm1.h5')
  else:
    model = load_model('model_bilstm.h5')
  return model

def proses_prediksi(choice, model):
  data = yf.download(choice, period='max')
  data.drop('Volume', inplace=True, axis=1)
  if 0 not in np.array(data.isna().sum()):
    data.dropna(inplace=True)
  
  scaler = MinMaxScaler(feature_range=(0,1))
  X = feature_input(data, scaler)
  y_pred = pred_target(X, model, scaler)
  return data, y_pred

PAGE_CONFIG = {"page_title":"Closing Price Prediction","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)
def main():
  menu = ["Home","BBRI.JK",'BBCA.JK','BBNI.JK','BMRI.JK',"About"]
  model = load_model('model_lstm2.h5')
  choice = st.sidebar.selectbox('Menu',menu)
  if choice == 'Home':
    with st.spinner('Klik toggle menu yang ada di pojok kiri untuk melihat hasil prediksi :)'):
      time.sleep(5)

    st.title("CLOSING PRICE PREDICTION")

    col1, col2 = st.columns(2)
    with col1:
        model = pilih_models()
        st.markdown("")
        
    with col2:
      st.write(
          """
          Aplikasi ini digunakan untuk melakukan prediksi harga penutupan di 
          lima hari kedepan dari berbagai macam bank mulai dari BRI, BCA, BNI 
          dan MANDIRI.

          Dataset pelatihan model menggunakan dataset BBRI.JK dari Yahoo Finance.
          Sementara dataset bank lainnya digunakan untuk inference sebagai transfer 
          learning dari model yang dibuat.
          
          """
      )
      st.markdown("")
      if st.button('View Notebook'):
        colab_link = '[go to colab](https://colab.research.google.com/drive/1L71MVwvb8AtvoKkEOfBiTCYGf-g1lD3w#scrollTo=7hEeBSYiOB5X)'
        st.markdown(colab_link, unsafe_allow_html=True)
                 
  elif choice == 'BBRI.JK':
    with st.spinner('Ini adalah dataset utama yang untuk melatih model'):
      time.sleep(3)

    data, y_pred = proses_prediksi(str(choice), model)
    content_ticker_selected(str(choice), data, y_pred)
  elif choice == 'BBCA.JK':
    data, y_pred = proses_prediksi(str(choice), model)
    content_ticker_selected(str(choice), data, y_pred)
  elif choice == 'BBNI.JK':
    data, y_pred = proses_prediksi(str(choice), model)
    content_ticker_selected(str(choice), data, y_pred)
  elif choice == 'BMRI.JK':
    data, y_pred = proses_prediksi(str(choice), model)
    content_ticker_selected(str(choice), data, y_pred)
  else:
    st.title("ABOUT US")
    st.subheader(
      """Tugas Akhir Microcredential Data Scientist
         Host: Institut Teknologi Bandung 02"""
    )
    with st.container():
      st.write("KELOMPOK 5")
      with st.expander("Adil Faruq Habibi"):
        st.write("[adilfaruqhabibi@gmail.com](%s)" % ("https://mail.google.com/mail/u/?authuser=adilfaruqhabibi@gmail.com"))
      with st.expander("Ismar Apuandi"):
        st.write("[ismarapw0212@gmail.com](%s)" % ("https://mail.google.com/mail/u/?authuser=ismarapw0212@gmail.com"))
      with st.expander("Joshua Galilea"):
        st.write("[joshuagalilea@gmail.com](%s)" % ("https://mail.google.com/mail/u/?authuser=joshuagalilea@gmail.com"))

if __name__ == '__main__':
  main()
