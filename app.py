import streamlit as st
import pickle
import numpy as np
import sklearn

# Load Model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title('Laptop Price Prediction')

Brand = st.selectbox('Select the Brand', df['Brand'].unique())

OS = st.selectbox('Select the Type Operating System',df['OS'].unique())

Processor = st.selectbox('Select the Type of Processor',df['Processor'].unique())

# RAM_Type = st.selectbox('Select the Type of RAM',df['RAM_Type'].unique())


RAM_Size = st.selectbox('Select the Size of RAM',df['RAM_Size'].unique())

Storage_Size = st.selectbox('Select the Amount of Storage',df['Storage_Size'].unique())

Storage_Type = st.selectbox('Select the Type of Storage',df['Storage_Type'].unique())

if st.button('Predict Price'):
    query = np.array([Brand,OS,Processor,RAM_Size,Storage_Size,Storage_Type])

    query = query.reshape(1,6)
    st.title("The predicted price of laptop is: " +'Rs.'+ str(int(np.exp(pipe.predict(query)[0]))))