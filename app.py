import streamlit as st
import pandas as pd
import pickle

# — Load artifacts —
model = pickle.load(open('time_to_next_purchase_model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
feat = pd.read_csv('customer_features.csv')

st.title("🕑 Next Purchase Predictor")

phone = st.selectbox("Choose a Customer Phone", feat['Customer_Phone'])
if st.button("Predict"):
    row = feat[feat['Customer_Phone']==phone]
    X = row[['Recency','Frequency','Monetary','Avg_Interpurchase_Interval','Day_of_Week','Month']]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    st.metric("Days until next purchase", f"{pred:.1f}")
