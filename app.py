import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Feature engineering functions ---
def calculate_rfm(df):
    latest = df['Delivered_date'].max()
    r = (
        df.groupby('Customer_Phone')['Delivered_date']
          .agg(lambda x: (latest - x.max()).days)
          .rename('Recency')
    )
    f = df.groupby('Customer_Phone')['Customer_Phone'].count().rename('Frequency')
    m = df.groupby('Customer_Phone')['Redistribution Value'].sum().rename('Monetary')
    return pd.concat([r, f, m], axis=1).reset_index()

def calculate_interpurchase(df):
    grp = df.groupby('Customer_Phone').agg(
        First=('Delivered_date','min'),
        Last =('Delivered_date','max'),
        Count=('Order_Id','nunique')
    )
    grp['Time_Since_First'] = (grp['Last'] - grp['First']).dt.days
    grp['Avg_Interpurchase_Interval'] = grp['Time_Since_First'] / grp['Count']
    grp.replace([np.inf, -np.inf], 0, inplace=True)
    return grp['Avg_Interpurchase_Interval'].reset_index()

# --- Load model & scaler ---
model = pickle.load(open('time_to_next_purchase_model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

# --- Load and preprocess data ---
df = pd.read_csv('Data Analysis - Sample File.csv')
df['Delivered_date'] = pd.to_datetime(df['Delivered_date'], dayfirst=True)
df['Redistribution Value'] = (
    df['Redistribution Value'].astype(str)
        .str.replace(',','')
        .astype(float)
)

# --- Build features dataframe ---
rfm_df = calculate_rfm(df)
ipi_df = calculate_interpurchase(df)
features_df = rfm_df.merge(ipi_df, on='Customer_Phone', how='left')
features_df['Day_of_Week'] = df.groupby('Customer_Phone')['Delivered_date'].max().dt.dayofweek.values
features_df['Month'] = df.groupby('Customer_Phone')['Delivered_date'].max().dt.month.values
features_df.fillna(0, inplace=True)

# --- Streamlit UI ---
st.title("Next Purchase Prediction")

# Customer selector
t_customer = features_df['Customer_Phone'].astype(str)
customer = st.selectbox("Select a Customer", t_customer)

# Predict for selected customer
cust_feat = features_df[features_df['Customer_Phone'] == customer]
X = cust_feat[['Recency','Frequency','Monetary','Avg_Interpurchase_Interval','Day_of_Week','Month']].values
X_scaled = scaler.transform(X)
pred_days = model.predict(X_scaled)[0]

st.metric("Predicted days until next purchase", f"{pred_days:.1f} days")

# Optional: show history
if st.checkbox("Show customer purchase history"):
    history = df[df['Customer_Phone'] == int(customer)].sort_values('Delivered_date')
    st.dataframe(history[['Delivered_date','Order_Id','Redistribution Value']])
