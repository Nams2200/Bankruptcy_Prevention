import streamlit as st 
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

st.sidebar.header("Company Bankrutcy Parameter")

industrial_risk  = st.sidebar.slider("Industrial Risk",0.0,1.0)
management_risk = st.sidebar.slider("Management Risk",0.0,1.0)
financial_risk  = st.sidebar.slider("Financial Risk",0.0,1.0)
creadibility = st.sidebar.slider("Creadibility",0.0,1.0)
competitiveness  = st.sidebar.slider("Competitive",0.0,1.0)
operating_risk  = st.sidebar.slider("Operating_risk",0.0,1.0)



st.markdown("""
       <style>
        .header {
             text-align: center;
             font-size: 40px ! important;
             color: Red;
        }
        </style>
        <p class="header">Bankruptcy Detector</p>
        """,unsafe_allow_html=True)

st.image("./download.jpeg",caption="Bankruptcy Image",width=700)
st.write("Bankurptcy Dataset")
df = pd.read_csv("./bankruptcy-prevention.csv",sep=';')
st.write(df.head())

st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        color: lightgreen;
    }
    </style>
    
    <p class="big-font">Correlation matrix</p>
    """, unsafe_allow_html=True)

df1 = df.iloc[:,0:6]
plot = sns.heatmap(df1.corr(),annot=True,cmap='viridis')
st.pyplot(plot.get_figure())

st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        color: White;
    }
    </style>
    
    <p class="big-font">Result:- </p>
    """, unsafe_allow_html=True)

model1 = joblib.load('./regression_rf__model.joblib')

ypred = model1.predict([[industrial_risk,management_risk,financial_risk,creadibility,competitiveness,operating_risk]])
if ypred == 0:
    st.markdown("""
    <style>
    .Non-bank {
        font-size:30px !important;
        color:green ;
    }
    </style>
    
    <p class="Non-bank">The Organisation is Non-Bankrupted No Need to file petition for Bankruptcy</p>
    """, unsafe_allow_html=True)
    st.image("./Non-Bankrupted.jpg",width=250)
else:
    st.markdown("""
    <style>
    .bank {
        font-size:30px !important;
        color:Red ;
    }
    </style>
    
    <p class="bank">The Organisation is Bankrupted Need to File petition for Bankruptcy</p>
    """, unsafe_allow_html=True)
    st.image("./download (1).jpeg",width=250)


    
