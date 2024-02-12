from ast import mod
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Title
app_name = 'Stock Market Forecasting App'
st.title(app_name)
st.subheader('This app is created to forecast the stock market price of the selected company.')
# st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")


st.sidebar.header('Select the parameters from below!')
start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2020, 12, 31))
ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox('Select the company', ticker_list)
data = yf.download(ticker, start=start_date, end=end_date)
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write('Data from', start_date, 'to', end_date)
st.write(data)

# plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')

st.write("Note: Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
fig = px.line(data, x='Date', y=data.columns, title="Closing price of the stock", width=1000, height=600)

st.plotly_chart(fig)

# add a select box to select column from data
column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])

data = data[['Date', column]]
st.write("Selected Data")
st.write(data)

# ADF test check stationarity
st.header('Is data Stationary?')
st.write(adfuller(data[column])[1] < 0.05)

# lets decompose the data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model='additive', period=12)
st.write(decomposition.plot())

st.write("## Plotting the decomposition in plotly")

trace_trend = go.Scatter(x=data["Date"], y=decomposition.trend, name='Trend', line=dict(color='Blue'))
trace_seasonal = go.Scatter(x=data["Date"], y=decomposition.seasonal, name='Seasonality', line=dict(color='green'))
trace_resid = go.Scatter(x=data["Date"], y=decomposition.resid, name='Residuals', line=dict(color='Red', dash='dot'))

fig = go.Figure(data=[trace_trend, trace_seasonal, trace_resid])

st.plotly_chart(fig)

p = st.number_input("Enter p value", value=1)
d = st.number_input("Enter d value", value=1)
q = st.number_input("Enter q value", value=2)


seasonal_p = st.number_input("Enter seasonal value", value=12)
model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, 12))
model_fit = model.fit(disp=0)


#print model summary
st.write("## Model summary")
st.write(model_fit.summary())
st.write("--")

st.write("<p style='color:green; font-size: 50px; font-weight: bold;'>Forecasting the data</p>", unsafe_allow_html=True)
forecast_period = st.number_input("## Enter forecast period in days", value=10)
# Make predictions
forecast_period = 10  # Set the desired forecast period
predictions = model_fit.get_forecast(steps=forecast_period)

predicted_values = predictions.predicted_mean
confidence_intervals = predictions.conf_int()

show_plots = False

if st.button('Show Separate Plots'):
    if not show_plots:
        # Plot actual data
        st.write(px.line(x=data.index, y=data[column], title='Actual', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_layout(
            showlegend=True,
            legend=dict(title='Actual')
        ))

        # Plot predicted values with confidence intervals
        fig = px.line(x=predicted_values.index, y=predicted_values, title='Predicted', width=1200, height=400, labels={'x': 'Date', 'y': 'Predicted Price'})
        
        # Add confidence intervals to the plot
        fig.add_trace(go.Scatter(x=confidence_intervals.index, y=confidence_intervals.iloc[:, 0], fill=None, mode='lines', line_color='rgba(255, 165, 0, 0.3)', name='Lower Bound'))
        fig.add_trace(go.Scatter(x=confidence_intervals.index, y=confidence_intervals.iloc[:, 1], fill='tonexty', mode='lines', line_color='rgba(255, 165, 0, 0.3)', name='Upper Bound'))
        
        fig.update_layout(
            showlegend=True,
            legend=dict(title='Predicted with Confidence Intervals')
        )

        st.plotly_chart(fig)

        show_plots = True