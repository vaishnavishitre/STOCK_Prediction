
# Stock Market Forecasting App

## Overview
This Streamlit app is designed to forecast the stock market price of a selected company using time series analysis and SARIMA (Seasonal AutoRegressive Integrated Moving Average) modeling. Users can choose a specific date range, select a company from a predefined list, and analyze and visualize the stock price trends.

## Features
- Data Visualization: Explore and visualize historical stock prices using interactive line charts.
- Stationarity Check: Conduct Augmented Dickey-Fuller (ADF) test to check the stationarity of the selected stock price column.
- Seasonal Decomposition: Visualize the decomposition of the time series into trend, seasonality, and residuals.
- SARIMA Modeling: Fit a SARIMA model to the selected stock price column and provide a summary of the model.
- Forecasting: Enter a forecast period and generate predictions along with confidence intervals.
- Separate Plots: Toggle between combined and separate plots for actual and predicted stock prices.

## Usage
1. Select the parameters in the sidebar, including start and end dates, and the target company.
2. Explore visualizations and perform stationarity checks.
3. Decompose the time series and fit a SARIMA model.
4. Enter the forecast period and view predictions with confidence intervals.
5. Toggle between combined and separate plots.

## Dependencies
- streamlit
- yfinance
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- statsmodels

## Installation
1. Clone the repository: `git clone https://github.com/vaishnavishitre/stock-market-forecasting-app.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Contributors
- Vaishnavi shitre

