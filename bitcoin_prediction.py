import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from tkinter import Tk, Button, Radiobutton, IntVar

def download_btc_data():
    btc_data = yf.download('BTC-USD', start='2010-01-01', end='2024-01-01')
    btc_data['Days'] = (btc_data.index - datetime(2009, 1, 3)).days
    return btc_data

def fit_model(btc_data):
    btc_data['log_days'] = np.log(btc_data['Days'])
    btc_data['log_price'] = np.log(btc_data['Close'])
    
    X = sm.add_constant(btc_data['log_days'])
    y = btc_data['log_price']
    model = sm.OLS(y, X).fit()
    return model

def predict_future_prices(model):
    future_days = pd.date_range(start=datetime(2024, 1, 1), end='2100-12-31')
    future_days_since_genesis = (future_days - datetime(2009, 1, 3)).days
    log_future_days = np.log(future_days_since_genesis)
    
    X_future = sm.add_constant(log_future_days)
    predicted_log_prices = model.predict(X_future)
    predicted_prices = np.exp(predicted_log_prices)
    
    future_data = pd.DataFrame({
        'Date': future_days,
        'Predicted_Price': predicted_prices
    })
    future_data.set_index('Date', inplace=True)
    return future_data

def plot_btc_data(log_scale):
    btc_data = download_btc_data()
    model = fit_model(btc_data)
    future_data = predict_future_prices(model)
    
    plt.figure(figsize=(12, 6))
    plt.plot(btc_data.index, btc_data['Close'], label='Historical Price', color='blue')
    plt.plot(future_data.index, future_data['Predicted_Price'], label='Prediction', color='red', linestyle='--')
    
    if log_scale:
        plt.yscale('log')
    
    plt.xlabel('Date')
    plt.ylabel('Bitcoin Price (USD)')
    plt.title('Historical Price and Bitcoin Price Prediction until 2100')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add trend line
    plt.plot(btc_data.index, np.exp(model.predict(sm.add_constant(np.log(btc_data['Days'])))), 
             color='green', linestyle='-', linewidth=1, label='Lower Trend Line')
    
    plt.legend()
    plt.show()

def main():
    root = Tk()
    root.title("Bitcoin Price Prediction")
    
    log_scale_var = IntVar()
    log_scale_var.set(1)  # Default to using logarithmic scale
    
    log_button = Radiobutton(root, text="Logarithmic Scale", variable=log_scale_var, value=1)
    linear_button = Radiobutton(root, text="Linear Scale", variable=log_scale_var, value=0)
    
    log_button.pack()
    linear_button.pack()
    
    def plot_button_clicked():
        plot_btc_data(log_scale_var.get() == 1)
    
    plot_button = Button(root, text="Draw Chart", command=plot_button_clicked)
    plot_button.pack()
    
    root.mainloop()

if __name__ == "__main__":
    main()

