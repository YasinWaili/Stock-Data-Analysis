from flask import Flask, request, render_template
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

API_KEY = 'A6gJb21HaC3lNeYdP1LukUSXT9DEcXmp'

# Create a directory for saving plots if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Get historical data
def getHistoricalData(stock_symbol):
    api_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{stock_symbol}?apikey={API_KEY}"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json().get('historical', [])
    else:
        return []

# Prepare data for linear regression
def prepareData(historical_data):
    df = pd.DataFrame(historical_data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    df['price_prev'] = df['close'].shift(1)
    df = df.dropna()

    X = df[['price_prev']]
    y = df['close']
    
    return X, y

# Fitting the linear regression model
def trainModel(X, y):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate standard error of the model
    predictions = model.predict(X_test)
    residuals = y_test - predictions
    se = np.sqrt(np.sum(residuals**2) / (len(y_test) - 2))
    
    return model, se

# Generate a graph for the stock price of the current day
def createStockGraph(stock_symbol):
    api_url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{stock_symbol}?apikey={API_KEY}"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df['close'], label='Stock Price', color='blue')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.title(f'{stock_symbol} Stock Price - Current Day')
            plt.legend()
            
            graph_path = f'static/{stock_symbol}_graph.png'
            plt.savefig(graph_path)
            plt.close()
            return graph_path
    return None

# Prepare stock info within dictionary
def getStock(stock_data):
    if stock_data and len(stock_data) > 0:
        stock_info = stock_data[0]
        symbol = stock_info.get('symbol')
        stock_logo = f"https://financialmodelingprep.com/image-stock/{symbol}.png"
        return {
            'symbol': stock_info.get('symbol'),
            'price': stock_info.get('price'),
            'change': round(stock_info.get('change'), 2),
            'changes_percentage': round(stock_info.get('changesPercentage'), 2),
            'name': stock_info.get('name'),
            'logo': stock_logo
        }
    else:
        return None

# Calculate the prediction price and the prediction interval
def predictStock(model, last_known_price, se, n, alpha=0.05):
    predicted_price = round(model.predict(np.array([[last_known_price]]))[0], 2)
    
    # Calculate prediction interval
    t_value = stats.t.ppf(1 - alpha/2, df=n - 2)
    margin_of_error = t_value * se * np.sqrt(1 + (1/n))
    
    lower_bound = round(predicted_price - margin_of_error, 2)
    upper_bound = round(predicted_price + margin_of_error, 2)
    
    return predicted_price, lower_bound, upper_bound

# Get the colors of the stock change/prices
def getColor(stock_info):
    if stock_info and len(stock_info) > 0:
        return 'red' if (stock_info.get('change') < 0) else 'green'
    else:
        return 'black'

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/result', methods=['GET'])
def result():
    stock_symbol = request.args.get('stock_symbol')
    if stock_symbol:
        api_url = f"https://financialmodelingprep.com/api/v3/quote/{stock_symbol}?apikey={API_KEY}"
        response = requests.get(api_url)
        if response.status_code == 200:
            stock_data = response.json()
        else:
            stock_data = None
    else:
        stock_data = None

    stock_info = getStock(stock_data)
    color = getColor(stock_info)

    # Get historical data for predictions
    historical_data = getHistoricalData(stock_symbol)
    graph_url = createStockGraph(stock_symbol) 

    if historical_data:
        X, y = prepareData(historical_data)
        model, se = trainModel(X, y)
            
        # Predict the next day's price and calculate prediction interval
        last_known_price = historical_data[0]['close']
        n = len(y)
        predicted_price, lower_bound, upper_bound = predictStock(model, last_known_price, se, n)

        return render_template("result.html", stock_info=stock_info, color=color, 
                               predicted_price=predicted_price, lower_bound=lower_bound, 
                               upper_bound=upper_bound, graph_url=graph_url)
    

    return render_template("result.html", stock_info=stock_info, color=color, graph_url=graph_url)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
