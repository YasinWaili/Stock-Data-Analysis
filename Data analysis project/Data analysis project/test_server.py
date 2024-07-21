from flask import Flask, request, render_template
import requests

app = Flask(__name__)

API_KEY = 'A6gJb21HaC3lNeYdP1LukUSXT9DEcXmp' 


# create a dictionairy with all the stock information within it
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
    
# get the colors of the stock change/prices
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
    return render_template("result.html",stock_info=stock_info, color=color)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
