from flask import Flask, request, jsonify, send_file, send_from_directory
from model import Model
from flask_cors import CORS
import os 


app = Flask(__name__, static_folder='build')
model = Model()

CORS(app)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path): # This function checks if the requested path exists in static folder and serves the react app build files 
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        print('Line 19')
        return send_from_directory(app.static_folder, path)
    else:
        print('Line 22')
        return send_from_directory(app.static_folder, 'index.html')



@app.route('/predict', methods = ['GET'])

def predict_stock():
    ticker = request.args.get('ticker')
    if not ticker :
        return jsonify({'error': 'Missing ticker parameter'}),400
    
    model.symbol = ticker
    model.get_data()
    today_stock, lr_pred, forecast_set, order_type = model.linear_reg_algo()
    
    print(today_stock)
    print(lr_pred)
    print(forecast_set)
    print(order_type)


    result = {
        'ticker': ticker,
        'today_stock' : today_stock.to_dict(),
        'lr_pred' : lr_pred,
        'decision' : order_type,
        'forecast_set' : forecast_set.tolist()
    }
    return jsonify(result)

@app.route('/get_image')
def get_image():
    return send_file('linear_reg_algo.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    


    
    
