from flask import Flask, render_template, request,app,jsonify,url_for
import pickle
import numpy as np
app = Flask(__name__)

# loding models

regression = pickle.load(open('random_forest_regressor.pkl','rb'))
scaling = pickle.load(open('scaling.pkl','rb'))

@app.route('/')

def home():
    return render_template('home_web.html')

@app.route('/predict', methods=['POST'])

def prediction():
    json_dict = request.json['data']
    print(json_dict)
    print(f'this is list {list(json_dict.values())}')
    print(np.array(list(json_dict.values())).reshape(1,-1))
    transformed_data = scaling.transform(np.array(list(json_dict.values())).reshape(1,-1))
    output = regression.predict(transformed_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == '__main__':
    app.run(debug=True)