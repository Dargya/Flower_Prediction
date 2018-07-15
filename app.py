import pickle
from flask import Flask, request, render_template
import numpy as np
import sys
import waitress

with open('ranfor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

prediction_categories = {0: 'Iris Setosa', 1: 'Iris Virginica', 2: 'Iris Versicolor'}

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def iris_prediction():
    if request.method == 'GET':
        return render_template('random.html')
    elif request.method == 'POST':
        data = request.get_json('data')
        data = data.split('&')
        data_json = {}
        for i in data:
            key, value = i.split('=')
            data_json[key] = value
        print('Data received is : ', data_json)
        predictions = model.predict(np.array([[data_json['s_length'], data_json['s_width'], data_json['p_length'], data_json['p_width']]]))
        return str(prediction_categories[predictions[0]])


if __name__ == '__main__':
    waitress.serve(host='0.0.0.0', port=5000, app=app)  # pass the default port to waitress
    # app.run(debug=True)
