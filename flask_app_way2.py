import pickle
from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
from flasgger import Swagger

with open('/Users/shubh/Desktop/Deploy_ML/ranfor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


app = Flask(__name__)
swagger = Swagger(app)


@app.route('/api', methods = ['POST'])
def iris_prediction():
    """Example endpoint returning iris prediction
    ---
    parameters:
      -name:s_length
       in:query
       type:number
       required :true

      -name:s_width
       in:query
       type:number
       required :true

      -name:p_length
       in:query
       type:number
       required :true

      -name:p_width
       in:query
       type:number
       required :true
    """


    result = request.form

    s_length = result['s_length']
    s_width = result['s_width']
    p_length = result['p_length']
    p_width = result['p_width']

    user_input = {'sepal_length':s_length, 'sepal_width':s_width, 'petal_length':p_length, 'petal_width': p_width}
    # encode the json object to one hot encoding so that it could fit our model
    a = input_to_one_hot(user_input)
    # get the price prediction
    flower_pred = model.predict([a])[0]


    return json.dumps({'category':flower_pred});

if __name__ == '__main__':
    app.run(port=8080)
