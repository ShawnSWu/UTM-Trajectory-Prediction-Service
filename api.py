from flask import Flask, request
from tensorflow.python.keras.models import load_model
import data_processor
import numpy as np

app = Flask( __name__ )

model = load_model( './uav_predict_model.h5', compile=False )

time_step = 4

@app.route( '/predict', methods=['POST'] )
def predict_all():
    global model_input_json
    if request.method == 'POST':
        model_input_json = request.get_json()

        for plan_and_trajectory in model_input_json:
            model_input_list_array = []
            model_input_list = []

            trajectory = plan_and_trajectory['trajectoryFeature']
            for f in trajectory:
                array = []
                latitude = f['latitude']
                longitude = f['longitude']
                windDegree = f['windDegree']
                windSpeed = f['windSpeed']
                array.append( latitude )
                array.append( longitude )
                array.append( windDegree )
                array.append( windSpeed )
                model_input_list.append( array )
            model_input_list_array.append( model_input_list )
            predict_result_dict = model_predict( model_input_list_array )
            plan_and_trajectory['trajectoryFeature'].append( predict_result_dict )

    return {"predictResult": model_input_json}


def model_predict(model_input_list_array):
    global predict_coordinate
    for t in model_input_list_array:
        np_trajectory = np.array( t )
        normalization_coordinate = data_processor.get_scaler().fit_transform( np_trajectory )

        reshape_normalization_coordinate = np.reshape( normalization_coordinate,
                                                       (1, normalization_coordinate.shape[0],
                                                        normalization_coordinate.shape[1]) )
        predict_normalization_coordinate = model.predict( reshape_normalization_coordinate )
        predict_coordinate = data_processor.get_scaler().inverse_transform( predict_normalization_coordinate )

    predict_result_dict = {
        'latitude': predict_coordinate[0].tolist()[0],
        'longitude': predict_coordinate[0].tolist()[1],
        'windDegree': predict_coordinate[0].tolist()[2],
        'windSpeed': predict_coordinate[0].tolist()[3]
    }

    return predict_result_dict


@app.route( "/" )
def home():
    return "welcome"


if __name__ == "__main__":
    app.run( host='0.0.0.0' )
