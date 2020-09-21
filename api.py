from flask import Flask, request
from tensorflow.python.keras.models import load_model
import data_processor
import numpy as np
import json

app = Flask( __name__ )

model = load_model( './uav_predict_model.h5', compile=False )


@app.route( '/predict', methods=['POST'] )
def predict_all():
    global result_list
    if request.method == 'POST':
        model_input = request.get_json()
        trajectory_list = model_input['trajectory_list']
        result_list = []
        for t in trajectory_list:
            np_trajectory = np.array( t )
            normalization_coordinate = data_processor.get_scaler().fit_transform( np_trajectory )

            reshape_normalization_coordinate = np.reshape( normalization_coordinate,
                                                           (1, normalization_coordinate.shape[0],
                                                            normalization_coordinate.shape[1]) )
            predict_normalization_coordinate = model.predict( reshape_normalization_coordinate )
            predict_coordinate = data_processor.get_scaler().inverse_transform( predict_normalization_coordinate )

            lat = predict_coordinate[0][0]
            lon = predict_coordinate[0][1]
            result_list.append( [lon, lat] )
    return json.loads( '{"predictTrajectoryPoint":%s}' % str( result_list ) )


@app.route( "/" )
def home():
    return "welcome"


if __name__ == "__main__":
    app.run( threaded=False )
