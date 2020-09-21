# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Predict the next point every four points
train_size, predict_size = 4, 1

# The data fields to be used for training, only take specific fields, a total of 10
data_column = ['lat', 'lon', 'wind_speed', 'wind_direction']  # 'x_gyro', 'y_gyro', 'z_gyro', 'x_acc', 'y_acc', 'z_acc',

path = r'DroneFlightData/WithoutTakeoff'


def window_data(data, window_size):
    X = []
    y = []
    i = 0
    while (i + window_size) <= len( data ) - 1:
        X.append( data[i:i + window_size] )
        y.append( data[i + window_size] )
        i += 1
    assert len( X ) == len( y )
    return X, y


# Read all csv files under the folder, save the file name in csv_file_list
def get_all_csv_file_list(path):
    csv_file_list = []
    for root, dirs, files in os.walk( path ):
        for f in files:
            if os.path.splitext( f )[1] == '.csv':
                csv_file_list.append( os.path.join( root, f ) )
    return csv_file_list


# Get all train data, label
def get_all_train_data_and_label_data(csv_file_list):
    global count
    all_train_data = []
    all_label = []
    for csv_file in csv_file_list:
        df = pd.read_csv( csv_file )
        dataset = df.loc[:, data_column].values
        training_set_scaled = get_scaler().fit_transform( dataset )
        if not np.isnan( training_set_scaled ).any():
            x, y = window_data( training_set_scaled, train_size )
            for a in x:
                all_train_data.append( a )
            for b in y:
                all_label.append( b )

    return np.array( all_train_data ), np.array( all_label )


sc = MinMaxScaler( feature_range=(0, 1) )


def get_scaler():
    return sc
