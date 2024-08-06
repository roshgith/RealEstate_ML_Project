import json
import pickle
import numpy as np
# global variables 
__locations = None
__data_columns = None
__model = None

def get_estimated_house_price(location, total_sqft, bhk, bath):
    load_saved_artifacts()

    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    
    # [x] is numpy array for scikit learn function predict()
    return round(__model.predict([x])[0], 2)

def get_location_names():
    load_saved_artifacts()
    return __locations
def load_saved_artifacts():
    print("loading saved artifacts..START")
    global __locations
    global __data_columns 


    with open("/Users/roshniravi/Documents/RealEstateML_Project/REML_Server/artifacts/columns.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model

    with open("/Users/roshniravi/Documents/RealEstateML_Project/REML_Server/artifacts/bangalore_home_prices_model.pickle",'rb') as f:
        __model = pickle.load(f)
    
    print("loading saved artifacts..FINISHED")



    






if __name__ == "__main__":
    print(get_estimated_house_price('1st Phase JP Nagar', 1000, 2, 3))

    load_saved_artifacts()
    print(get_location_names())
