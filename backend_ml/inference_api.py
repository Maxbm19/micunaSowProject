# %%

# import packages

    # apis from weater apps

import openmeteo_requests
import requests_cache
from retry_requests import retry

    # basis packages

import pandas as pd
import warnings
from datetime import date,timedelta
import requests

    # web server

from flask import Flask, request,jsonify

    # xgboost

import xgboost as xgb

    # set up env

warnings.filterwarnings('ignore')

# %%

# init sessions

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

#functions for prediccions

def get_xgboost_predictions(model,X):

    dtest = xgb.DMatrix(X)

    predictions = model.predict(dtest)

    return predictions


# start the server for the seeker api

app = Flask(__name__)

#ports for any query by http request
@app.route('/predicions',methods=['GET'])
def probabilities():

    # print(query)

    # read request

    # query = {"latitude": -17.985770, "longitude": -62.386167}

    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')

    print(f"Latitude: {latitude}, Longitude: {longitude}")
    # latitude = query["latitude"]
    # longitude = query["longitude"]

    # define date for call the API

    date_to_extract = (date.today() - timedelta(days=3)).strftime("%Y-%m-%d")

    date_to_backward = (date.today() - timedelta(days=11)).strftime("%Y-%m-%d")

    # date_to_extract = (date.today() - timedelta(days=3+14)).strftime("%Y-%m-%d")

    # date_to_backward = (date.today() - timedelta(days=11+14)).strftime("%Y-%m-%d")

    print(f'extraction date: {date_to_extract}')

    # part 1, meteo files

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude":longitude,
        "start_date": date_to_backward,
        "end_date": date_to_extract,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum", "rain_sum", "precipitation_hours", "shortwave_radiation_sum", "et0_fao_evapotranspiration"],
        "timezone": "America/New_York"
    }

    
    # retreival data from api

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # transform data

    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_temperature_2m_mean = daily.Variables(2).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(3).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(4).ValuesAsNumpy()
    daily_precipitation_hours = daily.Variables(5).ValuesAsNumpy()
    daily_shortwave_radiation_sum = daily.Variables(6).ValuesAsNumpy()
    daily_et0_fao_evapotranspiration = daily.Variables(7).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}

    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["rain_sum"] = daily_rain_sum
    daily_data["precipitation_hours"] = daily_precipitation_hours
    daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
    daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration

    daily_dataframe = pd.DataFrame(data = daily_data)

    # define key for join

    daily_dataframe['date'] = daily_dataframe['date'].dt.strftime('%Y%m%d')
    
    # part 2, NASA

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    date_to_extract_NASA = (date.today() - timedelta(days=3)).strftime("%Y%m%d")

    # date_to_extract_NASA = (date.today() - timedelta(days=3+14)).strftime("%Y%m%d")

    # Definir los parámetros para la solicitud, incluyendo todos los mencionados
    params = {
        "start": date_to_extract_NASA,  # Fecha de inicio
        "end": date_to_extract_NASA,     # Fecha de fin
        "latitude": latitude,  # Latitud (tu ubicación deseada)
        "longitude":longitude,  # Longitud (tu ubicación deseada)
        "community": "AG",  # Comunidad de Agricultura
        "parameters": "QV2M,WD2M,PS,RH2M,PRECTOT,WS2M,WS50M,ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN"
                    ,  # Todos los parámetros mencionados
        "format": "JSON"  # Formato de la respuesta
    }

    # Realizar la solicitud a la API
    response = requests.get(url, params=params)

    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        data = response.json()

        # Extraer los parámetros de la respuesta
        parameters = data['properties']['parameter']

        # Crear un DataFrame a partir de los parámetros
        df_1_nasa = pd.DataFrame(parameters)

        # Transponer el DataFrame para que las fechas sean las filas
        df_1_nasa = df_1_nasa.reset_index()  # Usar reset_index para convertir el índice en una columna
        
        # Renombrar la columna del índice transpuesto a 'Date'
        df_1_nasa.rename(columns={'index': 'date'}, inplace=True)
    else:
        print(f"Error: {response.status_code}")


    # join frames

    df_x = pd.merge(daily_dataframe, df_1_nasa, on='date', how='left')

    df_x['rain_sum_now_7days'] = df_x['precipitation_sum'].rolling(window=8).sum()
    df_x['rain_sum_7to14days'] = df_x['precipitation_sum'].shift(7).rolling(window=7).sum()
    df_x['temp_sum_now_7days'] = df_x['temperature_2m_mean'].rolling(window=8).sum()
    df_x['temp_avg_now_7days'] = df_x['temperature_2m_mean'].rolling(window=8).mean()

    df_x = df_x.sort_values('date', ascending = False).head(1)

    print(df_x)

    # define order for predictions

    list_sorted = ['temperature_2m_max', 'temperature_2m_min',
       'temperature_2m_mean', 'precipitation_sum', 'rain_sum',
       'precipitation_hours', 'shortwave_radiation_sum',
       'et0_fao_evapotranspiration', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'PS',
       'CLRSKY_SFC_SW_DWN', 'WS2M', 'QV2M', 'WD2M', 'WS50M', 'PRECTOTCORR',
       'rain_sum_now_7days', 'rain_sum_7to14days', 'temp_sum_now_7days',
       'temp_avg_now_7days']
    
    df_x_test = df_x[list_sorted]
    
    print(f'nan values for sorted dataframe: {df_x_test.isna().sum()}', end = '\n\n'  )

    print(f'shape: {df_x.shape}, total nan values: {df_x.isna().sum().sum()}', end = '\n\n')

    print(df_x.isna().sum())

    df_x_test = df_x_test.fillna(0) # fill nan values with 0

    # retreival model

    model_from_file = xgb.Booster()
    model_from_file.load_model("./model/model.xgb")

    # calculate the probability

    # df_x_test_ = df_x_test.sort_values('')

    y_predictions = get_xgboost_predictions(model_from_file,df_x_test)

    print(f'the prob is: {y_predictions}')

    recomendacion = 'Sembrar' if y_predictions[0] >0.44 else 'No sembrar'

    data = {
        "probabilidad": float(y_predictions[0]),
        "recommendation": recomendacion,
        "temperatura": float(df_x_test['temperature_2m_mean'].values[0]),
        "precipitacion": float(df_x_test['precipitation_sum'].values[0])
    }

    return jsonify(data)
    # return y_predictions[0]


if __name__ == "__main__":
    print('Prediction server of Micuna is active for all queries - check port 8086 \n')
    app.run(debug=True, host = 'localhost',port= 8086)
