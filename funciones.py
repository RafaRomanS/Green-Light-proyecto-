# Librerias necesarias
import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import requests
from bs4 import BeautifulSoup
from datetime import datetime




# Función transformación de tabla df_electricity
def electricity_transformation(df):
    
    # Crear nuevas columnas 'hour' y 'day'
    df['datetime'] = pd.to_datetime(df['forecast_date'])
    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date

    # Calcular la media por día y asignarla a la columna 'price_per_day'
    df["price_per_day"] = df.groupby('date')['euros_per_mwh'].transform("mean")

    # Calcular la diferencia de precio con respecto al valor anterior
    df["price_diff_with_previous"] = df["euros_per_mwh"].diff()

    # Crear columnas para los precios anteriores con shift
    df['previous_price_t-hour'] = df['euros_per_mwh'].shift(1) # 1 hora
    df['previous_price_t-day'] = df['euros_per_mwh'].shift(24) # 1 dia
    df['previous_price_t-week'] = df['euros_per_mwh'].shift(168) # 1 semana
    df['previous_price_t-month'] = df['euros_per_mwh'].shift(720) #  1 mes
    
    # Eliminamos la columna
    df.drop(columns=['forecast_date'], inplace=True)

    return df




# Función transformación de tabla df_gas
def gas_transformation(df):
    
    # Pasamos a formato fecha para poder unir
    df['date'] = pd.to_datetime(df['forecast_date'])
    
    # Creamos la columna average price 
    df['average_price'] = df[['lowest_price_per_mwh' , 'highest_price_per_mwh']].mean(axis=1)
    
    # Creamos la columna price_difference
    df['price_difference'] = df['highest_price_per_mwh'] - df['lowest_price_per_mwh']
    
    # Eliminamos la columna
    df.drop(columns=['forecast_date'], inplace=True)
    
    return df




# Función transformación de tabla df_client
def client_transformation(df):
    
    # Convertimos a formato fecha
    df['date'] = pd.to_datetime(df['date'])
    
    # Proporción de la capacidad instalada con respecto al total
    df['capacity_ratio'] = df['installed_capacity'] / df.groupby('date')['installed_capacity'].transform('sum')
    
    return df




# Función transformación de tabla df_train + agregar holidays
def train_transformation(df, holiday):
    
    # Formato fecha-hora
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Crear nuevas columnas derivadas
    df['date'] = df['datetime'].dt.date
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['datetime'].dt.year # año
    df['month'] = df['datetime'].dt.month # mes
    df['hour'] = df['datetime'].dt.hour # hora
    df['day_of_month'] = df['datetime'].dt.day # dia del mes
    df['day_of_week'] = df['datetime'].dt.day_of_week # dia de la semana
    
    # Creamos la columna que indica si es festivo o no
    df['holiday'] =  df['date'].isin(holiday).astype(int)
    
    # Lagged-target
    df['lagged_target_1day'] = df.groupby(['prediction_unit_id', 'is_consumption'])['target'].shift(24) # 1 dia
    df['lagged_target_2day'] = df.groupby(['prediction_unit_id', 'is_consumption'])['target'].shift(48) # 2 dias
    df['lagged_target_3day'] = df.groupby(['prediction_unit_id', 'is_consumption'])['target'].shift(72) # 3 dias
    df['lagged_target_4day'] = df.groupby(['prediction_unit_id', 'is_consumption'])['target'].shift(96) # 4 dias
    df['lagged_target_5day'] = df.groupby(['prediction_unit_id', 'is_consumption'])['target'].shift(120) # 5 dias
    df['lagged_target_6day'] = df.groupby(['prediction_unit_id', 'is_consumption'])['target'].shift(144) # 6 dias
    df['lagged_target_7day'] = df.groupby(['prediction_unit_id', 'is_consumption'])['target'].shift(168) # 7 dias
    df['lagged_target_15day'] = df.groupby(['prediction_unit_id', 'is_consumption'])['target'].shift(360) # 15 dias
    df['lagged_target_1month'] = df.groupby(['prediction_unit_id', 'is_consumption'])['target'].shift(720) # 30 dias
    
    # Tendencia
    df['target_trend'] = df['lagged_target_2day'] - df['lagged_target_1day']
    df['target_ratio'] = np.where(df['lagged_target_1day'] != 0, (df['lagged_target_2day'] - df['lagged_target_1day']) / df['lagged_target_1day'], np.nan)
    df['target_diff_seasonal'] = df['lagged_target_7day'] - df['lagged_target_1day']
    
    # Columnas que indican si es findesemana o no
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(bool)
    df['is_working_day'] = ~df['is_weekend'].astype(bool)
    
    # Agrega columnas de funciones seno y coseno para la fecha y hora
    df['sin_datetime'] = np.sin(2 * np.pi * df['datetime'].dt.dayofyear / 365)
    df['cos_datetime'] = np.cos(2 * np.pi * df['datetime'].dt.dayofyear / 365)
    
    # Eliminamos esta columna ya que no nos aporta valor
    df.drop(columns=['day_of_week'], inplace=True)

    return df



# Función transformación de tabla df_historical
def historical_transformation(df, df_location):
    
    # Guardamos estas columnas para usar despues
    dates = df['datetime']
    latitude = df['latitude']
    longitude = df['longitude'] 
    
    # Borramos las columnas que no aportan valor
    columas_elim = ['latitude', 'longitude', 'data_block_id', 'datetime','rain', 'snowfall', 'winddirection_10m', 'windspeed_10m', 'cloudcover_high', 'cloudcover_mid']
    df = df.drop(columns = columas_elim, axis=1)
    
    # Scaler
    scaler = StandardScaler().fit(df)
    dt = scaler.transform(df)
    df_historical_scaled = pd.DataFrame(dt, columns=df.columns)
    
    # Aplicamos pca de 4, despues de estudiar cual es el mejor numero de componentes
    pca = PCA(n_components=4, random_state = 42) 
    pca = pca.fit(df_historical_scaled)
    df_historical_transformed = pca.transform(df_historical_scaled)
    
    # Utilizamos este número de clusters porque son los más adecuados, despues de realizar un estudio
    kmeans = KMeans(n_clusters=4, n_init = "auto")
    kmeans_labels = kmeans.fit(df_historical_transformed)
    kmeans.fit(df_historical_transformed)
    labels = kmeans.predict(df_historical_transformed)

    # Agregar los clústers al DataFrame original
    df['labels'] = labels
    
    # Utilizamos este número de clusters porque son los más adecuados, despues de realizar un estudio, para lograr un Kmeans mas especifico
    kmeans = KMeans(n_clusters=10, n_init = 'auto')
    kmeans_labels = kmeans.fit(df_historical_transformed)
    kmeans.fit(df_historical_transformed)
    specific_labels = kmeans.predict(df_historical_transformed)

    # Agregar los clústers al DataFrame original
    df['specific_labels'] = specific_labels

    # Agregamos la columna datetime al DataFrame original
    df['datetime'] = dates
    df['latitude'] = latitude
    df['longitude'] = longitude
    
    # Unimos con la tabla df_location para agregar la columna "county"
    df = pd.merge(df, df_location, how='left', on=['longitude', 'latitude']) 
    
    # Creamos una columna que diferencie temperaturas
    df['temperature_dewpoint_diff_hist'] = df['temperature'] - df['dewpoint']
    
    # Radiación solar total ajustada por la cobertura de nubes
    df['adjusted_solar_radiation'] = df['shortwave_radiation'] * (1 - df['cloudcover_total'])
    
    # Relación entre la temperatura y la presión atmosférica
    df['temperature_pressure_ratio'] = df['temperature'] / df['surface_pressure']
    
    # Sacamos la hora, el dia y el mes para poder obtener variables segun la hora, el dia y el mes para cada region
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day_of_year
    df['month'] = df['datetime'].dt.month
    
    # Creamos una columna de temperatura media por hora, dia y mes para cada region
    df['temperature_per_hour_hist'] = df.groupby(['hour', 'county'])['temperature'].transform(lambda x: x.expanding().mean())
    df['temperature_per_day_hist'] = df.groupby(['day', 'county'])['temperature'].transform(lambda x: x.expanding().mean())
    df['temperature_per_month_hist'] = df.groupby(['month', 'county'])['temperature'].transform(lambda x: x.expanding().mean())
    
    # Creamos una columna con la nubosidad, por hora, dia y mes para cada region
    df['cloudcover_total_per_hour'] = df.groupby(['hour', 'county'])['cloudcover_total'].transform(lambda x: x.expanding().mean())
    df['cloudcover_total_per_day'] = df.groupby(['day', 'county'])['cloudcover_total'].transform(lambda x: x.expanding().mean())
    df['cloudcover_total_per_month'] = df.groupby(['month', 'county'])['cloudcover_total'].transform(lambda x: x.expanding().mean())
    
    # Creamos una columna con la shortwave, por hora, dia y mes para cada region
    df['shortwave_rad_per_hour'] = df.groupby(['hour', 'county'])['shortwave_radiation'].transform(lambda x: x.expanding().mean())
    df['shortwave_rad_per_day'] = df.groupby(['day', 'county'])['shortwave_radiation'].transform(lambda x: x.expanding().mean())
    df['shortwave_rad_per_month'] = df.groupby(['month', 'county'])['shortwave_radiation'].transform(lambda x: x.expanding().mean())
    
    # Creamos una columna con la radiacion directa, por hora, dia y mes para cada region
    df['direct_solar_rad_per_hour_hist'] = df.groupby(['hour', 'county'])['direct_solar_radiation'].transform(lambda x: x.expanding().mean())
    df['direct_solar_rad_per_day_hist'] = df.groupby(['day', 'county'])['direct_solar_radiation'].transform(lambda x: x.expanding().mean())
    df['direct_solar_rad_per_month_hist'] = df.groupby(['month', 'county'])['direct_solar_radiation'].transform(lambda x: x.expanding().mean())
    
    # Creamos una lista con las columnas a las que hacerle la media
    col_mean = [col for col in df.columns if col not in ['labels', 'specific_labels', 'datetime', 'county']]
    
    # Hacemos la media de las columnas por county y en la columna de "labels" utilizamos la moda
    df = df.groupby(['datetime', 'county']).agg({'labels': lambda x: x.mode().iloc[0], 
                                                 'specific_labels': lambda x: x.mode().iloc[0],
                                                 **{col: 'mean' for col in col_mean}}).reset_index()
    
    return df




# Función para obtener los festivos mediante web-scrapping
def get_holiday(year):

    url = f'https://www.timeanddate.com/holidays/estonia/{year}?hol=1' # entramos en la pagina web segun el año

    response = requests.get(url)
    bool(response)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    x = soup.find('section', attrs={'class': 'table-data__table'})
    
    fechas = [] # creamos lista donde almacenaremos las fechas

    for f in x.find_all('tr', attrs={'class': 'showrow'}):
        fecha = f.find('th', attrs={'class':'nw'}).text # obtenemos el mes y dia de la pagina web
        fecha = fecha + ' ' + str(year) # añadimos el año a la fecha
        fecha = replace_month(fecha) # utilizamos función para reemplazar el nombre del mes
        fecha = datetime.strptime(fecha, '%d %b %Y') # convertimos a formato fecha
        fecha = fecha.strftime('%Y-%m-%d') # modificamos al formato fecha ingles
        fechas.append(fecha) # lo agregamos a la lista
        
    return fechas




# Creamos una función que nos permita reemplazar los meses en español por en ingles para detectarlo con el formato fecha
def replace_month(spanish_date):
    months_translation = {
        'ene': 'Jan',
        'feb': 'Feb',
        'mar': 'Mar',
        'abr': 'Apr',
        'may': 'May',
        'jun': 'Jun',
        'jul': 'Jul',
        'ago': 'Aug',
        'sep': 'Sep',
        'oct': 'Oct',
        'nov': 'Nov',
        'dic': 'Dec'
    }

    # Reemplazar los nombres de los meses en español por sus equivalentes en inglés
    for month_es, month_en in months_translation.items():
        spanish_date = spanish_date.replace(f'de {month_es}', month_en)

    return spanish_date




# Función de transformación del df final
def df_transformation_total(df):
    
    # Lagged_target
    df['lagged_target_by_weather_1'] = df.groupby(['prediction_unit_id', 'is_consumption', 'hour', 'labels'])['target'].shift(1)
    df['lagged_target_by_weather_2'] = df.groupby(['prediction_unit_id', 'is_consumption', 'hour', 'labels'])['target'].shift(2) 
    df['lagged_target_by_weather_3'] = df.groupby(['prediction_unit_id', 'is_consumption', 'hour', 'labels'])['target'].shift(3)
    
    df['lagged_target_specific_weather'] = df.groupby(['prediction_unit_id', 'is_consumption', 'hour', 'specific_labels'])['target'].shift(1)
    
    # Creamos una nueva variable
    df['rad_install_cap_relation'] = df['shortwave_radiation'] / df['installed_capacity']
    
    # Eliminamos las filas en las que hay nulos
    df = df.dropna()
    
    # Eliminamos las columnas que no aportan valor
    columnas_a_eliminar = ['datetime', 'data_block_id', 'prediction_unit_id', 'date']
    df = df.drop(columnas_a_eliminar, axis=1)
    
    # Modificamos el tipo de datos de alguna de las variables
    df['county'] = df['county'].astype('category')
    df['product_type'] = df['product_type'].astype('category')
    df['labels'] = df['labels'].astype('category')
    df['specific_labels'] = df['specific_labels'].astype('category')
    df['is_business'] = df['is_business'].astype(bool)
    df['is_consumption'] = df['is_consumption'].astype(bool)
    df['holiday'] = df['holiday'].astype(bool)
    
    return df