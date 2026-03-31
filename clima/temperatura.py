import requests
import pandas as pd
from utils.guardar import guardar_df_csv

def descargar_temperatura_ayacucho(año=2024):
    """
    Descarga datos horarios de temperatura para Ayacucho desde Open-Meteo
    y guarda el CSV en datasets/ con nombre estructurado.
    """

    lat, lon = -13.1588, -74.2239
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={año}-01-01&end_date={año}-12-31"
        f"&hourly=temperature_2m"
        f"&timezone=America/Lima"
    )

    response = requests.get(url)
    data = response.json()

    df_temp = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "Temp_C": data["hourly"]["temperature_2m"]
    })

    # Guardado 
    guardar_df_csv(df_temp, f"temperatura_ayacucho_{año}.csv")

    return df_temp


if __name__ == "__main__":
    descargar_temperatura_ayacucho(2024)