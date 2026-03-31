import requests
import pandas as pd
from utils.guardar import guardar_df_csv

def descargar_irradiancia_ayacucho(año=2024):
    """
    Descarga datos horarios de irradiancia para Ayacucho desde Open-Meteo
    y guarda el CSV en datasets/ con nombre estructurado.
    """

    lat, lon = -13.1588, -74.2239
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={año}-01-01&end_date={año}-12-31"
        f"&hourly=shortwave_radiation"
        f"&timezone=America/Lima"
    )

    response = requests.get(url)
    data = response.json()

    df_irrad = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "Irr_Wm2": data["hourly"]["shortwave_radiation"]
    })

    # Guardado 
    guardar_df_csv(df_irrad, f"irradiancia_ayacucho_{año}.csv")

    return df_irrad


if __name__ == "__main__":
    descargar_irradiancia_ayacucho(2024)