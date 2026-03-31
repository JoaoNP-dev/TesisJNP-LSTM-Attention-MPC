import requests
import pandas as pd
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.guardar import guardar_df_csv


def descargar_temp_nasa_ayacucho(anio: int = 2025) -> pd.DataFrame:
    """
    Descarga temperatura a 2m (T2M) horaria desde NASA POWER para Ayacucho.

    """

    lat, lon = -13.1588, -74.2239

    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?"
        "parameters=T2M&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={anio}0101&end={anio}1231"
        "&format=JSON"
    )

    print(f"Solicitando temperatura horaria desde NASA POWER – Ayacucho ({anio})")
    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        raise ConnectionError(f"Error HTTP {response.status_code} al consultar NASA POWER")

    data = response.json()

    try:
        registros = data["properties"]["parameter"]["T2M"]
    except KeyError as e:
        raise KeyError("La variable T2M no está presente en la respuesta de NASA POWER") from e

    filas = []
    for clave, valor in registros.items():
        try:
            ts = datetime.strptime(clave, "%Y%m%d%H")
            filas.append({"timestamp": ts, "temp_c": valor})
        except ValueError:
            print(f"Timestamp inválido ignorado: {clave}")

    df = pd.DataFrame(filas)

    if df.empty:
        raise ValueError("No se obtuvieron datos válidos de temperatura.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Validación 
    if len(df) != 8760:
        print(f"Se esperaban 8760 registros, se obtuvieron {len(df)}.")

    # Limpieza 
    df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    n_nan = int(df["temp_c"].isna().sum())
    if n_nan > 0:
        print(f"{n_nan} valores NaN en temp_c. Se interpolan y se completan bordes.")
        df["temp_c"] = df["temp_c"].interpolate(limit_direction="both")

    # Guardar
    ruta = guardar_df_csv(df, f"temperatura_nasa_ayacucho_{anio}.csv", subcarpeta="datasets")
    print(f"Archivo temperatura guardado en: {ruta}")

    return df


if __name__ == "__main__":
    descargar_temp_nasa_ayacucho(2025)
