import requests
import pandas as pd
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.guardar import guardar_df_csv


def descargar_ghi_ayacucho(anio: int = 2023) -> pd.DataFrame:
    """
    Descarga la irradiancia solar global horizontal (GHI) horaria
    desde NASA POWER para Ayacucho (Perú).

    """

    lat, lon = -13.1588, -74.2239

    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?"
        "parameters=ALLSKY_SFC_SW_DWN&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={anio}0101&end={anio}1231"
        "&format=JSON"
    )

    print(f"Solicitando GHI horario desde NASA POWER – Ayacucho ({anio})")
    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        raise ConnectionError(f"Error HTTP {response.status_code} al consultar NASA POWER")

    data = response.json()

    try:
        registros = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    except KeyError as e:
        raise KeyError("La variable ALLSKY_SFC_SW_DWN no está presente en la respuesta") from e

    # Parseo
    filas = []
    for clave, valor in registros.items():
        try:
            ts = datetime.strptime(clave, "%Y%m%d%H")
            filas.append({"timestamp": ts, "ghi_wm2": valor})
        except ValueError:
            print(f"Timestamp inválido ignorado: {clave}")

    df = pd.DataFrame(filas)

    if df.empty:
        raise ValueError("No se obtuvieron datos válidos de GHI.")

    # Ordenar y validar
    df = df.sort_values("timestamp").reset_index(drop=True)

    
    if len(df) != 8760:
        print(f"Se esperaban 8760 registros, se obtuvieron {len(df)}.")

    # Limpieza
    df["ghi_wm2"] = pd.to_numeric(df["ghi_wm2"], errors="coerce")
    n_nan = int(df["ghi_wm2"].isna().sum())
    if n_nan > 0:
        print(f"[AVISO] {n_nan} valores NaN en ghi_wm2. Se imputan como 0.")
        df["ghi_wm2"] = df["ghi_wm2"].fillna(0.0)

    # condicion GHI, no puede ser negativa
    df.loc[df["ghi_wm2"] < 0, "ghi_wm2"] = 0.0

    # Guardar
    ruta = guardar_df_csv(df, f"ghi_ayacucho_{anio}.csv", subcarpeta="datasets")
    print(f"Archivo GHI guardado en: {ruta}")

    return df


if __name__ == "__main__":
    descargar_ghi_ayacucho(2023)
