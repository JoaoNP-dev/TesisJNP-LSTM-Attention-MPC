import pandas as pd
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from inputs.calendario.evento_social import obtener_eventos_sociales_2025


def guardar_df_csv(df, nombre_archivo, subcarpeta="datasets"):
    ruta_proyecto = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ruta_base = os.path.join(ruta_proyecto, subcarpeta)
    os.makedirs(ruta_base, exist_ok=True)
    ruta_csv = os.path.join(ruta_base, nombre_archivo)
    df.to_csv(ruta_csv, index=False)
    print(f"Archivo guardado en: {ruta_csv}")
    return ruta_csv


# Calendario 2025 (8760 horas)
calendario = pd.date_range(start="2025-01-01 00:00:00", end="2025-12-31 23:00:00", freq="H")
df_cal = pd.DataFrame({"timestamp": calendario})

# Validaciones basicas 
assert df_cal["timestamp"].nunique() == 8760, "Error: calendario incompleto (no son 8760 horas)"
assert df_cal["timestamp"].dt.date.nunique() == 365, "Error: faltan días (no son 365)"
assert df_cal.groupby(df_cal["timestamp"].dt.date).size().eq(24).all(), "Error: algún día no tiene 24 horas"

# Se añaden eventos sociales
fechas_evento_social = obtener_eventos_sociales_2025()
fechas_evento_social = pd.to_datetime(fechas_evento_social, errors="coerce").normalize()
if pd.isna(fechas_evento_social).any():
    raise ValueError("obtener_eventos_sociales_2025() devolvió fechas inválidas.")

# Se añade el tipo de dia
df_cal["tipo_dia"] = df_cal["timestamp"].dt.dayofweek.map(lambda x: "laboral" if x < 5 else "fin_de_semana")
es_evento = df_cal["timestamp"].dt.normalize().isin(fechas_evento_social)
df_cal.loc[es_evento, "tipo_dia"] = "evento"

# Se añaden el bloque horario 
def asignar_bloque_horario(hora: int) -> str:
    if 0 <= hora < 6:
        return "madrugada"
    elif 6 <= hora < 12:
        return "manana"
    elif 12 <= hora < 18:
        return "tarde"
    else:
        return "noche"

df_cal["bloque_horario"] = df_cal["timestamp"].dt.hour.map(asignar_bloque_horario)

# Se pone evento social como binario 
df_cal["evento_social"] = es_evento.astype(int)

# Se añade la hora y mes
df_cal["hora"] = df_cal["timestamp"].dt.hour
df_cal["mes"] = df_cal["timestamp"].dt.month

# Se añade el dia de la semana en formato estandar
df_cal["dia_semana_0_6"] = df_cal["timestamp"].dt.dayofweek  # 0=lun ... 6=dom

# Dia de la semana en formato iso
df_cal["dia_semana_iso"] = df_cal["dia_semana_0_6"] + 1

# Dia de la semana con codificación cíclica perfecto para ML
w = df_cal["dia_semana_iso"].astype(float)  # 1..7
df_cal["dow_sin"] = np.sin(2 * np.pi * (w / 7.0))
df_cal["dow_cos"] = np.cos(2 * np.pi * (w / 7.0))

# hora cíclica para entrenar el LSTM
h = df_cal["hora"].astype(float)  # 0..23
df_cal["hora_sin"] = np.sin(2 * np.pi * (h / 24.0))
df_cal["hora_cos"] = np.cos(2 * np.pi * (h / 24.0))

# Guardar
guardar_df_csv(df_cal, "calendario_2025.csv", subcarpeta="datasets")
