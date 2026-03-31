# main.py
# Dataset maestro 2025 (Ayacucho) incluye:
# Irradiancia
# temperatura
# tipo dia
# bloque horario
# evento social
# Ocupación 
# Demanda
# Energia FV
# SOC base 
# Energia excedente curtaliment 
# Demanda no servida
# Disponibilidad EV (opciona para futuros trabajos)

import pandas as pd

from inputs.clima.Irr2 import descargar_ghi_ayacucho
from inputs.clima.Temp2 import descargar_temp_nasa_ayacucho
from inputs.solar.calcular_potencia_fv import calcular_potencia_fv

from inputs.control.soc_bateria import simular_soc_bateria
from inputs.ev.modelo_ev import simular_ev_precontrol

from utils.guardar import guardar_df_csv


def main(
    anio: int = 2025,
    n_ocupantes: int = 5,
    # Parámetros del sistema FV
    area_m2: float = 24.0,
    eficiencia_fv: float = 0.2231,
    correccion_inclinacion: float = 1.10,
    umbral_irrad_wm2: float = 15.0,
    
    # Parámetros de la bateria
    
    capacidad_bat_kwh: float = 10.0,
    eficiencia_carga_bat: float = 0.95,
    eficiencia_descarga_bat: float = 0.95,
    soc_bat_inicial: float = 0.50,
    soc_bat_min: float = 0.10,
    soc_bat_max: float = 1.00,
    
    # Parámetros por si se añade un EV
    
    ev_ready: int = 1,
    p_cargador_kw: float = 3.3,
    p_ausencia_aleatoria: float = 0.05,
    seed_ev: int = 2025,
):
    print("Generando dataset maestro")
    print(f"Año: {anio} | N ocupantes: {n_ocupantes}")
    
    # Clima
    print("Descargando irradiancia")
    df_irr = descargar_ghi_ayacucho(anio)
    if df_irr is None or df_irr.empty:
        raise RuntimeError("No se pudo descargar irradiancia")

    # Normalizado
    if "GHI_Wm2" in df_irr.columns:
        df_irr = df_irr.rename(columns={"GHI_Wm2": "ghi_wm2"})
    elif "ghi_wm2" not in df_irr.columns:
        raise KeyError("No se encontró columna de irradiancia")

    print("Descargando temperatura")
    df_temp = descargar_temp_nasa_ayacucho(anio)
    if df_temp is None or df_temp.empty:
        raise RuntimeError("No se pudo descargar temperatura")

    # Normalizado
    if "Temp_C" in df_temp.columns:
        df_temp = df_temp.rename(columns={"Temp_C": "temp_c"})
    elif "temp_c" not in df_temp.columns:
        raise KeyError("No se encontró columna de temperatura")

    # Calendario
    print("Cargando calendario")
    df_cal = pd.read_csv("datasets/calendario_2025.csv", parse_dates=["timestamp"])

    # Ocupación
    print("Cargando ocupación zonal y binaria")
    df_occ = pd.read_csv(f"datasets/ocupacion_zonas_2025_N{n_ocupantes}.csv", parse_dates=["timestamp"])
    df_occ_bin = pd.read_csv(f"datasets/ocupacion_binaria_2025_N{n_ocupantes}.csv", parse_dates=["timestamp"])

    # Combinar dormitorios
    if "occ_dormitorios" not in df_occ.columns:
        if ("occ_dorm_padres" in df_occ.columns) and ("occ_dorm_hijos" in df_occ.columns):
            df_occ["occ_dormitorios"] = df_occ["occ_dorm_padres"] + df_occ["occ_dorm_hijos"]

    cols_occ = ["timestamp"]
    for c in ["occ_dormitorios", "occ_sala", "occ_cocina", "occ_servicios", "occ_total"]:
        if c in df_occ.columns:
            cols_occ.append(c)
    df_occ = df_occ[cols_occ]

    # Demanda
    print("Cargando demanda de la casa")
    df_demanda = pd.read_csv(f"datasets/demanda_casa_2025_N{n_ocupantes}.csv", parse_dates=["timestamp"])

    # Verificación
    if "demanda_total_hora" not in df_demanda.columns:
        raise KeyError("El archivo de demanda debe incluir la columna 'demanda_total_hora' (kWh/h).")

    # Combinación de datos basicos para entrenamiento 
    print("Integrando variables")
    df = (
        df_irr
        .merge(df_temp, on="timestamp", how="inner")
        .merge(df_cal, on="timestamp", how="left")
        .merge(df_occ, on="timestamp", how="left")
        .merge(df_occ_bin, on="timestamp", how="left")
        .merge(df_demanda, on="timestamp", how="left")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Sistema FV
    print("Calculando potencia FV")
    df = calcular_potencia_fv(
        df,
        col_irradiancia="ghi_wm2",
        area_m2=area_m2,
        eficiencia=eficiencia_fv,
        correccion_inclinacion=correccion_inclinacion,
        umbral_irrad_wm2=umbral_irrad_wm2,
        crear_potencia_kw=True
    )

    # Batería
    print("Simulando batería")
    df = simular_soc_bateria(
        df_maestro=df,
        col_generacion="energia_fv_kwh",
        col_demanda="demanda_total_hora",
        capacidad_kwh=capacidad_bat_kwh,
        eficiencia_carga=eficiencia_carga_bat,
        eficiencia_descarga=eficiencia_descarga_bat,
        soc_inicial=soc_bat_inicial,
        soc_min=soc_bat_min,
        soc_max=soc_bat_max
    )

    df = df.rename(columns={
        "SOC_bateria_hora": "SOC_bateria_base",
        "energia_bateria_kwh": "energia_bateria_base_kwh",
        "energia_curtail_kwh": "energia_curtail_base_kwh",
        "demanda_no_servida_kwh": "demanda_no_servida_base_kwh",
    })

    df = df.rename(columns={"demanda_total_hora": "demanda_casa_hora"})

    # Carga adicional opcional para expansion de la tesis
    print("Integrando EV")
    if "tipo_dia" not in df.columns:
        raise KeyError("Falta tipo de dia")

    df = simular_ev_precontrol(
        df_base=df,
        col_timestamp="timestamp",
        col_tipo_dia="tipo_dia",
        usar_ventana_disponible=True,
        generar_presencia_real=True,
        p_ausencia_aleatoria=p_ausencia_aleatoria,
        seed=seed_ev,
        ev_ready=ev_ready,
        p_cargador_kw=p_cargador_kw,
    )

    # Resumen
    print("Dataset maestro creado para entrenamiento")
    print("  Filas:", len(df))
    print("  Periodo:", df["timestamp"].min(), "→", df["timestamp"].max())

    cols_preview = [
        "timestamp",
        "ghi_wm2",
        "temp_c",
        "energia_fv_kwh",
        "potencia_fv_kw",
        "demanda_casa_hora",
        "SOC_bateria_base",
        "demanda_no_servida_base_kwh",
        "energia_curtail_base_kwh",
        "ev_ready",
        "ev_disponible",
        "ev_presente",
        "ev_cargable",
        "p_ev_max_kw",
        "e_ev_max_kwh_h",
        "tipo_dia",
        "bloque_horario",
        "evento_social",
        "occ_total",
        "occ_dormitorios",
        "occ_sala",
        "occ_cocina",
        "occ_servicios",
    ]
    cols_preview = [c for c in cols_preview if c in df.columns]
    print("Vista previa:")
    print(df[cols_preview].head())

    # Exportar
    nombre_salida = f"df_maestro_2025_N{n_ocupantes}.csv"
    guardar_df_csv(df, nombre_salida, subcarpeta="resultados")
    print(f"Dataset guardado en resultados/{nombre_salida}")


if __name__ == "__main__":
    main(anio=2025, n_ocupantes=5)