import numpy as np
import pandas as pd

def simular_soc_bateria(
    df_maestro: pd.DataFrame,
    col_generacion: str = "energia_fv_kwh",
    col_demanda: str = "demanda_total_hora",
    capacidad_kwh: float = 10.0,
    eficiencia_carga: float = 0.95,
    eficiencia_descarga: float = 0.95,
    soc_inicial: float = 0.5,
    soc_min: float = 0.1,
    soc_max: float = 1.0,
) -> pd.DataFrame:
    """
    Simula SOC horario basado en balance energético (kWh por hora).

    Convenciones:
    - generacion_kwh: energía FV generada en la hora (kWh)
    - demanda_kwh: energía consumida en la hora (kWh)
    - capacidad_kwh: capacidad total de batería (kWh)
    - SOC ∈ [soc_min, soc_max]

    Salidas:
    - SOC_bateria_hora
    - energia_bateria_kwh: + carga almacenada / - energía entregada (kWh)
    - energia_curtail_kwh: FV desperdiciada por batería llena (kWh)
    - demanda_no_servida_kwh: demanda que no pudo cubrirse por batería vacía (kWh)
    """

    df = df_maestro.copy()

    if col_generacion not in df.columns:
        raise KeyError(f"Falta columna de generación '{col_generacion}'.")
    if col_demanda not in df.columns:
        raise KeyError(f"Falta columna de demanda '{col_demanda}'.")

    gen = pd.to_numeric(df[col_generacion], errors="coerce").fillna(0.0).to_numpy()
    dem = pd.to_numeric(df[col_demanda], errors="coerce").fillna(0.0).to_numpy()

    n = len(df)
    soc = np.zeros(n, dtype=float)
    energia_bat = np.zeros(n, dtype=float)
    energia_curtail = np.zeros(n, dtype=float)
    demanda_no_servida = np.zeros(n, dtype=float)

    soc[0] = soc_inicial

    for t in range(1, n):
        soc_prev = soc[t-1]

        # Energía almacenada disponible y espacio libre en batería (kWh)
        energia_en_bat = soc_prev * capacidad_kwh
        energia_min = soc_min * capacidad_kwh
        energia_max = soc_max * capacidad_kwh

        espacio_libre = max(0.0, energia_max - energia_en_bat)
        energia_disponible = max(0.0, energia_en_bat - energia_min)

        # Balance (kWh)
        balance = gen[t] - dem[t]

        if balance >= 0:
            # Exceso: intento cargar
            energia_para_cargar = balance * eficiencia_carga  # energía efectiva que podría almacenarse
            energia_cargada = min(energia_para_cargar, espacio_libre)

            energia_no_usada = max(0.0, balance - (energia_cargada / eficiencia_carga if eficiencia_carga > 0 else 0.0))
            energia_curtail[t] = energia_no_usada

            energia_en_bat_nueva = energia_en_bat + energia_cargada
            energia_bat[t] = +energia_cargada  # positivo = carga almacenada

        else:
            # Déficit
            faltante = -balance  # kWh necesarios para cubrir demanda
            energia_requerida_de_bat = faltante / eficiencia_descarga  # kWh que deben salir de batería (antes de pérdidas)
            energia_descargada = min(energia_requerida_de_bat, energia_disponible)

            # Lo que la batería sí entrega a la carga (después de eficiencia)
            energia_entregada_a_carga = energia_descargada * eficiencia_descarga
            no_servido = max(0.0, faltante - energia_entregada_a_carga)
            demanda_no_servida[t] = no_servido

            energia_en_bat_nueva = energia_en_bat - energia_descargada
            energia_bat[t] = -energia_descargada  # negativo = energía extraída de batería

        # Actualizar SOC
        soc[t] = np.clip(energia_en_bat_nueva / capacidad_kwh, soc_min, soc_max)

    df["SOC_bateria_hora"] = soc
    df["energia_bateria_kwh"] = energia_bat
    df["energia_curtail_kwh"] = energia_curtail
    df["demanda_no_servida_kwh"] = demanda_no_servida

    return df
