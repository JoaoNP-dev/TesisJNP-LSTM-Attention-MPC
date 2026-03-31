import numpy as np
import pandas as pd

from utils.guardar import guardar_df_csv


def generar_demanda_por_zona_y_equipos(
    n_ocupantes: int = 4,
    path_ocupacion: str | None = None,
    path_calendario: str = "datasets/calendario_2025.csv",
    seed: int = 2025,
    
    # Parámetros (W) - CASA
    # Base 24/7 (promedios): refri + router + standby
    refri_w_avg: float = 110.0,     # refrigeradora promedio (W)
    router_w: float = 12.0,
    standby_w: float = 35.0,

    # Iluminación por zona (W por persona presente en zona)
    luz_w_por_persona: float = 12.0,

    # Enchufes misceláneos por persona (W por persona presente en zona)
    enchufes_w_por_persona: float = 25.0,

    # TV en sala (W) cuando hay ocupación en sala y en horario típico
    tv_w: float = 90.0,
    p_tv_si_sala: float = 0.65,     # prob TV encendida si hay gente en sala en franja típica

    # Laptops/PC (W)  1 a 2 laptops en uso si hay ocupación (tarde/noche)
    laptop_w: float = 45.0,
    max_laptops_en_uso: int = 2,
    p_laptop_si_casa_tarde: float = 0.35,
    p_laptop_si_casa_noche: float = 0.55,

    # Cocina electricidad pequeña (iluminación + pequeños electrodomésticos)
    # Microondas/hervidor ocasional (picos cortos). probabilidad de pico en horas de comida.
    pico_cocina_w: float = 900.0,
    p_pico_cocina_comida: float = 0.25,  # más bajo porque cocina a gas

    # Lavadora: 2 veces/semana (fin de semana) y ciclo de 2 horas aprox.
    lavadora_w: float = 600.0,
    lavadora_ciclos_por_semana: int = 2,
    lavadora_duracion_horas: int = 2,

    # Servicios (baños/hall): extractor/luces 
    servicios_w_si_uso: float = 80.0,
):
    """
    Genera demanda horaria (kWh/h) por zonas y equipos comunes.
    Cocina: picos eléctricos menores y menos frecuentes.

    Salida: datasets/demanda_casa_2025_N{n}.csv
    Columnas:
      - timestamp
      - demanda_dormitorios_kwh
      - demanda_sala_kwh
      - demanda_cocina_kwh
      - demanda_servicios_kwh
      - load_base_kwh (refri+router+standby)
      - load_tv_kwh
      - load_laptops_kwh
      - load_lavadora_kwh
      - demanda_total_hora (kWh/h)
    """

    rng = np.random.default_rng(seed)

    if path_ocupacion is None:
        path_ocupacion = f"datasets/ocupacion_zonas_2025_N{n_ocupantes}.csv"

    df_occ = pd.read_csv(path_ocupacion, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df_cal = pd.read_csv(path_calendario, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Merge para tener tipo_dia/bloque_horario/evento_social 
    df = df_occ.merge(df_cal[["timestamp", "tipo_dia", "bloque_horario", "evento_social"]],
                      on="timestamp", how="left")

    # Validación de columnas mínimas
    req = ["occ_dorm_padres", "occ_dorm_hijos", "occ_sala", "occ_cocina", "occ_servicios"]
    faltan = [c for c in req if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas de ocupación en {path_ocupacion}: {faltan}")

    # dormitorios
    df["occ_dormitorios"] = (df["occ_dorm_padres"] + df["occ_dorm_hijos"]).astype(int)

    # Hora
    df["hora"] = df["timestamp"].dt.hour

    # Horas de comida
    es_hora_comida = df["hora"].isin([7, 8, 12, 13, 14, 19, 20])

    # Carga base 24/7 (W)
    
    base_w = refri_w_avg + router_w + standby_w

    # Cargas por zona (W)
    #    (luces + enchufes misceláneos)
    def zona_w(occ_serie):
        return occ_serie * (luz_w_por_persona + enchufes_w_por_persona)

    dorm_w = zona_w(df["occ_dormitorios"])
    sala_w = zona_w(df["occ_sala"])
    cocina_w = zona_w(df["occ_cocina"])
    servicios_w = df["occ_servicios"].clip(0, 1) * servicios_w_si_uso

    # Equipos por actividad
    
    # TV: solo si hay ocupación en sala y en franja típica (18-23 y 12-15 fin de semana)
    tv_flags = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        h = int(df.at[i, "hora"])
        tipo = str(df.at[i, "tipo_dia"]) if "tipo_dia" in df.columns else "laboral"
        occ_sala = int(df.at[i, "occ_sala"])

        franja_noche = (18 <= h <= 23)
        franja_mediodia = (12 <= h <= 15) and (tipo in ("fin_de_semana", "evento"))

        if occ_sala > 0 and (franja_noche or franja_mediodia):
            tv_flags[i] = 1 if rng.random() < p_tv_si_sala else 0

    load_tv_w = tv_flags * tv_w

    # Laptops: si hay gente en casa (occ_total>0) y franja tarde/noche
    load_laptops_w = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        h = int(df.at[i, "hora"])
        occ_total = int(df.at[i, "occ_total"]) if "occ_total" in df.columns else int(
            df.at[i, "occ_dormitorios"] + df.at[i, "occ_sala"] + df.at[i, "occ_cocina"] + df.at[i, "occ_servicios"]
        )
        if occ_total <= 0:
            continue

        if 13 <= h <= 17:
            p = p_laptop_si_casa_tarde
        elif 18 <= h <= 23:
            p = p_laptop_si_casa_noche
        else:
            p = 0.05  # uso muy bajo fuera de esas franjas

        # cuántas laptops en uso (0..max), acotado por ocupantes
        max_posibles = min(max_laptops_en_uso, occ_total)
        if rng.random() < p and max_posibles > 0:
            n_laptops = 1 if (max_posibles == 1 or rng.random() < 0.65) else 2
        else:
            n_laptops = 0

        load_laptops_w[i] = n_laptops * laptop_w

    # Pico cocina por pequeños electrodomésticos en horas de comida si hay ocupación en cocina
    pico_cocina_flags = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if bool(es_hora_comida.iat[i]) and int(df.at[i, "occ_cocina"]) > 0:
            pico_cocina_flags[i] = 1 if rng.random() < p_pico_cocina_comida else 0
    load_pico_cocina_w = pico_cocina_flags * pico_cocina_w

    # Lavadora: Se modelan ciclos en fin de semana/evento, 2 ciclos por semana, 2h cada uno.
    load_lavadora_w = np.zeros(len(df), dtype=float)
    
    df["iso_year"] = df["timestamp"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["timestamp"].dt.isocalendar().week.astype(int)

    weeks = df[["iso_year", "iso_week"]].drop_duplicates().values.tolist()
    for (yy, ww) in weeks:
        idx_week = df.index[(df["iso_year"] == yy) & (df["iso_week"] == ww)].to_numpy()
        if len(idx_week) == 0:
            continue

        sub = df.loc[idx_week]
        candidatos = sub.index[
            sub["tipo_dia"].isin(["fin_de_semana", "evento"]) &
            sub["hora"].between(9, 18)
        ].to_list()

        if len(candidatos) == 0:
            continue

        rng_week = np.random.default_rng(seed + int(yy)*100 + int(ww))
        rng_week.shuffle(candidatos)

        ciclos_colocados = 0
        ocupados = set()

        for start_idx in candidatos:
            if ciclos_colocados >= lavadora_ciclos_por_semana:
                break
            # validar duración
            pos = df.index.get_loc(start_idx)
            if pos + lavadora_duracion_horas > len(df):
                continue
            seq = df.index[pos:pos + lavadora_duracion_horas].to_list()
            if any(s in ocupados for s in seq):
                continue
            # colocar ciclo
            for s in seq:
                load_lavadora_w[df.index.get_loc(s)] = lavadora_w
                ocupados.add(s)
            ciclos_colocados += 1

    # TOTAL (W) y conversión a kWh/h
    total_w = (
        base_w
        + dorm_w + sala_w + cocina_w + servicios_w
        + load_tv_w + load_laptops_w + load_pico_cocina_w + load_lavadora_w
    )

    out = pd.DataFrame({
        "timestamp": df["timestamp"],

        # por zonas
        "demanda_dormitorios_kwh": dorm_w / 1000.0,
        "demanda_sala_kwh": sala_w / 1000.0,
        "demanda_cocina_kwh": (cocina_w + load_pico_cocina_w) / 1000.0,
        "demanda_servicios_kwh": servicios_w / 1000.0,

        # equipos agregados
        "load_base_kwh": base_w / 1000.0,
        "load_tv_kwh": load_tv_w / 1000.0,
        "load_laptops_kwh": load_laptops_w / 1000.0,
        "load_lavadora_kwh": load_lavadora_w / 1000.0,

        # total
        "demanda_total_hora": total_w / 1000.0
    })

    nombre = f"demanda_casa_2025_N{n_ocupantes}.csv"
    guardar_df_csv(out, nombre, subcarpeta="datasets")
    return out


if __name__ == "__main__":
    generar_demanda_por_zona_y_equipos(n_ocupantes=4, seed=2025)
    generar_demanda_por_zona_y_equipos(n_ocupantes=5, seed=2025)
