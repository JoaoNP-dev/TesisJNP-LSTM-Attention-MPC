import sys
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from utils.guardar import guardar_df_csv


def _clip_int(x, lo, hi):
    return int(max(lo, min(hi, x)))


def generar_ocupacion_2025(
    path_calendario: str = None,
    n_ocupantes: int = 4,
    perfil: str = "P2",
    seed: int = 2025,
    jitter_horas: int = 1,
):
    """
    Generador sintético de ocupación horaria por zonas (2025).

    Zonas humanas:
    - occ_dorm_padres
    - occ_dorm_hijos
    - occ_sala
    - occ_cocina
    - occ_servicios  (baños/hall/transitorio)

    Salidas:
    - datasets/ocupacion_zonas_2025_N{n}.csv
    - datasets/ocupacion_binaria_2025_N{n}.csv

   
    """

    rng = np.random.default_rng(seed)

    if path_calendario is None:
        path_calendario = os.path.join(BASE_DIR, "datasets", "calendario_2025.csv")

    print("Leyendo calendario desde:", path_calendario)

    if not os.path.exists(path_calendario):
        raise FileNotFoundError(f"No se encontró el archivo: {path_calendario}")

    df_cal = pd.read_csv(path_calendario, parse_dates=["timestamp"])
    df_cal = df_cal.sort_values("timestamp").reset_index(drop=True)

    req = ["timestamp", "tipo_dia", "bloque_horario", "evento_social"]
    faltan = [c for c in req if c not in df_cal.columns]
    if faltan:
        raise KeyError(f"El calendario no contiene columnas requeridas: {faltan}")

    df_cal["tipo_dia"] = df_cal["tipo_dia"].astype(str)
    df_cal["bloque_horario"] = df_cal["bloque_horario"].astype(str)
    df_cal["evento_social"] = pd.to_numeric(df_cal["evento_social"], errors="coerce").fillna(0).astype(int)

    n = len(df_cal)

    # Outputs
    occ_padres = np.zeros(n, dtype=int)
    occ_hijos = np.zeros(n, dtype=int)
    occ_sala = np.zeros(n, dtype=int)
    occ_coc = np.zeros(n, dtype=int)
    occ_serv = np.zeros(n, dtype=int)

    # Perfiles (controlan casa vacía / presencia diurna)
    if perfil == "P1":
        p_fuera_dia_laboral = 0.65
        p_al_menos_uno_en_casa_laboral = 0.20
    elif perfil == "P3":
        p_fuera_dia_laboral = 0.15
        p_al_menos_uno_en_casa_laboral = 0.70
    else:  # P2
        p_fuera_dia_laboral = 0.35
        p_al_menos_uno_en_casa_laboral = 0.45

    # Asignación familiar base
    n_padres = 2 if n_ocupantes >= 2 else 1
    n_hijos = max(0, n_ocupantes - n_padres)

    # FACTORES por tipo de día para servicios (baño/hall)
    # laboral: rutina diaria (mañana/noche)
    # fin_de_semana: más permanencia en casa => más uso distribuido
    # evento: más actividad social => mayor frecuencia (sobre todo noche)
    factor_serv_tipo_dia = {
        "laboral": 1.00,
        "fin_de_semana": 1.20,
        "evento": 1.35,
    }

    for i in range(n):
        ts = df_cal.loc[i, "timestamp"]
        h = int(ts.hour)
        tipo = df_cal.loc[i, "tipo_dia"]
        bloque = df_cal.loc[i, "bloque_horario"]
        evento = int(df_cal.loc[i, "evento_social"])

        # Jitter controlado en transiciones típicas
        jitter = 0
        if h in (6, 7, 8, 17, 18, 22, 23):
            jitter = int(rng.integers(-jitter_horas, jitter_horas + 1))
        h_eff = _clip_int(h + jitter, 0, 23)

        # Cuántas personas están en casa (occ_total)
        if tipo == "laboral":
            if 0 <= h_eff <= 5:
                occ_total = n_ocupantes
            elif 6 <= h_eff <= 8:
                if rng.random() < p_al_menos_uno_en_casa_laboral:
                    occ_total = 1
                else:
                    occ_total = 0 if rng.random() < p_fuera_dia_laboral else 1
            elif 9 <= h_eff <= 16:
                if rng.random() < p_fuera_dia_laboral:
                    occ_total = 0
                else:
                    occ_total = 1 if rng.random() < 0.75 else 2
            elif 17 <= h_eff <= 21:
                occ_total = n_ocupantes if rng.random() < 0.85 else max(1, n_ocupantes - 1)
            else:
                occ_total = n_ocupantes

        elif tipo == "fin_de_semana":
            if 0 <= h_eff <= 6:
                occ_total = n_ocupantes
            elif 7 <= h_eff <= 11:
                occ_total = n_ocupantes if rng.random() < 0.80 else max(1, n_ocupantes - 1)
            elif 12 <= h_eff <= 17:
                if rng.random() < 0.25:
                    occ_total = 0
                else:
                    occ_total = n_ocupantes if rng.random() < 0.70 else max(1, n_ocupantes - 1)
            else:
                occ_total = n_ocupantes

        else:  # evento
            if 0 <= h_eff <= 6:
                occ_total = n_ocupantes
            elif 7 <= h_eff <= 11:
                occ_total = n_ocupantes if rng.random() < 0.85 else max(1, n_ocupantes - 1)
            elif 12 <= h_eff <= 17:
                occ_total = n_ocupantes if rng.random() < 0.75 else max(1, n_ocupantes - 1)
            else:
                occ_total = n_ocupantes

        occ_total = _clip_int(occ_total, 0, n_ocupantes)

        # Distribución por zonas
        p = hjs = sala = cocina = serv = 0

        if occ_total == 0:
            p = hjs = sala = cocina = serv = 0
        else:
            # SERVICIOS (baño/hall) 
            # Probabilidad base por franja horaria
            if 5 <= h_eff <= 8:
                p_serv_base = 0.30  # pico matutino
            elif 18 <= h_eff <= 22:
                p_serv_base = 0.25  # pico nocturno
            elif 9 <= h_eff <= 17:
                p_serv_base = 0.07  # uso ocasional diurno
            else:
                p_serv_base = 0.04  # madrugada muy bajo

            # Modulación por tipo de día
            f_tipo = factor_serv_tipo_dia.get(tipo, 1.00)
            p_serv = p_serv_base * f_tipo

            # Refuerzo por evento_social en noche
            if evento == 1 and 18 <= h_eff <= 22:
                p_serv *= 1.25

            # Limite de probabilidad
            p_serv = min(p_serv, 0.60)

            # Ocurre 0 o 1 persona en servicios (transitorio)
            serv = 1 if rng.random() < p_serv else 0
            serv = min(serv, occ_total)

            # COCINA: picos de comida 
            es_hora_comida = (7 <= h_eff <= 8) or (12 <= h_eff <= 14) or (19 <= h_eff <= 20)
            if es_hora_comida and (occ_total - serv) > 0:
                cocina = 1 + (1 if (occ_total >= 4 and rng.random() < 0.45) else 0)
                cocina = min(cocina, occ_total - serv)

            # DORMITORIOS vs SALA según franja horaria 
            restante = max(0, occ_total - serv - cocina)

            if 0 <= h_eff <= 5:
                # madrugada: casi todos en dormitorios
                p = min(n_padres, restante)
                hjs = min(n_hijos, restante - p)
                sala = restante - p - hjs

            elif 22 <= h_eff <= 23:
                # transición para dormir
                if evento == 1:
                    sala = min(restante, max(1, restante - 2))
                else:
                    sala = min(restante, max(1, restante - 3))
                dorm_rest = restante - sala
                p = min(n_padres, dorm_rest)
                hjs = min(n_hijos, dorm_rest - p)

            elif bloque == "noche":
                # noche: sala + dormitorios
                if evento == 1:
                    sala = min(restante, max(1, restante - 1))
                else:
                    sala = min(restante, max(1, restante - 2))
                dorm_rest = restante - sala
                p = min(n_padres, dorm_rest)
                hjs = min(n_hijos, dorm_rest - p)

            else:
                # mañana/tarde: sala domina; posible siesta pequeña
                prob_siesta = 0.08 if tipo == "laboral" else 0.12
                if rng.random() < prob_siesta and restante > 0:
                    if n_hijos > 0 and rng.random() < 0.6:
                        hjs = 1
                    else:
                        p = 1
                    sala = restante - p - hjs
                else:
                    sala = restante
                    p = 0
                    hjs = 0

            # Ajuste final suma exacta
            suma = p + hjs + sala + cocina + serv
            if suma != occ_total:
                delta = occ_total - suma
                sala = max(0, sala + delta)

        occ_padres[i] = _clip_int(p, 0, n_ocupantes)
        occ_hijos[i] = _clip_int(hjs, 0, n_ocupantes)
        occ_sala[i] = _clip_int(sala, 0, n_ocupantes)
        occ_coc[i] = _clip_int(cocina, 0, n_ocupantes)
        occ_serv[i] = _clip_int(serv, 0, n_ocupantes)

    df_occ = pd.DataFrame({
        "timestamp": df_cal["timestamp"],
        "occ_dorm_padres": occ_padres,
        "occ_dorm_hijos": occ_hijos,
        "occ_sala": occ_sala,
        "occ_cocina": occ_coc,
        "occ_servicios": occ_serv,
    })

    df_occ["occ_total"] = (
        df_occ[["occ_dorm_padres", "occ_dorm_hijos", "occ_sala", "occ_cocina", "occ_servicios"]]
        .sum(axis=1)
        .clip(0, n_ocupantes)
    )

    df_bin = pd.DataFrame({
        "timestamp": df_occ["timestamp"],
        "ocupacion_binaria": (df_occ["occ_total"] > 0).astype(int)
    })

    # Guardar
    nombre_occ = f"ocupacion_zonas_2025_N{n_ocupantes}.csv"
    nombre_bin = f"ocupacion_binaria_2025_N{n_ocupantes}.csv"
    guardar_df_csv(df_occ, nombre_occ, subcarpeta="datasets")
    guardar_df_csv(df_bin, nombre_bin, subcarpeta="datasets")

    return df_occ, df_bin


if __name__ == "__main__":
    # Caso base y sensibilidad
    generar_ocupacion_2025(n_ocupantes=4, perfil="P2", seed=2025)
    generar_ocupacion_2025(n_ocupantes=5, perfil="P2", seed=2025)
