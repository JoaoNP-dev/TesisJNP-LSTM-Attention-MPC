import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.guardar import guardar_df_csv

def _project_root() -> str:
    
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _find_file(base_dir: str, preferred_path: str, filename_contains: str = "") -> str:
   
    if os.path.exists(preferred_path):
        return preferred_path

    candidates = []
    search_dirs = [
        os.path.join(base_dir, "resultados"),
        os.path.join(base_dir, "datasets"),
    ]

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(".csv"):
                if filename_contains.lower() in fn.lower():
                    candidates.append(os.path.join(d, fn))

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # Si son varios, por heurística se elige el primero mas parecido
        candidates_sorted = sorted(candidates, key=lambda x: len(os.path.basename(x)))
        return candidates_sorted[0]

    raise FileNotFoundError(f"No existe: {preferred_path}")


def agregar_ducha_controlable(
    
    archivo_in: str = "resultados/df_maestro_2025_N5_mpc_ready.csv",
    archivo_out: str = "df_mpc_ready_ducha_2025_N5.csv",
    
    n_ocupantes: int = 5,
    alpha_duchas_por_persona_dia: float = 0.9,   
    
    # Intervalos donde se usa la ducha de forma tipicamente
    ventana_manana: tuple[int, int] = (5, 8),    
    ventana_noche: tuple[int, int] = (19, 22),   
    
    # Caracteristicas basicas de la ducha
    potencia_ducha_kw: float = 3.5,
    # Límite de uso de la ducha con bateria
    minutos_bateria: int = 10,
    # uso de la ducha en modo de generacion directa FV 
    minutos_fv_min: int = 10,
    minutos_fv_max: int = 20,
    # excedente de generacion directa para evitar ruido
    excedente_umbral_kwh: float = 0.20,  
    seed: int = 2025,
):
    rng = np.random.default_rng(seed)

       
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ruta_in = os.path.join(base_dir, archivo_in)

    if not os.path.exists(ruta_in):
        raise FileNotFoundError(f"No existe: {ruta_in}")

    df = pd.read_csv(ruta_in, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Validaciones
    
    req = ["timestamp", "energia_fv_kwh", "demanda_critica_hora"]
    faltan = [c for c in req if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas requeridas en MPC: {faltan}")

    df["hora"] = df["timestamp"].dt.hour
    df["fecha"] = df["timestamp"].dt.date

    # Validar el excedente de FV vs la demanda crítica (kWh/h)
    excedente = df["energia_fv_kwh"].astype(float) - df["demanda_critica_hora"].astype(float)
    df["hay_excedente_fv"] = (excedente > excedente_umbral_kwh).astype(int)

    # Inicializar columnas
    df["ducha_evento"] = 0
    df["ducha_minutos"] = 0
    df["ducha_potencial_kwh"] = 0.0
    df["ducha_modo"] = "none"

    
    # Generar eventos por día
    
    dias = pd.Series(df["fecha"].unique()).sort_values().to_list()
    lam = float(alpha_duchas_por_persona_dia) * float(n_ocupantes)  

    total_eventos = 0

    for d in dias:
        # número de eventos al día (Poisson)
        n_eventos = int(rng.poisson(lam=lam))

        if n_eventos <= 0:
            continue

        # indices del día
        idx_dia = df.index[df["fecha"] == d].to_numpy()
        if len(idx_dia) == 0:
            continue

        # candidatos por ventana
        h = df.loc[idx_dia, "hora"].to_numpy()

        cand_manana = idx_dia[(h >= ventana_manana[0]) & (h < ventana_manana[1])]
        cand_noche  = idx_dia[(h >= ventana_noche[0])  & (h < ventana_noche[1])]

        candidatos = np.concatenate([cand_manana, cand_noche])
        if len(candidatos) == 0:
            continue

        # Validacion del número de eventos
        if n_eventos > len(candidatos):
            # si exceden se permite reemplazo 
            elegidos = rng.choice(candidatos, size=n_eventos, replace=True)
        else:
            elegidos = rng.choice(candidatos, size=n_eventos, replace=False)

        for idx in elegidos:
            # Determinar modo por excedente en esa hora
            if int(df.at[idx, "hay_excedente_fv"]) == 1:
                minutos = int(rng.integers(minutos_fv_min, minutos_fv_max + 1))
                modo = "fv"
            else:
                minutos = int(minutos_bateria)
                modo = "bateria"

            # Energía potencial (kWh)
            energia = potencia_ducha_kw * (minutos / 60.0)

            # Registrar
            df.at[idx, "ducha_evento"] = 1
            df.at[idx, "ducha_minutos"] = minutos
            df.at[idx, "ducha_potencial_kwh"] = float(energia)
            df.at[idx, "ducha_modo"] = modo

            total_eventos += 1

    # Limpieza
    df.drop(columns=["fecha"], inplace=True)

    # Guardar
    ruta_out = guardar_df_csv(df, archivo_out, subcarpeta="resultados")

    print(" Ducha controlable agregada para MPC ")
    print("Entrada :", ruta_in)
    print("Salida  :", ruta_out)
    print(f"N ocupantes: {n_ocupantes} | alpha: {alpha_duchas_por_persona_dia} -> lambda={lam:.2f} eventos/día")
    print(f"Total eventos ducha: {total_eventos}")
    print(f"Energía potencial total (kWh): {df['ducha_potencial_kwh'].sum():.3f}")
    print(f"Modo batería (min): {minutos_bateria} | Modo FV (min): [{minutos_fv_min},{minutos_fv_max}]")
    

    return df


if __name__ == "__main__":
    agregar_ducha_controlable(
        archivo_in="resultados/df_maestro_2025_N5_mpc_ready.csv",
        archivo_out="df_mpc_ready_ducha_2025_N5.csv",
        n_ocupantes=5,
        alpha_duchas_por_persona_dia=0.9,
        seed=2025
    )
