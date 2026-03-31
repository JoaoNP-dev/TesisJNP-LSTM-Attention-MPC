
import os
import sys
import numpy as np
import pandas as pd

# Añadir raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.guardar import guardar_df_csv


def preparar_demanda_mpc(
    n_ocupantes: int = 5,
    ruta_df_maestro: str | None = None,
    subcarpeta_entrada: str = "resultados",
    subcarpeta_salida: str = "resultados",
    # Iluminación mínima nocturna (kWh/h)
    e_lux_min_kwh: float = 0.015,   # ~15 Wh por hora (ej: 1-2 LED)
    horas_noche: tuple = (20, 21, 22, 23, 0, 1, 2, 3, 4, 5),
    # Si no existe load_lavadora_kwh se toma cero, es opcional
    permitir_sin_lavadora: bool = True,
):
    """
    Prepara columnas de demanda desagregada para MPC:
      - demanda_critica_hora = load_base_kwh + iluminacion_minima_nocturna
      - demanda_flexible_hora = load_lavadora_kwh (si existe, sino 0)
      - demanda_controlable_hora = demanda_casa_hora - critica - flexible
      - demanda_efectiva_u1 = critica + 1*controlable + flexible (debe = demanda_casa_hora)
    """

    if ruta_df_maestro is None:
        ruta_df_maestro = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            subcarpeta_entrada,
            f"df_maestro_2025_N{n_ocupantes}.csv"
        )

    df = pd.read_csv(ruta_df_maestro, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Validaciones mínimas
    required = ["timestamp", "demanda_casa_hora", "load_base_kwh"]
    faltan = [c for c in required if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas requeridas en df_maestro: {faltan}")

    # Flexible: lavadora si existe
    if "load_lavadora_kwh" not in df.columns:
        if permitir_sin_lavadora:
            df["load_lavadora_kwh"] = 0.0
        else:
            raise KeyError("No existe 'load_lavadora_kwh' y permitir_sin_lavadora=False")

    # Iluminación mínima nocturna (crítica)
    hora = df["timestamp"].dt.hour
    lux_min = np.where(hora.isin(horas_noche), e_lux_min_kwh, 0.0)

    # Columnas MPC
    df["demanda_critica_hora"] = df["load_base_kwh"].astype(float) + lux_min
    df["demanda_flexible_hora"] = df["load_lavadora_kwh"].astype(float)

    # Controlable = lo demás
    df["demanda_controlable_hora"] = (
        df["demanda_casa_hora"].astype(float)
        - df["demanda_critica_hora"]
        - df["demanda_flexible_hora"]
    )

    # Seguridad: no permitir negativos por redondeo
    df["demanda_controlable_hora"] = df["demanda_controlable_hora"].clip(lower=0.0)

    # Validación de suma con u=1
    df["demanda_efectiva_u1"] = (
        df["demanda_critica_hora"]
        + df["demanda_controlable_hora"]
        + df["demanda_flexible_hora"]
    )

    # Check de consistencia
    err = (df["demanda_efectiva_u1"] - df["demanda_casa_hora"]).abs().max()
    print(f"Max |demanda_efectiva_u1 - demanda_casa_hora| = {err:.6f} kWh/h")

    # Guardar salida
    nombre_out = f"df_maestro_2025_N{n_ocupantes}_mpc_ready.csv"
    guardar_df_csv(df, nombre_out, subcarpeta=subcarpeta_salida)
    print(f"OK -> {subcarpeta_salida}/{nombre_out}")

    return df


if __name__ == "__main__":
    preparar_demanda_mpc(n_ocupantes=5)
