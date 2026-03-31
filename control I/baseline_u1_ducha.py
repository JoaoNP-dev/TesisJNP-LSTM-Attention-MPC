# Línea base u=1 + DUCHA (FÍSICA CORREGIDA Y MÉTRICAS HOMOLOGADAS)
# u_shed(t)=1  -> no se recorta por control
# u_ducha(t)=1 -> si se usa la ducha cuando hay evento
# Simula la batería con balance entre FV-demanda respetando el inversor (5 kW)

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from utils.guardar import guardar_df_csv


def simular_bateria_balance(
    df: pd.DataFrame,
    col_gen_kwh: str,
    col_dem_kwh: str,
    capacidad_kwh: float = 10.0,
    P_inv_max: float = 5.0,
    eff_carga: float = 0.95,
    eff_descarga: float = 0.95,
    soc_ini: float = 0.50,
    soc_min: float = 0.10,
    soc_max: float = 1.00,
) -> pd.DataFrame:
    out = df.copy()

    gen = pd.to_numeric(out[col_gen_kwh], errors="coerce").fillna(0.0).to_numpy()
    dem = pd.to_numeric(out[col_dem_kwh], errors="coerce").fillna(0.0).to_numpy()
    n = len(out)

    soc = np.zeros(n, dtype=float)
    e_bat = np.zeros(n, dtype=float)          # + carga neta almacenada / - descarga neta extraída
    e_charge = np.zeros(n, dtype=float)       # energía enviada a batería antes de eficiencia
    e_discharge = np.zeros(n, dtype=float)    # energía entregada desde batería al sistema
    e_curt = np.zeros(n, dtype=float)
    e_unmet = np.zeros(n, dtype=float)
    e_served = np.zeros(n, dtype=float)

    # Estado inicial
    e_prev_val = soc_ini * capacidad_kwh

    for t in range(n):
        e_min = soc_min * capacidad_kwh
        e_max = soc_max * capacidad_kwh

        espacio_libre = max(0.0, e_max - e_prev_val)
        disponible = max(0.0, e_prev_val - e_min)

        balance = gen[t] - dem[t]

        if balance >= 0:
            # Modo carga
            uc = min(balance, P_inv_max, espacio_libre / max(eff_carga, 1e-6))

            e_charge[t] = uc
            e_discharge[t] = 0.0
            e_bat[t] = uc * eff_carga
            e_curt[t] = balance - uc
            e_unmet[t] = 0.0
            e_served[t] = dem[t]

            e_prev_val += e_bat[t]

        else:
            # Modo descarga
            faltante = -balance
            ud = min(faltante, P_inv_max, disponible * eff_descarga)

            e_charge[t] = 0.0
            e_discharge[t] = ud
            e_bat[t] = -(ud / max(eff_descarga, 1e-6))
            e_unmet[t] = faltante - ud
            e_curt[t] = 0.0
            e_served[t] = dem[t] - e_unmet[t]

            e_prev_val += e_bat[t]

        e_prev_val = np.clip(e_prev_val, e_min, e_max)
        soc[t] = np.clip(e_prev_val / capacidad_kwh, soc_min, soc_max)

    out["SOC_sim"] = soc
    out["energia_bateria_kwh"] = e_bat
    out["bat_charge_kwh"] = e_charge
    out["bat_discharge_kwh"] = e_discharge
    out["energia_curtail_kwh"] = e_curt
    out["ENS_total_kwh"] = e_unmet
    out["demanda_servida_kwh"] = e_served

    return out


def main(
    n_ocupantes: int = 5,
    ruta_in: str | None = None,
    capacidad_bat_kwh: float = 10.0,
    eff_carga: float = 0.95,
    eff_descarga: float = 0.95,
    soc_ini: float = 0.50,
    soc_min: float = 0.10,
    soc_max: float = 1.00,
):
    print("BASELINE u=1 + DUCHA (Física corregida, métricas homologadas)")

    if ruta_in is None:
        ruta_in = os.path.join(BASE_DIR, "resultados", f"df_mpc_ready_ducha_2025_N{n_ocupantes}.csv")

    ruta_in = os.path.abspath(ruta_in)
    if not os.path.exists(ruta_in):
        raise FileNotFoundError(f"No existe: {ruta_in}")

    df = pd.read_csv(ruta_in, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Baseline: no hay recorte por control
    df["u_shed"] = 1.0
    df["u_ducha"] = df["ducha_evento"].astype(float)

    df["demanda_controlable_serv"] = pd.to_numeric(df["demanda_controlable_hora"], errors="coerce").fillna(0.0)
    df["demanda_flexible_serv"] = pd.to_numeric(df["demanda_flexible_hora"], errors="coerce").fillna(0.0)
    df["demanda_ducha_serv"] = df["u_ducha"] * pd.to_numeric(df["ducha_potencial_kwh"], errors="coerce").fillna(0.0)

    # Base física homogénea para métricas finales
    df["demanda_total_potencial_kwh"] = (
        pd.to_numeric(df["demanda_critica_hora"], errors="coerce").fillna(0.0)
        + df["demanda_controlable_serv"]
        + df["demanda_flexible_serv"]
        + df["demanda_ducha_serv"]
    )

    df_sim = simular_bateria_balance(
        df=df,
        col_gen_kwh="energia_fv_kwh",
        col_dem_kwh="demanda_total_potencial_kwh",
        capacidad_kwh=capacidad_bat_kwh,
        P_inv_max=5.0,
        eff_carga=eff_carga,
        eff_descarga=eff_descarga,
        soc_ini=soc_ini,
        soc_min=soc_min,
        soc_max=soc_max,
    )

    ENS_kwh = float(df_sim["ENS_total_kwh"].sum())
    Curt_kwh = float(df_sim["energia_curtail_kwh"].sum())
    demanda_pot_kwh = float(df_sim["demanda_total_potencial_kwh"].sum())
    energia_total_atendida_kwh = float(df_sim["demanda_servida_kwh"].sum())
    Charge_kwh = float(df_sim["bat_charge_kwh"].sum())
    Discharge_kwh = float(df_sim["bat_discharge_kwh"].sum())

    cobertura_energetica_pct = 100.0 * (
        energia_total_atendida_kwh / max(demanda_pot_kwh, 1e-9)
    )

    ducha_pot_total = float(df_sim["demanda_ducha_serv"].sum())
    ducha_serv_total = float(df_sim["demanda_ducha_serv"].sum())  # baseline: si hay evento, se intenta servir toda la ducha
    ducha_serv_pct = 100.0 * ducha_serv_total / max(ducha_pot_total, 1e-9)

    print(f"Demanda potencial total (kWh) : {demanda_pot_kwh:.3f}")
    print(f"Energía total atendida (kWh)  : {energia_total_atendida_kwh:.3f}")
    print(f"ENS (kWh)                     : {ENS_kwh:.3f}")
    print(f"Cobertura energética (%)      : {cobertura_energetica_pct:.2f}")
    print(f"Curtailment (kWh)             : {Curt_kwh:.3f}")
    print(f"Carga batería (kWh)           : {Charge_kwh:.3f}")
    print(f"Descarga batería (kWh)        : {Discharge_kwh:.3f}")
    print(f"Ducha servida                 : {ducha_serv_total:.3f} / {ducha_pot_total:.3f} ({ducha_serv_pct:.2f}%)")
    print(
        f"SOC min/mean/max              : "
        f"{df_sim['SOC_sim'].min():.3f} / "
        f"{df_sim['SOC_sim'].mean():.3f} / "
        f"{df_sim['SOC_sim'].max():.3f}"
    )

    out_name = f"df_baseline_u1_ducha_2025_N{n_ocupantes}.csv"
    guardar_df_csv(df_sim, out_name, subcarpeta="resultados")
    print(f"OK -> resultados/{out_name}")


if __name__ == "__main__":
    main(n_ocupantes=5)