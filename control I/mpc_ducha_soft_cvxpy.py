# MPC ORÁCULO PERFECTO (H=12) - CORREGIDO FÍSICAMENTE Y HOMOLOGADO EN MÉTRICAS
# Sin LSTM, usa los datos reales del futuro en ventana de 12 horas.

import os
import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except Exception as e:
    raise ImportError("No está instalado cvxpy. Instala con: pip install cvxpy ecos osqp scs") from e


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def ruta(*parts) -> str:
    return os.path.join(project_root(), *parts)


def mpc_ducha_soft_oracle(
    n_ocupantes: int = 5,
    ruta_in: str | None = None,
    ruta_out: str | None = None,
    Np: int = 12,
    capacidad_kwh: float = 10.0,
    P_inv_max: float = 5.0,
    eta_c: float = 0.95,
    eta_d: float = 0.95,
    soc_min: float = 0.10,
    soc_max: float = 1.00,
    soc_init: float = 0.50,
    w_unmet: float = 5000.0,
    w_shed: float = 10.0,
    w_ducha: float = 15.0,
    w_curt: float = 0.001,
    w_bat: float = 0.05,
    solver_primary: str = "OSQP",
    solver_fallback: str = "SCS",
    verbose: bool = False,
):

    if ruta_in is None:
        ruta_in = ruta("resultados", f"df_mpc_ready_ducha_2025_N{n_ocupantes}.csv")
    if ruta_out is None:
        ruta_out = ruta("resultados", f"df_mpc_oraculo_perfecto_2025_N{n_ocupantes}_H{Np}.csv")

    if not os.path.exists(ruta_in):
        raise FileNotFoundError(f"No existe: {ruta_in}")

    df = pd.read_csv(ruta_in, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    pv = pd.to_numeric(df["energia_fv_kwh"], errors="coerce").fillna(0.0).to_numpy()
    d_crit = pd.to_numeric(df["demanda_critica_hora"], errors="coerce").fillna(0.0).to_numpy()
    d_ctrl = pd.to_numeric(df["demanda_controlable_hora"], errors="coerce").fillna(0.0).to_numpy()
    d_flex = pd.to_numeric(df["demanda_flexible_hora"], errors="coerce").fillna(0.0).to_numpy()
    ducha_evento = pd.to_numeric(df["ducha_evento"], errors="coerce").fillna(0.0).to_numpy()
    ducha_pot = pd.to_numeric(df["ducha_potencial_kwh"], errors="coerce").fillna(0.0).to_numpy()

    n = len(df)

    # Outputs
    u_shed_hist = np.ones(n, dtype=float)
    u_ducha_hist = np.zeros(n, dtype=float)
    soc_hist = np.zeros(n, dtype=float)
    ens_hist = np.zeros(n, dtype=float)
    curt_hist = np.zeros(n, dtype=float)
    served_total = np.zeros(n, dtype=float)
    ducha_servida = np.zeros(n, dtype=float)
    P_c = np.zeros(n, dtype=float)
    P_d = np.zeros(n, dtype=float)

    # Base física homogénea para métricas finales
    demanda_total_pot = d_crit + d_flex + d_ctrl + ducha_pot

    e_bat = soc_init * capacidad_kwh
    e_min = soc_min * capacidad_kwh
    e_max = soc_max * capacidad_kwh

    soc_hist[0] = soc_init

    print(f"Ejecutando Oráculo Perfecto (H={Np}) con reglas físicas estrictas...")

    for t0 in range(n):
        Hh = min(Np, n - t0)

        pv_h = pv[t0:t0 + Hh]
        dcrit_h = d_crit[t0:t0 + Hh]
        dctrl_h = d_ctrl[t0:t0 + Hh]
        dflex_h = d_flex[t0:t0 + Hh]
        ev_h = ducha_evento[t0:t0 + Hh]
        pot_h = ducha_pot[t0:t0 + Hh]

        u = cp.Variable(Hh)
        v = cp.Variable(Hh)
        e = cp.Variable(Hh + 1)
        uc = cp.Variable(Hh, nonneg=True)
        ud = cp.Variable(Hh, nonneg=True)
        curt = cp.Variable(Hh, nonneg=True)
        ens = cp.Variable(Hh, nonneg=True)

        cons = []
        cons += [u >= 0, u <= 1]
        cons += [v >= 0, v <= 1]
        cons += [uc <= P_inv_max]
        cons += [ud <= P_inv_max]

        cons += [e[0] == e_bat]
        cons += [e >= e_min, e <= e_max]
        cons += [e[1:] == e[:-1] + eta_c * uc - (1.0 / max(eta_d, 1e-6)) * ud]

        for k in range(Hh):
            demanda_serv = dcrit_h[k] + dflex_h[k] + u[k] * dctrl_h[k] + v[k] * pot_h[k]

            if ev_h[k] <= 0 or pot_h[k] <= 1e-12:
                cons += [v[k] == 0]

            cons += [pv_h[k] + ud[k] == demanda_serv + uc[k] + curt[k] - ens[k]]
            cons += [ens[k] <= demanda_serv + 1e-6]

        cost = 0
        cost += w_unmet * cp.sum(ens)
        cost += w_curt * cp.sum(curt)
        cost += w_bat * cp.sum(uc + ud)
        cost += w_shed * cp.sum(cp.multiply((1 - u), dctrl_h))
        cost += w_ducha * cp.sum(cp.multiply((1 - v), pot_h))

        prob = cp.Problem(cp.Minimize(cost), cons)

        solved = False
        for sol in [solver_primary, solver_fallback]:
            try:
                prob.solve(solver=getattr(cp, sol), verbose=verbose, warm_start=True)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    solved = True
                    break
            except Exception:
                continue

        if not solved:
            uc0, ud0, v0 = 0.0, 0.0, 0.0
            u0 = 1.0
            d_tot_base = dcrit_h[0] + dflex_h[0] + dctrl_h[0]
            ens0 = max(0.0, d_tot_base - pv_h[0] - max(0.0, (e_bat - e_min) * eta_d))
            curt0 = max(0.0, pv_h[0] - d_tot_base)
        else:
            uc0 = float(uc.value[0])
            ud0 = float(ud.value[0])
            ens0 = float(ens.value[0])
            curt0 = float(curt.value[0])
            u0 = float(u.value[0])
            v0 = float(v.value[0])

        e_bat = float(np.clip(e_bat + eta_c * uc0 - (1.0 / max(eta_d, 1e-6)) * ud0, e_min, e_max))

        soc_hist[t0] = e_bat / capacidad_kwh
        u_shed_hist[t0] = np.clip(u0, 0.0, 1.0)
        u_ducha_hist[t0] = np.clip(v0, 0.0, 1.0)
        P_c[t0] = max(0.0, uc0)
        P_d[t0] = max(0.0, ud0)
        ens_hist[t0] = max(0.0, ens0)
        curt_hist[t0] = max(0.0, curt0)

        ducha_servida[t0] = u_ducha_hist[t0] * ducha_pot[t0]

        # Energía realmente atendida sobre base física homogénea
        served_total[t0] = (
            d_crit[t0]
            + d_flex[t0]
            + u_shed_hist[t0] * d_ctrl[t0]
            + ducha_servida[t0]
            - ens_hist[t0]
        )

    out = df.copy()
    out["SOC_mpc"] = soc_hist
    out["bat_charge_kwh"] = P_c
    out["bat_discharge_kwh"] = P_d
    out["ENS_total_kwh"] = ens_hist
    out["Curt_kwh"] = curt_hist
    out["u_shed"] = u_shed_hist
    out["u_ducha"] = u_ducha_hist
    out["ducha_servida_kwh"] = ducha_servida
    out["demanda_total_potencial_kwh"] = demanda_total_pot
    out["demanda_servida_kwh"] = served_total

    # Resumen anual homologado
    ENS = float(np.sum(ens_hist))
    CURT = float(np.sum(curt_hist))
    Charge_kwh = float(np.sum(P_c))
    Discharge_kwh = float(np.sum(P_d))
    demanda_pot_kwh = float(np.sum(demanda_total_pot))
    energia_total_atendida_kwh = float(np.sum(served_total))

    cobertura_energetica_pct = 100.0 * (
        energia_total_atendida_kwh / max(1e-9, demanda_pot_kwh)
    )

    ducha_pot_total = float(np.sum(ducha_pot))
    ducha_serv_total = float(np.sum(ducha_servida))
    ducha_serv_pct = 100.0 * ducha_serv_total / max(1e-9, ducha_pot_total)

    os.makedirs(os.path.dirname(ruta_out), exist_ok=True)
    out.to_csv(ruta_out, index=False)

    print(f"\n--- MPC ORÁCULO PERFECTO (H={Np}) ---")
    print(f"Demanda potencial total (kWh) : {demanda_pot_kwh:.3f}")
    print(f"Energía total atendida (kWh)  : {energia_total_atendida_kwh:.3f}")
    print(f"ENS (kWh)                     : {ENS:.3f}")
    print(f"Cobertura energética (%)      : {cobertura_energetica_pct:.2f}")
    print(f"Curtailment (kWh)             : {CURT:.3f}")
    print(f"Carga batería (kWh)           : {Charge_kwh:.3f}")
    print(f"Descarga batería (kWh)        : {Discharge_kwh:.3f}")
    print(f"Ducha servida                 : {ducha_serv_total:.3f} / {ducha_pot_total:.3f} ({ducha_serv_pct:.2f}%)")
    print(f"SOC min/mean/max              : {out['SOC_mpc'].min():.3f} / {out['SOC_mpc'].mean():.3f} / {out['SOC_mpc'].max():.3f}")
    print(f"OK -> guardado: {ruta_out}")


if __name__ == "__main__":
    mpc_ducha_soft_oracle(
        n_ocupantes=5,
        Np=12,
        solver_primary="OSQP",
        solver_fallback="SCS"
    )