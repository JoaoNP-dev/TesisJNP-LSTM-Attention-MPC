# MPC DUCHA SOFT + LSTM attention (PyTorch) 12 horas
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# Validaciones
try:
    import cvxpy as cp
except Exception as e:
    raise ImportError("No está instalado cvxpy.") from e

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise ImportError("No está instalado torch.") from e


# =========================================================
# UTILS Y PREPROCESAMIENTO
# =========================================================
def one_hot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=cols, drop_first=False)


def fit_scaler_train(df_feat: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xtr = df_feat.loc[train_mask].to_numpy(dtype=float)
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu, sigma


def transform_scaler(df_feat: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    X = df_feat.to_numpy(dtype=float)
    return (X - mu) / sigma


def build_sequences_3d(X2d: np.ndarray, L: int) -> np.ndarray:
    n, f = X2d.shape
    if n < L:
        raise ValueError(f"No hay suficientes muestras para L={L}. N={n}")
    X_seq = np.zeros((n - L + 1, L, f), dtype=np.float32)
    for i in range(L - 1, n):
        X_seq[i - (L - 1)] = X2d[i - L + 1:i + 1, :]
    return X_seq


# =========================================================
# MODELO LSTM MULTI-STEP CON ATENCIÓN
# =========================================================
class LSTMAttentionFlexible(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        head_type: str = "single",
        head_hidden: int | None = None,
        output_dim: int = 12
    ):
        super().__init__()
        self.head_type = head_type
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.attn = nn.Linear(hidden_dim, 1)

        if head_type == "sequential":
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, output_dim)
            )
        elif head_type == "single":
            self.fc = nn.Linear(hidden_dim, output_dim)
        elif head_type == "two_layer":
            h_hid = head_hidden if head_hidden else hidden_dim
            self.fc1 = nn.Linear(hidden_dim, h_hid)
            self.fc2 = nn.Linear(h_hid, output_dim)
        else:
            raise ValueError(f"head_type no soportado: {head_type}")

    def forward(self, x: torch.Tensor):
        h, _ = self.lstm(x)
        scores = self.attn(h).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(h * weights.unsqueeze(-1), dim=1)

        if self.head_type in ["sequential", "single"]:
            y = self.fc(context)
        else:
            y = self.fc2(torch.relu(self.fc1(context)))

        if y.shape[-1] == 1:
            y = y.squeeze(-1)
        return y, weights


def _infer_head_from_state_dict(state: dict) -> tuple[str, int | None, int]:
    if any("fc.0.weight" in k for k in state.keys()):
        out_dim = state["fc.3.weight"].shape[0]
        return "sequential", None, out_dim
    if "fc.weight" in state:
        out_dim = state["fc.weight"].shape[0]
        return "single", None, out_dim
    if "fc1.weight" in state:
        h_hid = int(state["fc1.weight"].shape[0])
        out_dim = state["fc2.weight"].shape[0]
        return "two_layer", h_hid, out_dim
    raise RuntimeError("No se pudo inferir el head del modelo.")


def _infer_arch_from_state_dict(state: dict) -> tuple[int, int, int]:
    key_ih = [k for k in state.keys() if "weight_ih_l0" in k][0]
    four_h, input_dim = state[key_ih].shape
    hidden_dim = four_h // 4
    num_layers = len([k for k in state.keys() if "weight_ih_l" in k])
    return input_dim, hidden_dim, num_layers


def load_model_from_pt(path_pt: str, device: str = "cpu") -> tuple[nn.Module, int]:
    state = torch.load(path_pt, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    in_d, hid_d, n_lay = _infer_arch_from_state_dict(state)
    h_type, h_hid, out_dim = _infer_head_from_state_dict(state)

    model = LSTMAttentionFlexible(in_d, hid_d, n_lay, h_type, h_hid, out_dim)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, in_d


# =========================================================
# INFERENCIA LSTM
# =========================================================
def predecir_demanda_lstm(
    df: pd.DataFrame,
    ruta_modelo_pt: str,
    ruta_npz: str,
    L: int = 24,
    H: int = 12,
    target_col: str = "demanda_casa_hora",
    cat_cols: list[str] | None = None,
    device: str = "cpu",
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:

    if cat_cols is None:
        cat_cols = ["tipo_dia", "bloque_horario"]

    npz = np.load(ruta_npz, allow_pickle=True)
    feature_cols_npz = list(npz["feature_cols"])

    df_ml = df[["timestamp", target_col] + cat_cols + [
        "ghi_wm2", "temp_c", "SOC_bateria_base", "energia_fv_kwh",
        "occ_total", "ev_presente", "ev_cargable",
        "hora", "dia_semana_iso", "mes",
        "dow_sin", "dow_cos", "hora_sin", "hora_cos"
    ]].copy()

    df_ml = df_ml.loc[:, ~df_ml.columns.duplicated()]
    df_ml = df_ml[[c for c in df_ml.columns if c in df.columns or c in ["timestamp", target_col] or c in cat_cols]]
    df_ml = one_hot(df_ml, cat_cols)
    df_feat = df_ml.reindex(columns=feature_cols_npz, fill_value=0.0)

    ts = pd.to_datetime(df["timestamp"]).to_numpy()
    train_mask = ts < np.datetime64("2025-10-01")
    mu, sigma = fit_scaler_train(df_feat, train_mask=train_mask)
    X_all = transform_scaler(df_feat, mu, sigma)

    model, _ = load_model_from_pt(ruta_modelo_pt, device=device)

    X_seq = build_sequences_3d(X_all, L=L)
    n_seq = X_seq.shape[0]

    y_hat = np.zeros((n_seq, H), dtype=np.float32)
    attn_w = np.zeros((n_seq, L), dtype=np.float32)

    with torch.no_grad():
        for i0 in range(0, n_seq, batch_size):
            i1 = min(i0 + batch_size, n_seq)
            xb = torch.from_numpy(X_seq[i0:i1]).to(device)
            yb, wb = model(xb)
            if yb.ndim == 1:
                yb = yb.unsqueeze(1)
            y_hat[i0:i1] = yb.detach().cpu().numpy().astype(np.float32)
            attn_w[i0:i1] = wb.detach().cpu().numpy().astype(np.float32)

    return y_hat, attn_w


# =========================================================
# MPC
# =========================================================
def mpc_ducha_soft_lstm(
    df_in: pd.DataFrame,
    demanda_pred_matrix: np.ndarray,
    L: int = 24,
    H: int = 12,
    horizon: int = 12,
    capacidad_bat_kwh: float = 10.0,
    P_inv_max: float = 5.0,
    soc_min: float = 0.10,
    soc_max: float = 1.00,
    eta_c: float = 0.95,
    eta_d: float = 0.95,
    w_unmet: float = 5000.0,
    w_shed: float = 10.0,
    w_curt: float = 0.001,
    w_bat: float = 0.05,
    w_ducha: float = 15.0,
    solver_primary: str = "OSQP",
    solver_fallback: str = "SCS",
    verbose: bool = False,
) -> pd.DataFrame:

    df = df_in.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    N = len(df)

    d_crit = df["demanda_critica_hora"].to_numpy(dtype=float)
    d_flex = df["demanda_flexible_hora"].to_numpy(dtype=float)
    d_ctrl = df["demanda_controlable_hora"].to_numpy(dtype=float)

    # Fallback
    fallback_data = d_crit + d_flex + d_ctrl

    Pred_Matrix = np.full((N, H), np.nan, dtype=float)
    n_seq = len(demanda_pred_matrix)

    for i in range(n_seq):
        t_obj = i + L
        if t_obj < N:
            Pred_Matrix[t_obj, :] = demanda_pred_matrix[i, :]

    for t in range(N):
        if np.isnan(Pred_Matrix[t, 0]):
            for k in range(H):
                idx = min(t + k, N - 1)
                Pred_Matrix[t, k] = fallback_data[idx]

    fv = df["energia_fv_kwh"].to_numpy(dtype=float)
    ducha_pot = df["ducha_potencial_kwh"].to_numpy(dtype=float)
    ducha_evt = df["ducha_evento"].to_numpy(dtype=int)

    soc0 = float(df["SOC_bateria_base"].iat[0])
    E0 = soc0 * capacidad_bat_kwh

    SOC_mpc = np.zeros(N, dtype=float)
    P_c = np.zeros(N, dtype=float)
    P_d = np.zeros(N, dtype=float)
    ENS = np.zeros(N, dtype=float)
    Curt = np.zeros(N, dtype=float)
    u_shed = np.ones(N, dtype=float)
    u_ducha = np.zeros(N, dtype=float)
    ducha_servida = np.zeros(N, dtype=float)
    load_servida = np.zeros(N, dtype=float)

    # Base física homogénea para métricas finales
    demanda_total_pot = d_crit + d_flex + d_ctrl + ducha_pot

    E = E0
    SOC_mpc[0] = np.clip(E / capacidad_bat_kwh, soc_min, soc_max)

    for t in range(N):
        Hh = min(horizon, N - t)

        uc = cp.Variable(Hh, nonneg=True)
        ud = cp.Variable(Hh, nonneg=True)
        e = cp.Variable(Hh + 1)
        ens = cp.Variable(Hh, nonneg=True)
        curt = cp.Variable(Hh, nonneg=True)
        u = cp.Variable(Hh)
        v = cp.Variable(Hh)

        cons = []
        cons += [u >= 0, u <= 1]
        cons += [v >= 0, v <= 1]
        cons += [e[0] == E]

        cons += [uc <= P_inv_max]
        cons += [ud <= P_inv_max]

        E_min = soc_min * capacidad_bat_kwh
        E_max = soc_max * capacidad_bat_kwh
        cons += [e >= E_min, e <= E_max]
        cons += [e[1:] == e[:-1] + eta_c * uc - (1.0 / max(eta_d, 1e-6)) * ud]

        for k in range(Hh):
            idx = t + k

            pred_k = Pred_Matrix[t, k]
            base_inflexible = max(0.0, pred_k - d_ctrl[idx])

            demanda_serv = base_inflexible + u[k] * d_ctrl[idx] + v[k] * ducha_pot[idx]

            if ducha_evt[idx] == 0 or ducha_pot[idx] <= 1e-12:
                cons += [v[k] == 0]

            cons += [fv[idx] + ud[k] == demanda_serv + uc[k] + curt[k] - ens[k]]
            cons += [ens[k] <= demanda_serv + 1e-6]

        obj = 0
        obj += w_unmet * cp.sum(ens)
        obj += w_curt * cp.sum(curt)
        obj += w_bat * cp.sum(uc + ud)
        obj += w_shed * cp.sum((1 - u) * d_ctrl[t:t + Hh])
        obj += w_ducha * cp.sum((1 - v) * ducha_pot[t:t + Hh])

        prob = cp.Problem(cp.Minimize(obj), cons)

        solved = False
        for sol in (solver_primary, solver_fallback):
            try:
                prob.solve(solver=getattr(cp, sol), verbose=verbose, warm_start=True)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    solved = True
                    break
            except Exception:
                continue

        base_pred_t = max(0.0, Pred_Matrix[t, 0] - d_ctrl[t])

        if not solved:
            uc0, ud0, v0 = 0.0, 0.0, 0.0
            u0 = 1.0
            ens0 = max(0.0, (base_pred_t + d_ctrl[t]) - fv[t] - max(0.0, (E - E_min) * eta_d))
            curt0 = max(0.0, fv[t] - (base_pred_t + d_ctrl[t]))
        else:
            uc0 = float(uc.value[0])
            ud0 = float(ud.value[0])
            ens0 = float(ens.value[0])
            curt0 = float(curt.value[0])
            u0 = float(u.value[0])
            v0 = float(v.value[0])

        E = float(np.clip(E + eta_c * uc0 - (1.0 / max(eta_d, 1e-6)) * ud0, E_min, E_max))

        SOC_mpc[t] = float(np.clip(E / capacidad_bat_kwh, soc_min, soc_max))
        P_c[t] = max(0.0, uc0)
        P_d[t] = max(0.0, ud0)
        ENS[t] = max(0.0, ens0)
        Curt[t] = max(0.0, curt0)
        u_shed[t] = float(np.clip(u0, 0.0, 1.0))
        u_ducha[t] = float(np.clip(v0, 0.0, 1.0))

        ducha_servida[t] = u_ducha[t] * ducha_pot[t]

        # Energía realmente atendida sobre base física homogénea
        load_servida[t] = (
            d_crit[t]
            + d_flex[t]
            + u_shed[t] * d_ctrl[t]
            + ducha_servida[t]
            - ENS[t]
        )

    df["SOC_mpc"] = SOC_mpc
    df["bat_charge_kwh"] = P_c
    df["bat_discharge_kwh"] = P_d
    df["ENS_total_kwh"] = ENS
    df["Curt_kwh"] = Curt
    df["u_shed"] = u_shed
    df["u_ducha"] = u_ducha
    df["ducha_servida_kwh"] = ducha_servida
    df["demanda_total_potencial_kwh"] = demanda_total_pot
    df["demanda_servida_kwh"] = load_servida

    return df


# =========================================================
# MAIN
# =========================================================
def main():
    print("MPC + LSTM (H=12 Multi-step)")

    N_OCUP = 5
    RUTA_IN = os.path.join(BASE_DIR, "resultados", f"df_mpc_ready_ducha_2025_N{N_OCUP}.csv")
    RUTA_OUT = os.path.join(BASE_DIR, "resultados", f"df_mpc_ducha_soft_lstm_2025_N{N_OCUP}_H12.csv")

    RUTA_MODEL = os.path.join(BASE_DIR, "resultados", "lstm_attention_L24_H12.pt")
    RUTA_NPZ = os.path.join(BASE_DIR, "datasets", "3ml_lstm_L24_H12.npz")

    L = 24
    H = 12
    device = "cpu"

    df = pd.read_csv(RUTA_IN, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    y_hat_matrix, att_w = predecir_demanda_lstm(
        df=df,
        ruta_modelo_pt=RUTA_MODEL,
        ruta_npz=RUTA_NPZ,
        L=L,
        H=H,
        target_col="demanda_casa_hora",
        cat_cols=["tipo_dia", "bloque_horario"],
        device=device,
        batch_size=512
    )

    import joblib
    RUTA_SCALER_Y = os.path.join(BASE_DIR, "modelos", "scaler_y.pkl")
    scaler_y = joblib.load(RUTA_SCALER_Y)

    y_hat_real_matrix = scaler_y.inverse_transform(y_hat_matrix.reshape(-1, 1)).reshape(-1, H)
    y_hat_real_matrix = np.maximum(0.0, y_hat_real_matrix)

    df_out = mpc_ducha_soft_lstm(
        df_in=df,
        demanda_pred_matrix=y_hat_real_matrix,
        L=L,
        H=H,
        horizon=12,
        capacidad_bat_kwh=10.0,
        P_inv_max=5.0,
        soc_min=0.10,
        soc_max=1.00,
        eta_c=0.95,
        eta_d=0.95,
        w_unmet=5000.0,
        w_shed=10.0,
        w_curt=0.001,
        w_bat=0.05,
        w_ducha=15.0,
        solver_primary="OSQP",
        solver_fallback="SCS",
        verbose=False
    )

    ENS_kwh = float(df_out["ENS_total_kwh"].sum())
    Curt_kwh = float(df_out["Curt_kwh"].sum())
    Charge_kwh = float(df_out["bat_charge_kwh"].sum())
    Discharge_kwh = float(df_out["bat_discharge_kwh"].sum())

    demanda_pot_kwh = float(df_out["demanda_total_potencial_kwh"].sum())
    energia_total_atendida_kwh = float(df_out["demanda_servida_kwh"].sum())

    cobertura_energetica_pct = 100.0 * (
        energia_total_atendida_kwh / max(demanda_pot_kwh, 1e-9)
    )

    ducha_serv = float(df_out["ducha_servida_kwh"].sum())
    ducha_pot_total = float(df_out["ducha_potencial_kwh"].sum())

    print("\n--- MPC DUCHA SOFT + LSTM (H=12) ---")
    print(f"Demanda potencial total (kWh) : {demanda_pot_kwh:.3f}")
    print(f"Energía total atendida (kWh)  : {energia_total_atendida_kwh:.3f}")
    print(f"ENS (kWh)                     : {ENS_kwh:.3f}")
    print(f"Cobertura energética (%)      : {cobertura_energetica_pct:.2f}")
    print(f"Curt (kWh)                    : {Curt_kwh:.3f}")
    print(f"Carga Batería (kWh)           : {Charge_kwh:.3f}")
    print(f"Descarga Batería (kWh)        : {Discharge_kwh:.3f}")

    if ducha_pot_total > 1e-9:
        print(f"Ducha servida                 : {ducha_serv:.3f} / {ducha_pot_total:.3f} ({100 * ducha_serv / ducha_pot_total:.2f}%)")

    print(
        f"SOC min/mean/max              : "
        f"{df_out['SOC_mpc'].min():.3f} / "
        f"{df_out['SOC_mpc'].mean():.3f} / "
        f"{df_out['SOC_mpc'].max():.3f}"
    )

    os.makedirs(os.path.dirname(RUTA_OUT), exist_ok=True)
    df_out.to_csv(RUTA_OUT, index=False)
    print(f"OK -> guardado: {RUTA_OUT}")


if __name__ == "__main__":
    main()