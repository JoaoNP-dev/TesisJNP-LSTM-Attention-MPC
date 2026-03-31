import numpy as np
import pandas as pd
import sys
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

try:
    from utils.guardar_npz import guardar_npz
except ImportError:
    def guardar_npz(arrays, nombre_archivo, subcarpeta):
        ruta = os.path.join(BASE_DIR, subcarpeta, nombre_archivo)
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        np.savez(ruta, **arrays)
        print(f"Archivo guardado en: {ruta}")

RUTA_ENTRADA = os.path.join(BASE_DIR, "resultados", "df_maestro_2025_N5.csv")

# Configuración del modelo LSTM
L = 24   # Ventana de entrada (Lookback)
H = 12   # Predice 12 horas futuras
TARGET = "demanda_casa_hora"

FEATURES_NUM = [
    "ghi_wm2", "temp_c",
    "SOC_bateria_base",
    "energia_fv_kwh",
    "occ_total",
    "ev_presente", "ev_cargable",
    "hora_sin", "hora_cos",
    "dow_sin", "dow_cos"
]

FEATURES_CAT = [
    "tipo_dia", "bloque_horario"
]

def one_hot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=cols, drop_first=False, dtype=float)

def build_sequences(X2d: np.ndarray, y1d: np.ndarray, L: int, H: int):
    X_seq, y_seq = [], []
    n = len(X2d)

    for t in range(L - 1, n - H):
        X_seq.append(X2d[t - L + 1:t + 1, :])
        y_seq.append(y1d[t + 1:t + H + 1])  # Salida multi-step directa

    return np.array(X_seq), np.array(y_seq)

def main():
    if not os.path.exists(RUTA_ENTRADA):
        raise FileNotFoundError(f"No se encontró el archivo: {RUTA_ENTRADA}")

    df = pd.read_csv(RUTA_ENTRADA, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df_ml = df[["timestamp", TARGET] + FEATURES_NUM + FEATURES_CAT].copy()
    df_ml = one_hot(df_ml, FEATURES_CAT)

    feature_cols = [c for c in df_ml.columns if c not in ["timestamp"]]

    ts = df_ml["timestamp"]
    train_mask = ts < "2025-10-01"

    # Escalado de X
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_x.fit(df_ml.loc[train_mask, feature_cols])
    X_all_2d = scaler_x.transform(df_ml[feature_cols])

    # Escalado de y
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(df_ml.loc[train_mask, [TARGET]])
    y_all_1d = scaler_y.transform(df_ml[[TARGET]]).flatten()

    # Construcción de secuencias multi-step
    X_seq, y_seq = build_sequences(X_all_2d, y_all_1d, L=L, H=H)

    # Timestamp de referencia: primera hora predicha
    ts_y = df_ml["timestamp"].to_numpy()
    ts_seq = np.array([ts_y[t + 1] for t in range(L - 1, len(ts_y) - H)])

    train_idx = ts_seq < np.datetime64("2025-10-01")
    val_idx   = (ts_seq >= np.datetime64("2025-10-01")) & (ts_seq < np.datetime64("2025-12-01"))
    test_idx  = ts_seq >= np.datetime64("2025-12-01")

    X_train, y_train = X_seq[train_idx], y_seq[train_idx]
    X_val, y_val     = X_seq[val_idx], y_seq[val_idx]
    X_test, y_test   = X_seq[test_idx], y_seq[test_idx]

    modelos_dir = os.path.join(BASE_DIR, "modelos")
    os.makedirs(modelos_dir, exist_ok=True)

    joblib.dump(scaler_x, os.path.join(modelos_dir, "scaler_x.pkl"))
    joblib.dump(scaler_y, os.path.join(modelos_dir, "scaler_y.pkl"))

    guardar_npz(
        arrays={
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "feature_cols": np.array(feature_cols, dtype=object)
        },
        nombre_archivo=f"3ml_lstm_L{L}_H{H}.npz",
        subcarpeta="datasets"
    )

    print("Dataset entrenamiento finalizado.")
    print(f"X_train shape: {X_train.shape} (Ventana de {L}h)")
    print(f"y_train shape: {y_train.shape} (Horizonte de {H}h)")
    print(f"Rango de Target: [{y_train.min():.2f} - {y_train.max():.2f}]")

if __name__ == "__main__":
    main()