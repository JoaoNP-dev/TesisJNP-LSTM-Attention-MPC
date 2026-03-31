import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import joblib

# Configuración Global
L = 24
H = 12
BATCH_SIZE = 64
EPOCHS = 150
LR = 1e-3
PATIENCE = 20
HIDDEN_DIM = 128
SEED = 2025

BASE_DIR = Path(__file__).resolve().parents[2]
NPZ_PATH = BASE_DIR / "datasets" / f"3ml_lstm_L{L}_H{H}.npz"
SCALER_Y_PATH = BASE_DIR / "modelos" / "scaler_y.pkl"
OUT_DIR = BASE_DIR / "resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / f"lstm_attention_L{L}_H{H}.pt"
HISTORY_CSV_PATH = OUT_DIR / f"history_lstm_attention_L{L}_H{H}.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 2025):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 12):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.attn = nn.Linear(hidden_dim, 1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, return_attention: bool = False):
        h_seq, _ = self.lstm(x)
        scores = self.attn(h_seq)
        alpha = torch.softmax(scores, dim=1)
        context = torch.sum(alpha * h_seq, dim=1)
        y = self.fc(context)

        if return_attention:
            return y, alpha.squeeze(-1)
        return y


def inverse_transform_multistep(arr, scaler_y):
    original_shape = arr.shape
    arr_2d = arr.reshape(-1, 1)
    arr_inv = scaler_y.inverse_transform(arr_2d)
    return arr_inv.reshape(original_shape)


def evaluate(model, loader, criterion, scaler_y=None):
    model.eval()
    total_loss = 0.0
    ys_real = []
    ps_real = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item()

            p_np = pred.cpu().numpy()
            y_np = yb.cpu().numpy()

            if scaler_y is not None:
                p_np = inverse_transform_multistep(p_np, scaler_y)
                y_np = inverse_transform_multistep(y_np, scaler_y)

            ps_real.append(p_np)
            ys_real.append(y_np)

    y_true = np.concatenate(ys_real, axis=0)
    y_pred = np.concatenate(ps_real, axis=0)

    mse_global = np.mean((y_pred - y_true) ** 2)
    mae_global = np.mean(np.abs(y_pred - y_true))
    rmse_global = np.sqrt(mse_global)

    mse_steps = np.mean((y_pred - y_true) ** 2, axis=0)
    mae_steps = np.mean(np.abs(y_pred - y_true), axis=0)
    rmse_steps = np.sqrt(mse_steps)

    return (
        total_loss / len(loader),
        mae_global,
        mse_global,
        rmse_global,
        mae_steps,
        mse_steps,
        rmse_steps
    )


def main():
    set_seed(SEED)

    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"No existe: {NPZ_PATH}")

    data = np.load(NPZ_PATH, allow_pickle=True)
    scaler_y = joblib.load(SCALER_Y_PATH)

    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32)
    X_val   = torch.tensor(data["X_val"], dtype=torch.float32)
    y_val   = torch.tensor(data["y_val"], dtype=torch.float32)
    X_test  = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test  = torch.tensor(data["y_test"], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMAttention(
        input_dim=X_train.shape[-1],
        hidden_dim=HIDDEN_DIM,
        output_dim=H
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.HuberLoss(delta=0.15)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    best_val_mse = float("inf")
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_mse": [],
        "val_rmse": [],
    }

    print("Iniciando entrenamiento: LSTM(2 layers) + Attention Multi-step")
    print(f"Features: {X_train.shape[-1]} | Horizonte: {H} | Dispositivo: {DEVICE}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        (
            val_loss,
            val_mae,
            val_mse,
            val_rmse,
            val_mae_steps,
            val_mse_steps,
            val_rmse_steps
        ) = evaluate(model, val_loader, criterion, scaler_y)

        scheduler.step(val_loss)

        train_loss_epoch = train_loss / len(train_loader)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_mse"].append(val_mse)
        history["val_rmse"].append(val_rmse)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss_epoch:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MAE: {val_mae:.4f} kWh | "
                f"Val RMSE: {val_rmse:.4f} kWh"
            )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[INFO] Early stopping en época {epoch}")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_CSV_PATH, index=False)
    print(f"\nHistorial de entrenamiento guardado en: {HISTORY_CSV_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    (
        _,
        t_mae,
        t_mse,
        t_rmse,
        t_mae_steps,
        t_mse_steps,
        t_rmse_steps
    ) = evaluate(model, test_loader, criterion, scaler_y)

    print("\nRESULTADOS FINALES (kWh/h)")
    print(f"MAE Global  : {t_mae:.4f}")
    print(f"MSE Global  : {t_mse:.4f}")
    print(f"RMSE Global : {t_rmse:.4f}")

    print("\nRMSE por horizonte:")
    for i, rmse_h in enumerate(t_rmse_steps, start=1):
        print(f"t+{i}: {rmse_h:.4f} kWh")

    print("\nMAE por horizonte:")
    for i, mae_h in enumerate(t_mae_steps, start=1):
        print(f"t+{i}: {mae_h:.4f} kWh")

    model.eval()
    with torch.no_grad():
        xb = X_test[:200].to(DEVICE)
        _, alpha = model(xb, return_attention=True)
        np.save(OUT_DIR / f"attention_weights_L{L}_H{H}.npy", alpha.cpu().numpy())

    print("\nPesos de atención guardados en resultados/")


if __name__ == "__main__":
    main()