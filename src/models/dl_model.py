import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List
# from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


# -------------------------
# Sliding Window Dataset
# -------------------------
class StreamingSlidingWindowDataset(Dataset):
    """
    Streaming sliding window dataset for multiple currencies.
    Produces seq2one samples: last timestep is target.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols=["y_high", "y_low"],
        seq_len=60,
        fill_value=0.0,
        currency_order=None,
    ):
        df = df.copy()
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.sort_values(["open_time", "currency"]).reset_index(drop=True)
        self.df = df
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.fill_value = fill_value
        self.currencies = currency_order or sorted(df["currency"].unique())
        self.n_currencies = len(self.currencies)
        self.times = df["open_time"].sort_values().unique()
        self.T = len(self.times)

        # Precompute valid start indices where target is available
        target_mask = (
            df.pivot(index="open_time", columns="currency", values=target_cols[0])
            .reindex(self.times, columns=self.currencies)
            .fillna(0)
            .values
        )
        target_mask = (target_mask != 0).astype(np.uint8)
        self.starts = np.where(target_mask[seq_len:].any(axis=1))[0]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        t0 = self.starts[idx]
        t_target = t0 + self.seq_len
        times_window = self.times[t0:t_target]

        df_win = self.df[self.df["open_time"].isin(times_window)].copy()

        # Features
        X_feat = np.zeros(
            (self.seq_len, self.n_currencies, len(self.feature_cols)), dtype=np.float32
        )
        for f_i, feat in enumerate(self.feature_cols):
            pivot_f = (
                df_win.pivot(index="open_time", columns="currency", values=feat)
                .reindex(times_window, columns=self.currencies)
                .fillna(self.fill_value)
            )
            X_feat[:, :, f_i] = pivot_f.values

        # Targets: last timestamp in window
        target_time = (
            self.times[t_target] if t_target < len(self.times) else self.times[-1]
        )
        df_target = self.df[self.df["open_time"] == target_time]

        y_high = (
            df_target.pivot(
                index="open_time", columns="currency", values=self.target_cols[0]
            )
            .reindex([target_time], columns=self.currencies, fill_value=0)
            .values.astype(np.float32)
        )
        y_low = (
            df_target.pivot(
                index="open_time", columns="currency", values=self.target_cols[1]
            )
            .reindex([target_time], columns=self.currencies, fill_value=0)
            .values.astype(np.float32)
        )
        Y = np.stack([y_high, y_low], axis=-1)[0]  # (C, 2)

        mask = ((Y[..., 0] != 0) & (Y[..., 1] != 0)).astype(np.uint8)

        return {
            "X": torch.from_numpy(X_feat),
            "y_reg": torch.from_numpy(Y),
            "target_mask": torch.from_numpy(mask),
        }


# -------------------------
# Temporal CNN + MLP attention
# -------------------------
class TemporalCNNCrossCurrency(nn.Module):
    def __init__(self, n_features, hidden_dim=128, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_features, hidden_dim, kernel_size, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reg_head = nn.Linear(hidden_dim, 2)

    def forward(self, X, avail_mask=None):
        """
        X: (batch, seq_len, C, F)
        Returns: y_reg (batch, C, 2)
        """
        b, seq_len, C, F = X.shape
        X_reshaped = X.permute(0, 2, 3, 1).reshape(b * C, F, seq_len)  # (b*C, F, seq)
        if avail_mask is not None:
            av = avail_mask.permute(0, 2, 1).reshape(b * C, 1, seq_len)
            X_reshaped = X_reshaped * av
        h = self.relu(self.conv1(X_reshaped))
        h = self.dropout(self.relu(self.conv2(h)))
        emb = h[:, :, -1].view(b, C, -1)  # (b,C,hidden)
        attn_out = self.attn_mlp(emb)
        post = self.dropout(emb + attn_out)
        y_reg = self.reg_head(post)
        return y_reg


# -------------------------
# Loss
# -------------------------
def masked_mse_loss(y_pred, y_true, mask):
    mask_f = mask.float().to(y_pred.device)
    mse = ((y_pred - y_true) ** 2).mean(-1)
    smape = (2 * abs((y_pred - y_true) / (y_true + y_pred + 1e-8))).mean(-1)
    loss = (mse * mask_f).sum() / (mask_f.sum() + 1e-8)
    smape = (smape * mask_f).sum() / (mask_f.sum() + 1e-8)
    return loss, smape


# -------------------------
# Training function
# -------------------------
def train_model(
    df,
    df_val,
    feature_cols,
    seq_len=60,
    batch_size=64,
    epochs=5,
    lr=1e-3,
    device=None,
    num_workers=4,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Datasets and loaders
    # -----------------------------
    train_dataset = StreamingSlidingWindowDataset(df, feature_cols, seq_len=seq_len)
    val_dataset = StreamingSlidingWindowDataset(df_val, feature_cols, seq_len=seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = TemporalCNNCrossCurrency(n_features=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler()

    print("\n🔹 Training the model...")

    for epoch in range(epochs):
        model.train()
        total_loss, total_smape = 0, 0

        for i, batch in enumerate(train_loader):
            print(
                f"🔹 Epoch {epoch + 1}/{epochs} | Batch {i + 1}/{len(train_loader)}...",
                end="\r",
            )
            X = batch["X"].to(device)
            y_reg = batch["y_reg"].to(device)
            mask = batch["target_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device):
                y_pred = model(X)
                loss, smape = masked_mse_loss(y_pred, y_reg, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_smape += smape.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_mape = total_smape / len(train_loader)

        model.eval()
        val_loss, val_smape = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                X = batch["X"].to(device)
                y_reg = batch["y_reg"].to(device)
                mask = batch["target_mask"].to(device)

                with torch.amp.autocast(device_type=device):
                    y_pred = model(X)
                    loss, smape = masked_mse_loss(y_pred, y_reg, mask)

                val_loss += loss.item()
                val_smape += smape.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mape = val_smape / len(val_loader)

        print(
            f"✅ Epoch {epoch + 1:02}/{epochs:02} | "
            f"Train Loss: {avg_train_loss:>10.4f}  (MAPE: {avg_train_mape:>6.2%}) | "
            f"Val Loss:   {avg_val_loss:>10.4f}  (MAPE: {avg_val_mape:>6.2%})"
        )

    return model, train_dataset


# -------------------------
# Prediction function
# -------------------------
def predict(model, df, feature_cols, seq_len=60, batch_size=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StreamingSlidingWindowDataset(df, feature_cols, seq_len=seq_len)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    model.eval()
    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            X = batch["X"].to(device)
            y_reg = batch["y_reg"].cpu().numpy()
            mask = batch["target_mask"].cpu().numpy()
            y_pred = model(X).cpu().numpy()
            for i in range(X.shape[0]):
                t0_idx = dataset.starts[batch_idx * batch_size + i]
                t_target_idx = t0_idx + seq_len
                open_time = (
                    dataset.times[t_target_idx]
                    if t_target_idx < len(dataset.times)
                    else dataset.times[-1]
                )
                for c_idx, currency in enumerate(dataset.currencies):
                    if mask[i, c_idx]:
                        rows.append(
                            {
                                "open_time": open_time,
                                "currency": currency,
                                "y_high": y_reg[i, c_idx, 0],
                                "y_low": y_reg[i, c_idx, 1],
                                "y_high_pred": y_pred[i, c_idx, 0],
                                "y_low_pred": y_pred[i, c_idx, 1],
                            }
                        )
    return pd.DataFrame(rows)
