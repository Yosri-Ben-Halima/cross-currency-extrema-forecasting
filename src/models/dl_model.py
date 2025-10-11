import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # force use of GPU 0 only

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List
from utils.helpers import downcast_numeric_columns


# -------------------------
# region: Streaming Dataset
# -------------------------
class StreamingSlidingWindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols=["y_high", "y_low"],
        meta_label_col="meta_label",
        seq_len=60,
        fill_value=0.0,
        currency_order=None,
    ):
        # --- Prepare dataframe
        df = downcast_numeric_columns(df.copy())
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.sort_values(["open_time", "currency"]).reset_index(drop=True)

        # --- Save params
        self.df = df
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.meta_label_col = meta_label_col
        self.fill_value = fill_value

        # --- Currency and time axis
        self.currencies = currency_order or sorted(df["currency"].unique())
        self.n_currencies = len(self.currencies)
        self.times = df["open_time"].sort_values().unique()
        self.T = len(self.times)

        # --- Compute valid start indices
        target_mask = (
            df.pivot(index="open_time", columns="currency", values=target_cols[0])
            .reindex(self.times, columns=self.currencies)
            .fillna(0)
            .values
        )
        target_mask = (target_mask != 0).astype(np.uint8)
        # Valid rows that have data after seq_len
        self.starts = np.where(target_mask[seq_len:].any(axis=1))[0]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        t0 = self.starts[idx]
        t_target = t0 + self.seq_len
        times_window = self.times[t0:t_target]

        df_win = self.df[self.df["open_time"].isin(times_window)].copy()

        # --- Build feature tensor safely
        X_feat = np.zeros(
            (self.seq_len, len(self.currencies), len(self.feature_cols)),
            dtype=np.float32,
        )
        for f_i, feat in enumerate(self.feature_cols):
            pivot_f = (
                df_win.pivot(index="open_time", columns="currency", values=feat)
                .reindex(times_window, columns=self.currencies)
                .fillna(self.fill_value)
            )
            X_feat[:, :, f_i] = pivot_f.values

        # --- Targets (next timestamp)
        if t_target < len(self.times):
            target_time = self.times[t_target]
        else:
            target_time = self.times[-1]

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

        # --- Meta labels
        if self.meta_label_col in self.df.columns:
            y_meta = (
                df_target.pivot(
                    index="open_time", columns="currency", values=self.meta_label_col
                )
                .reindex([target_time], columns=self.currencies, fill_value=-1)
                .values.astype(np.int64)[0]
            )
        else:
            y_meta = np.full(len(self.currencies), -1, dtype=np.int64)

        # --- Mask (1 = valid)
        mask = ((Y[..., 0] != 0) & (Y[..., 1] != 0)).astype(np.uint8)

        return {
            "X": torch.from_numpy(X_feat),
            "y_reg": torch.from_numpy(Y),
            "y_meta": torch.from_numpy(y_meta),
            "target_mask": torch.from_numpy(mask),
        }


# -------------------------
# region: Temporal CNN + MLP attention
# -------------------------
class TemporalCNNCrossCurrency(nn.Module):
    def __init__(
        self, n_features, hidden_dim=128, kernel_size=3, cls_num_classes=3, dropout=0.1
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # Temporal CNN per currency (shared)
        self.conv1 = nn.Conv1d(
            n_features, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # MLP cross-currency attention
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Heads
        self.reg_head = nn.Linear(hidden_dim, 2)
        self.cls_head = nn.Linear(hidden_dim, cls_num_classes)

    def forward(self, X, avail_mask=None):
        """
        X: (batch, seq_len, C, F)
        """
        b, seq_len, C, F = X.shape
        # CNN expects (batch*C, F, seq_len)
        X_reshaped = X.permute(0, 2, 3, 1).reshape(b * C, F, seq_len)  # (b*C, F, seq)
        if avail_mask is not None:
            av = avail_mask.permute(0, 2, 1).reshape(b * C, 1, seq_len)
            X_reshaped = X_reshaped * av

        h = self.relu(self.conv1(X_reshaped))
        h = self.dropout(self.relu(self.conv2(h)))
        # take last timestep
        emb = h[:, :, -1]  # (b*C, hidden)
        emb = emb.view(b, C, -1)  # (b,C,hidden)

        # Cross-currency MLP attention
        attn_out = self.attn_mlp(emb)  # (b,C,hidden)
        post = self.dropout(attn_out + emb)

        y_reg = self.reg_head(post)
        y_cls = self.cls_head(post)
        return y_reg, y_cls


# -------------------------
# Loss helper
# -------------------------
def multitask_loss_masked(
    y_reg_pred, y_reg_true, y_cls_logits, y_cls_true, mask, alpha=1.0
):
    device = y_reg_pred.device
    mask_f = mask.float().to(device)
    mse = ((y_reg_pred - y_reg_true) ** 2).mean(-1)
    reg_loss = (mse * mask_f).sum() / (mask_f.sum() + 1e-8)

    valid_pos = (mask_f == 1) & (y_cls_true >= 0)
    if valid_pos.sum() > 0:
        logits_flat = y_cls_logits[valid_pos.bool()]
        labels_flat = y_cls_true[valid_pos.bool()].long().to(device)
        ce = nn.CrossEntropyLoss()
        cls_loss = ce(logits_flat, labels_flat)
    else:
        cls_loss = torch.tensor(0.0, device=device)
    loss = 0.5 * reg_loss + alpha * cls_loss
    return loss, reg_loss.item(), cls_loss.item()


# -------------------------
# region: Training function
# -------------------------
def train_model(
    df,
    feature_cols,
    seq_len=60,
    batch_size=64,
    epochs=5,
    lr=1e-3,
    alpha=0.5,
    # device=None,
    num_workers=4,
    grad_accum_steps=1,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("🔹 Preparing data slides...")
    dataset = StreamingSlidingWindowDataset(df, feature_cols, seq_len=seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    print("✅ Created data slides.")
    model = TemporalCNNCrossCurrency(n_features=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device)
    model.train()
    print("🔹 Training the model...")
    for epoch in range(epochs):
        total_loss, total_reg, total_cls = 0.0, 0.0, 0.0
        optimizer.zero_grad()
        print("zabab fel kabab")
        for i, batch in enumerate(dataloader):
            print(
                f"🔹 Epoch {epoch + 1}/{epochs}: Batch {i + 1}/{len(dataloader)}",
                end="\r",
            )
            X = batch["X"].to(device)
            y_reg = batch["y_reg"].to(device)
            y_meta = batch["y_meta"].to(device)
            mask = batch["target_mask"].to(device)

            with torch.amp.autocast(device):
                y_reg_pred, y_cls_logits = model(X)
                loss, reg_l, cls_l = multitask_loss_masked(
                    y_reg_pred, y_reg, y_cls_logits, y_meta, mask, alpha
                )

            scaler.scale(loss / grad_accum_steps).backward()

            if (i + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_reg += reg_l
            total_cls += cls_l

        n_batches = len(dataloader)
        print(
            f"✅ Epoch {epoch + 1}/{epochs} | Loss {total_loss / n_batches:.3f}"  # | Reg {total_reg / n_batches:.6f} | Cls {total_cls / n_batches:.6f}"
        )

    return model, dataset


# -------------------------
# region: Prediction utility
# -------------------------
def predict(model, test, feature_cols, seq_len, batch_size=64, device=None):
    """
    Generate predictions from a trained model and prepare a DataFrame compatible with Evaluator.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StreamingSlidingWindowDataset(
        test, feature_cols=feature_cols, seq_len=seq_len
    )
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

            y_reg_pred, _ = model(X)
            y_reg_pred = y_reg_pred.cpu().numpy()

            # iterate over batch and currencies
            for i in range(X.shape[0]):
                t0_idx = dataset.starts[batch_idx * batch_size + i]
                t_target_idx = t0_idx + dataset.seq_len
                open_time = dataset.times[t_target_idx]
                for c_idx, currency in enumerate(dataset.currencies):
                    if mask[i, c_idx]:  # only include valid targets
                        rows.append(
                            {
                                "open_time": open_time,
                                "currency": currency,
                                "y_high": y_reg[i, c_idx, 0],
                                "y_low": y_reg[i, c_idx, 1],
                                "y_high_pred": y_reg_pred[i, c_idx, 0],
                                "y_low_pred": y_reg_pred[i, c_idx, 1],
                            }
                        )

    df_preds = pd.DataFrame(rows)
    return df_preds
