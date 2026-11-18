"""Leakage-safe batch and incremental features in one shared module.

History is strictly earlier than event time; equal-time rows cannot see each other.
Windows include their left boundary. Amount z-scores use sample std (ddof=1).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from math import cos, log1p, pi, sin, sqrt

import numpy as np
import pandas as pd

VELOCITY_WINDOWS = {
    "txn_count_1h": 3600,
    "txn_count_6h": 21600,
    "txn_count_24h": 86400,
    "txn_count_7d": 604800,
    "txn_amount_sum_1h": 3600,
    "txn_amount_sum_24h": 86400,
}
AMOUNT_FEATURES = ["amount_zscore", "amount_to_acct_mean_ratio", "amount_to_acct_max_ratio"]
TIME_FEATURES = ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]
CHANGE_FEATURES = ["device_changed", "email_domain_changed"]
ENGINEERED_FEATURE_NAMES = (
    list(VELOCITY_WINDOWS)
    + AMOUNT_FEATURES
    + TIME_FEATURES
    + CHANGE_FEATURES
    + ["seconds_since_last_txn", "log_amount"]
)


def _ordered(df, time_col="TransactionDT"):
    return df.sort_values(time_col, kind="stable").copy()


def compute_velocity_features(
    df, entity_col="pseudo_user_id", time_col="TransactionDT", amount_col="TransactionAmt"
):
    """Use sorted window boundaries and prefix sums, avoiding quadratic scans."""
    df = _ordered(df, time_col)
    result = np.zeros((len(df), len(VELOCITY_WINDOWS) + 1))
    result[:, -1] = -1
    all_times, all_amounts = df[time_col].to_numpy(), df[amount_col].to_numpy()
    for positions in df.groupby(entity_col, sort=False).indices.values():
        times, amounts = all_times[positions], all_amounts[positions]
        right = np.searchsorted(times, times, side="left")
        sums = np.r_[0.0, np.cumsum(amounts, dtype=float)]
        for j, (name, window) in enumerate(VELOCITY_WINDOWS.items()):
            left = np.searchsorted(times, times - window, side="left")
            result[positions, j] = (
                right - left if name.startswith("txn_count") else sums[right] - sums[left]
            )
        prior = right > 0
        result[positions[prior], -1] = times[prior] - times[right[prior] - 1]
    df[list(VELOCITY_WINDOWS) + ["seconds_since_last_txn"]] = result
    return df


def compute_amount_features(df, entity_col="pseudo_user_id", amount_col="TransactionAmt"):
    """Expanding history with stable variance, excluding all equal-time rows."""
    order = np.argsort(df["TransactionDT"].to_numpy(), kind="stable")
    df = df.iloc[order].copy()
    result = np.tile([0.0, 1.0, 1.0], (len(df), 1))
    all_amounts = df[amount_col].to_numpy(dtype=float)
    all_times = df["TransactionDT"].to_numpy()
    for positions in df.groupby(entity_col, sort=False).indices.values():
        amounts, times = all_amounts[positions], all_times[positions]
        count = len(amounts)
        means, m2, maxima = np.zeros(count + 1), np.zeros(count + 1), np.zeros(count + 1)
        for i, amount in enumerate(amounts, 1):
            delta = amount - means[i - 1]
            means[i] = means[i - 1] + delta / i
            m2[i] = m2[i - 1] + delta * (amount - means[i])
            maxima[i] = max(maxima[i - 1], amount)
        prior = np.searchsorted(times, times, side="left")
        std = np.sqrt(np.maximum(0, m2[prior] / np.maximum(prior - 1, 1)))
        result[positions, 0] = np.divide(
            amounts - means[prior], std, out=np.zeros(count), where=(prior > 1) & (std > 0)
        )
        result[positions, 1] = np.divide(
            amounts, means[prior], out=np.ones(count), where=means[prior] > 0
        )
        result[positions, 2] = np.divide(
            amounts, maxima[prior], out=np.ones(count), where=maxima[prior] > 0
        )
    df[AMOUNT_FEATURES] = result
    df["log_amount"] = np.log1p(all_amounts)
    return df.iloc[np.argsort(order)]


def compute_time_features(df):
    """Encode clock/calendar cycles; derive missing clock columns from event time."""
    df = df.copy()
    dt = pd.Timestamp("2017-12-01") + pd.to_timedelta(df["TransactionDT"], unit="s")
    hour = df["transaction_hour"] if "transaction_hour" in df else dt.dt.hour
    dow = df["transaction_dayofweek"] if "transaction_dayofweek" in df else dt.dt.dayofweek
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    df["day_sin"], df["day_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    return df


def compute_change_features(df, entity_col="pseudo_user_id"):
    """Compare to the last row at a strictly earlier time (stable arrival order)."""
    df = _ordered(df)
    result = np.zeros((len(df), 2))
    all_times = df["TransactionDT"].to_numpy()
    values_by_column = {
        column: df[column].to_numpy() for column in ["DeviceType", "P_emaildomain"] if column in df
    }
    for positions in df.groupby(entity_col, sort=False).indices.values():
        times = all_times[positions]
        previous = np.searchsorted(times, times, side="left") - 1
        valid = previous >= 0
        for j, column in enumerate(["DeviceType", "P_emaildomain"]):
            if column not in values_by_column:
                continue
            values = values_by_column[column][positions]
            current, past = values[valid], values[previous[valid]]
            known = pd.notna(current) & pd.notna(past)
            result[positions[valid][known], j] = current[known] != past[known]
    df[CHANGE_FEATURES] = result
    return df


def engineer_features(df, entity_col="pseudo_user_id", compute_velocity=True):
    """Batch path; preserve original indices and use only earlier history."""
    df = compute_amount_features(df, entity_col)
    df = compute_time_features(df)
    if compute_velocity:
        df = compute_velocity_features(df, entity_col)
        df = compute_change_features(df, entity_col)
    return df


@dataclass
class _Account:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    maximum: float = 0.0
    last_time: float | None = None
    device: object = None
    email: object = None
    pending_time: float | None = None
    pending: list = field(default_factory=list)
    windows: dict = field(
        default_factory=lambda: {w: deque() for w in sorted(set(VELOCITY_WINDOWS.values()))}
    )
    sums: dict = field(
        default_factory=lambda: {w: 0.0 for w in sorted(set(VELOCITY_WINDOWS.values()))}
    )


class OnlineFeatureEngine:
    """Score-before-update history for ordered per-account transaction replay.

    Equal-time observations stay pending until the next timestamp arrives.
    shortcut: in-memory state, add durable recovery in the streaming phase.
    """

    def __init__(self):
        self.accounts = {}

    def export_state(self, account):
        return json.dumps(asdict(self.accounts[account]), default=list, allow_nan=False)

    def restore_state(self, account, payload):
        values = json.loads(payload)
        values["windows"] = {int(k): deque(v) for k, v in values["windows"].items()}
        values["sums"] = {int(k): v for k, v in values["sums"].items()}
        self.accounts[account] = _Account(**values)

    def process(self, transaction: dict) -> dict[str, float]:
        account = transaction["pseudo_user_id"]
        time = float(transaction["TransactionDT"])
        amount = float(transaction["TransactionAmt"])
        if pd.isna(account) or not np.isfinite(time) or not np.isfinite(amount) or amount < 0:
            raise ValueError("Account, finite event time and nonnegative finite amount required")
        state = self.accounts.setdefault(account, _Account())
        if state.pending_time is not None and time < state.pending_time:
            raise ValueError("Out-of-order event: replay in event-time order per account")
        if state.pending_time is not None and time > state.pending_time:
            for prior_amount, device, email in state.pending:
                state.count += 1
                delta = prior_amount - state.mean
                state.mean += delta / state.count
                state.m2 += delta * (prior_amount - state.mean)
                state.maximum = max(state.maximum, prior_amount)
                for window, events in state.windows.items():
                    events.append((state.pending_time, prior_amount))
                    state.sums[window] += prior_amount
                state.device, state.email = device, email
            state.last_time = state.pending_time
            state.pending.clear()
        for window, events in state.windows.items():
            while events and events[0][0] < time - window:
                state.sums[window] -= events.popleft()[1]
        features = {
            name: float(len(state.windows[window]))
            if name.startswith("txn_count")
            else state.sums[window]
            for name, window in VELOCITY_WINDOWS.items()
        }
        std = sqrt(max(0, state.m2 / (state.count - 1))) if state.count > 1 else 0
        dt = pd.Timestamp("2017-12-01") + pd.Timedelta(seconds=time)
        hour = transaction.get("transaction_hour", dt.hour)
        dow = transaction.get("transaction_dayofweek", dt.dayofweek)
        device, email = transaction.get("DeviceType"), transaction.get("P_emaildomain")
        features.update(
            {
                "amount_zscore": (amount - state.mean) / std if std > 0 else 0.0,
                "amount_to_acct_mean_ratio": amount / state.mean if state.mean > 0 else 1.0,
                "amount_to_acct_max_ratio": amount / state.maximum if state.maximum > 0 else 1.0,
                "hour_sin": sin(2 * pi * hour / 24),
                "hour_cos": cos(2 * pi * hour / 24),
                "day_sin": sin(2 * pi * dow / 7),
                "day_cos": cos(2 * pi * dow / 7),
                "is_weekend": float(dow >= 5),
                "device_changed": float(
                    pd.notna(device) and pd.notna(state.device) and device != state.device
                ),
                "email_domain_changed": float(
                    pd.notna(email) and pd.notna(state.email) and email != state.email
                ),
                "seconds_since_last_txn": time - state.last_time
                if state.last_time is not None
                else -1,
                "log_amount": log1p(amount),
            }
        )
        state.pending_time = time
        state.pending.append((amount, device, email))
        return features
