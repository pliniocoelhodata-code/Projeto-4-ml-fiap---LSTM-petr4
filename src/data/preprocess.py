import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib

LOOKBACK = 60
HORIZON = 30

def load_close_prices(csv_path="data/raw/petr4_raw.csv"):
    df = pd.read_csv(csv_path)
    df = df.sort_values("Date")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df["Close"].values.reshape(-1, 1)


def train_val_test_split(series, train_size=0.7, val_size=0.15):
    n = len(series)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train = series[:train_end]
    val = series[train_end:val_end]
    test = series[val_end:]

    return train, val, test


def scale_series(train, val, test, scaler_path="models/scaler.pkl"):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    return train_scaled, val_scaled, test_scaled


def create_windows(series, lookback=LOOKBACK, horizon=HORIZON):
    X, y = [], []

    for i in range(len(series) - lookback - horizon):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])

    return np.array(X), np.array(y)


def preprocess_pipeline():
    series = load_close_prices()

    train, val, test = train_val_test_split(series)

    train_s, val_s, test_s = scale_series(train, val, test)

    X_train, y_train = create_windows(train_s)
    X_val, y_val = create_windows(val_s)
    X_test, y_test = create_windows(test_s)

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_pipeline()

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
