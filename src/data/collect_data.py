import yfinance as yf
import pandas as pd
from datetime import datetime

def collect_petr4_data(
    ticker="PETR4.SA",
    start_date="2015-01-01",
    end_date=None,
    save_path="data/raw/petr4_raw.csv"
):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False
    )

    df.reset_index(inplace=True)
    df.to_csv(save_path, index=False)

    return df


if __name__ == "__main__":
    df = collect_petr4_data()
    print(df.head())
    print(df.tail())
    print(f"Total de registros: {len(df)}")
