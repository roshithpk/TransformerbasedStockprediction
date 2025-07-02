import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

# --- Transformer Model Definition ---
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward=128):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_fc(src)
        src = self.transformer(src)
        output = self.output_fc(src[:, -1, :])
        return output

# --- Labeled Helper ---
def prepare_data(data, n_steps=30):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# --- Technical Indicators ---
def add_indicators(df):
    close = df["Close"]
    df["RSI"] = RSIIndicator(close).rsi()
    df["EMA_20"] = EMAIndicator(close, window=20).ema_indicator()
    df.fillna(method="bfill", inplace=True)
    return df

# --- Streamlit Function ---
def run_ai_prediction():
    st.title("🤖 Transformer Stock Predictor")

    col1, col2 = st.columns(2)
    user_stock = col1.text_input("Enter Stock Symbol", value="INFY")
    pred_days = col2.slider("Days to Forecast", 3, 10, 5)

    if st.button("🚀 Predict with Transformer"):
        ticker = user_stock.strip().upper() + ".NS"

        with st.spinner("Downloading and preparing data..."):
            df = yf.download(ticker, period="2y", progress=False)
            if df.empty:
                st.error("Stock data not found.")
                return

            df = add_indicators(df)
            features = ["Close", "RSI", "EMA_20"]

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df[features])
            X, y = prepare_data(scaled)

            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

            model = TransformerModel(input_dim=X.shape[2], d_model=64, nhead=4, num_layers=2)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(20):  # Short training for demo
                model.train()
                optimizer.zero_grad()
                output = model(X_tensor)
                loss = criterion(output, y_tensor)
                loss.backward()
                optimizer.step()

        st.success("✅ Model trained")

        # Forecasting
        last_seq = scaled[-30:]
        future_preds = []
        df_forecast = df.copy()

        model.eval()
        for _ in range(pred_days):
            inp = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)
            pred_scaled = model(inp).item()

            dummy_row = np.zeros((1, len(features)))
            dummy_row[0, 0] = pred_scaled
            next_close = scaler.inverse_transform(dummy_row)[0, 0]
            future_preds.append(next_close)

            next_date = df_forecast.index[-1] + timedelta(days=1)
            new_row = pd.DataFrame([[np.nan]*df_forecast.shape[1]], columns=df_forecast.columns, index=[next_date])
            new_row.at[next_date, "Close"] = next_close
            df_forecast = pd.concat([df_forecast, new_row])

            df_forecast = add_indicators(df_forecast)
            last_seq = scaler.transform(df_forecast[features].iloc[-30:])

        forecast_df = pd.DataFrame({
            "Date": pd.date_range(df.index[-1] + timedelta(days=1), periods=pred_days),
            "Predicted Close": np.round(future_preds, 2)
        })

        st.subheader("📅 Forecasted Prices")
        st.dataframe(forecast_df)

        st.line_chart(forecast_df.set_index("Date"))

