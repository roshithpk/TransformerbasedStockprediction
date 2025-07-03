import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from st_aggrid import AgGrid, GridOptionsBuilder
import torch
import torch.nn as nn
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange
import math

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# --- Transformer Model ---
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1, max_len=500):
        super().__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_linear(output[:, -1, :])
        return output

# --- Feature Engineering ---
def add_indicators(df):
    df = df.copy()
    df['RSI'] = RSIIndicator(close=df['Close'].squeeze(), window=14).rsi()
    df['EMA20'] = EMAIndicator(close=df['Close'].squeeze(), window=20).ema_indicator()
    df['MACD'] = MACD(close=df['Close'].squeeze()).macd()
    df['ADX'] = ADXIndicator(
        high=df['High'].squeeze(),
        low=df['Low'].squeeze(),
        close=df['Close'].squeeze()
    ).adx()
    df['ATR'] = AverageTrueRange(
        high=df['High'].squeeze(),
        low=df['Low'].squeeze(),
        close=df['Close'].squeeze()
    ).average_true_range()
    df['Volume'] = df['Volume'] 
    df.dropna(inplace=True)
    return df

# --- Sequence Creation ---
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i + seq_len]
        y = data[i + seq_len, 0]  # Predict Close only
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).flatten()

# --- Main Function ---
def run_ai_prediction():
    st.title("📈 Transformer-based Stock Forecast with Technical Indicators")

    with st.expander("⚙️ Settings", expanded=True):
        col1, col2 = st.columns(2)
        user_stock = col1.text_input("Stock Symbol (e.g., INFY)", value="INFY")
        pred_days = col2.slider("Forecast Days", 5, 15, 7)

    if st.button("🚀 Predict with Transformer"):
        ticker = f"{user_stock.upper().strip()}.NS"
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if df.empty:
                st.error("No data found for this stock")
                return

            df = add_indicators(df)
            features = ['Close', 'RSI', 'EMA20', 'MACD', 'ADX', 'ATR','Volume']
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df[features])

            seq_len = 30
            X, y = create_sequences(scaled, seq_len)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

            model = TransformerModel(
                input_size=len(features),
                d_model=128,
                nhead=8,
                num_layers=4,
                dropout=0.2
            )

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # --- Training with Progress Bar ---
            model.train()
            progress_bar = st.progress(0)
            status_text = st.empty()

            for epoch in range(50):
                optimizer.zero_grad()
                output = model(X_tensor)
                loss = loss_fn(output.view(-1), y_tensor)
                loss.backward()
                optimizer.step()

                # Update progress
                percent_complete = int(((epoch + 1) / 50) * 100)
                progress_bar.progress(percent_complete)
                status_text.text(f"Training progress: {percent_complete}% (Epoch {epoch+1}/50)")

            status_text.text("✅ Training completed!")

            # --- Prediction ---
            model.eval()
            preds = []
            input_seq = X_tensor[-1].unsqueeze(0)
            last_known = df.copy()

            for i in range(pred_days):
                with torch.no_grad():
                    pred = model(input_seq).item()
                st.write(f"🔢 Forecast {i+1}: Raw prediction value: {pred}")
                st.write(f"📊 Input to scaler.inverse_transform (pred only): {[pred] + [0]*(len(features)-1)}")

                # Copy last row’s feature values and replace Close
                last_features = last_known[features].iloc[-1].copy()
                last_features['Close'] = pred  # Replace only Close with new prediction
                
                # Inverse transform using realistic indicator values (not zeroes)
                predicted_row_scaled = last_features.values
                pred_close = scaler.inverse_transform([predicted_row_scaled])[0][0]
                
                # Prepare new row with predicted Close
                new_row = pd.Series(index=last_known.columns, dtype='float64')
                new_row['Close'] = pred_close

                next_date = last_known.index[-1] + timedelta(days=1)
                last_known.loc[next_date] = new_row
                last_known = add_indicators(last_known)
                st.write(f"📅 Added row for date {next_date.date()} with Close = {pred_close}")
                last_scaled = scaler.transform(last_known[features].iloc[-seq_len:])
                input_seq = torch.tensor(last_scaled[np.newaxis], dtype=torch.float32)
                st.write(f"📈 Input to model for next step — shape: {input_seq.shape}")
                st.write(f"📈 Last sequence to model: {input_seq.numpy().squeeze()[-1]}")
                preds.append(pred)

            # Forecast dates from *next day*
            forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=pred_days)

            forecast_close = scaler.inverse_transform(
                np.hstack([np.array(preds).reshape(-1, 1), np.zeros((pred_days, len(features)-1))]))[:, 0]
            forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Close": forecast_close})

            # --- Model Signal ---
            current_price = float(df['Close'].iloc[-1])
            predicted_price = float(forecast_df['Predicted Close'].iloc[0])
            pct_diff = ((predicted_price - current_price) / current_price) * 100

            if pct_diff >= 2:
                signal = "BUY"
                reason = f"📈 Forecasted to rise by {pct_diff:.2f}%"
            elif pct_diff <= -2:
                signal = "SELL"
                reason = f"📉 Forecasted to fall by {pct_diff:.2f}%"
            else:
                signal = "HOLD"
                reason = f"🔄 Minimal change expected ({pct_diff:.2f}%)"

            st.markdown("### 🧠 Model Signal")
            if signal == "BUY":
                st.success(f"✅ SIGNAL: **{signal}**  \n**Reason:** {reason}")
            elif signal == "SELL":
                st.error(f"❌ SIGNAL: **{signal}**  \n**Reason:** {reason}")
            else:
                st.warning(f"🔄 SIGNAL: **{signal}**  \n**Reason:** {reason}")

            # --- Metrics ---
            col1, col2 = st.columns(2)
            col1.metric("Current Price", f"₹{current_price:.2f}")
            col2.metric("Predicted Price", f"₹{predicted_price:.2f}", f"{pct_diff:.2f}%")

            # --- Forecast Table ---
            gb = GridOptionsBuilder.from_dataframe(forecast_df)
            gb.configure_default_column(resizable=True, wrapText=True)
            grid_options = gb.build()
            AgGrid(forecast_df, gridOptions=grid_options, theme="balham", height=350)

            # --- Plot Chart ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], mode='lines', name='Historical'))
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Close'], mode='lines+markers', name='Predicted'))
            fig.update_layout(title=f"{user_stock.upper()} Forecast", xaxis_title="Date", yaxis_title="Close Price")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

if __name__ == "__main__":
    run_ai_prediction()
