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

# --- TRANSFORMER MODEL ---
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_linear(src)
        src = self.transformer_encoder(src)
        output = self.output_linear(src[:, -1, :])
        return output


# --- PREPARE SEQUENCES ---
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i + seq_len]
        y = data[i + seq_len, 0]  # Only Close
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# --- MAIN TRANSFORMER PREDICTION FUNCTION ---
def run_ai_prediction():
    st.title("📈 Transformer-based Stock Forecast")

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

            df.dropna(inplace=True)
            features = ['Close']
            data = df[features].copy()
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data)

            seq_len = 30
            X, y = create_sequences(scaled, seq_len)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

            model = TransformerModel(input_size=len(features))
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for epoch in range(30):
                optimizer.zero_grad()
                out = model(X_tensor)
                loss = loss_fn(out.view(-1), y_tensor)
                loss.backward()
                optimizer.step()

            model.eval()
            preds = []
            input_seq = X_tensor[-1].unsqueeze(0)
            last_known = df.copy()

            for _ in range(pred_days):
                with torch.no_grad():
                    pred = model(input_seq).item()
                new_row = np.array([pred])
                last_known.loc[last_known.index[-1] + timedelta(days=1)] = [np.nan] * len(df.columns)
                last_known.at[last_known.index[-1], 'Close'] = scaler.inverse_transform([[pred]])[0, 0]

                new_scaled = scaler.transform(last_known[features].values[-seq_len:])
                input_seq = torch.tensor(new_scaled[np.newaxis], dtype=torch.float32)
                preds.append(pred)

            forecast_dates = pd.date_range(start=last_known.index[-pred_days], periods=pred_days)
            forecast_close = scaler.inverse_transform(
                np.hstack([np.array(preds).reshape(-1, 1)]))[:, 0]
            forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Close": forecast_close})

            st.success("🎯 Forecast Complete with Transformer")

            # --- Trading Signal ---
            current_price = df['Close'].iloc[-1]
            predicted_price = forecast_df['Predicted Close'].iloc[0]
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

            # --- Display Forecast Table ---
            col1, col2 = st.columns(2)
            col1.metric("Current Price", f"₹{current_price:.2f}")
            col2.metric("Predicted Price", f"₹{predicted_price:.2f}", f"{pct_diff:.2f}%")

            gb = GridOptionsBuilder.from_dataframe(forecast_df)
            gb.configure_default_column(resizable=True, wrapText=True)
            grid_options = gb.build()

            AgGrid(forecast_df, gridOptions=grid_options, theme="balham", height=350)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], mode='lines', name='Historical'))
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Close'], mode='lines+markers', name='Predicted'))
            fig.update_layout(title=f"{user_stock.upper()} Forecast", xaxis_title="Date", yaxis_title="Close Price")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    run_ai_prediction()
