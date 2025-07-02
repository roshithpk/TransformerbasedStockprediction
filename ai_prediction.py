import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from st_aggrid import AgGrid, GridOptionsBuilder

# --- LSTM HELPER FUNCTION ---
def prepare_lstm_data(data, n_steps=30):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i])
        y.append(data[i, 0])  # Predicting close price only
    return np.array(X), np.array(y)

# --- TECHNICAL INDICATORS ---
def add_technical_indicators(df):
    try:
        # Check if we have enough data (minimum 30 days)
        if len(df) < 30:
            st.error(f"Need at least 30 data points, only have {len(df)}")
            return None
        
        # Ensure we have required price columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Convert to pandas Series with proper index
        close = pd.Series(df['Close'].values.flatten(), index=df.index)
        high = pd.Series(df['High'].values.flatten(), index=df.index)
        low = pd.Series(df['Low'].values.flatten(), index=df.index)
        volume = pd.Series(df['Volume'].values.flatten(), index=df.index)


        def sma_rsi(series, window=14):
            delta = series.diff()
        
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
        
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
        
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        # Calculate each indicator with individual error handling
        indicators = {
            'RSI': lambda: RSIIndicator(close=close, window=14).rsi(),
            'Stoch_%K': lambda: StochasticOscillator(high=high, low=low, close=close, window=14).stoch(),
            'Stoch_%D': lambda: StochasticOscillator(high=high, low=low, close=close, window=14).stoch_signal(),
            'EMA_20': lambda: EMAIndicator(close=close, window=20).ema_indicator(),
            'EMA_50': lambda: EMAIndicator(close=close, window=50).ema_indicator(),
            'MACD': lambda: MACD(close=close).macd(),
            'MACD_Signal': lambda: MACD(close=close).macd_signal(),
            'MACD_Hist': lambda: MACD(close=close).macd_diff(),
            'BB_Upper': lambda: BollingerBands(close=close, window=20, window_dev=2).bollinger_hband(),
            'BB_Middle': lambda: BollingerBands(close=close, window=20, window_dev=2).bollinger_mavg(),
            'BB_Lower': lambda: BollingerBands(close=close, window=20, window_dev=2).bollinger_lband(),
            'VWAP': lambda: VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume, window=14).volume_weighted_average_price()
        }

        for name, calc in indicators.items():
            try:
                df[name] = calc()
            except Exception as e:
                st.warning(f"Could not calculate {name}: {str(e)}")
                df[name] = np.nan

        # Fill any remaining NA values
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Verify we have the essential indicators
        required_indicators = ['RSI', 'EMA_20', 'MACD', 'BB_Upper', 'BB_Lower']
        if not all(ind in df.columns for ind in required_indicators):
            st.error("Failed to calculate essential indicators")
            return None
            
        return df
    
    except Exception as e:
        st.error(f"Technical indicators failed: {str(e)}")
        return None

# --- TRADING SIGNALS ---
# --- TRADING SIGNALS ---
def generate_signals(df, forecast, min_volume=1.2):
    try:
        last_row = df.iloc[-1]
        current_close = float(last_row['Close'].iloc[0]) if isinstance(last_row['Close'], pd.Series) else float(last_row['Close'])
        pred_close = float(forecast['Predicted Close'].iloc[0])
        reasons = []

        # --- 1. Core AI Forecast ---
        price_diff = (pred_close - current_close) / current_close
        if price_diff > 0.02:
            base_signal = "BUY"
            reasons.append(f"AI forecast suggests +{price_diff:.2%} increase")
        elif price_diff < -0.02:
            base_signal = "SELL"
            reasons.append(f"AI forecast suggests {price_diff:.2%} drop")
        else:
            base_signal = "HOLD"
            reasons.append("AI forecast suggests minor movement")

        # --- 2. Custom Breakout Logic (New) ---
        if len(df) >= 3:
            prev_close_1 = float(df['Close'].iloc[-2])
            prev_close_2 = float(df['Close'].iloc[-3])
            if current_close > prev_close_1 and current_close > prev_close_2:
                reasons.append("Price > last 2 days’ closes (momentum)")
            else:
                reasons.append("Price not above last 2 days")
                if base_signal == "BUY":
                    base_signal = "HOLD"

        # --- 3. Custom Volume Logic (New) ---
        if len(df) >= 5:
            last_volume = float(df['Volume'].iloc[-1])
            volume_avg_5 = float(df['Volume'].rolling(window=5).mean().iloc[-1])
            if last_volume > volume_avg_5 * min_volume:
                reasons.append("Volume > 5-day avg × multiplier")
            else:
                reasons.append("Volume not strong (vs 5-day avg)")
                if base_signal == "BUY":
                    base_signal = "HOLD"

        # --- 4. RSI, MACD, BB (same as before) ---
        confidence_votes = {"BUY": 0, "SELL": 0}
        
        if 'RSI' in df.columns:
            rsi = float(last_row['RSI'].iloc[0]) if isinstance(last_row['RSI'], pd.Series) else float(last_row['RSI'])
            if rsi < 30:
                confidence_votes["BUY"] += 1
                reasons.append(f"RSI {rsi:.1f} (oversold)")
            elif rsi > 70:
                confidence_votes["SELL"] += 1
                reasons.append(f"RSI {rsi:.1f} (overbought)")

        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            macd = float(last_row['MACD'].iloc[0]) if isinstance(last_row['MACD'], pd.Series) else float(last_row['MACD'])
            macd_signal = float(last_row['MACD_Signal'].iloc[0]) if isinstance(last_row['MACD_Signal'], pd.Series) else float(last_row['MACD_Signal'])
            if macd > macd_signal:
                confidence_votes["BUY"] += 1
                reasons.append("MACD crossover bullish")
            else:
                confidence_votes["SELL"] += 1
                reasons.append("MACD crossover bearish")

        if 'BB_Lower' in df.columns and 'BB_Upper' in df.columns:
            bb_lower = float(last_row['BB_Lower'].iloc[0]) if isinstance(last_row['BB_Lower'], pd.Series) else float(last_row['BB_Lower'])
            bb_upper = float(last_row['BB_Upper'].iloc[0]) if isinstance(last_row['BB_Upper'], pd.Series) else float(last_row['BB_Upper'])
            if current_close < bb_lower:
                confidence_votes["BUY"] += 1
                reasons.append("Price below lower Bollinger Band")
            elif current_close > bb_upper:
                confidence_votes["SELL"] += 1
                reasons.append("Price above upper Bollinger Band")

        # --- 5. Final Signal Decision ---
        if base_signal == "BUY" and confidence_votes["SELL"] >= 2:
            final_signal = "HOLD"
            reasons.append("Conflicting indicators reduced BUY to HOLD")
        elif base_signal == "SELL" and confidence_votes["BUY"] >= 2:
            final_signal = "HOLD"
            reasons.append("Conflicting indicators reduced SELL to HOLD")
        else:
            final_signal = base_signal

        return final_signal, list(set(reasons))

    except Exception as e:
        st.error(f"Signal generation failed: {str(e)}")
        return "ERROR", ["Could not generate signals"]




# --- MAIN APP ---
def run_ai_prediction():
    st.title("📈 AI Stock Prediction Dashboard")
    
    with st.expander("⚙️ Settings", expanded=True):
        col1, col2 = st.columns(2)
        user_stock = col1.text_input("Stock Symbol (e.g., INFY)", value="INFY")
        pred_days = col2.slider("Forecast Days", 5, 15, 7)
    
    if st.button("🚀 Generate Forecast"):
        ticker = f"{user_stock.upper().strip()}.NS"
        
        with st.spinner("Processing..."):
            try:
                # 1. Data Collection
                df = yf.download(ticker, period="2y", interval="1d", progress=False)
                if df.empty:
                    st.error("No data found for this stock")
                    return
                
                # 2. Add Technical Indicators
                df = add_technical_indicators(df)
                if df is None:
                    st.error("Failed to calculate technical indicators")
                    return
                
                # 3. Prepare LSTM Data
                features = ['Close', 'RSI', 'EMA_20', 'MACD', 'BB_Upper', 'BB_Lower']
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df[features])
                
                X, y = prepare_lstm_data(scaled_data)
                X = X.reshape((X.shape[0], X.shape[1], len(features)))
                
                # 4. Build and Train Model
                model = Sequential([
                    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                    Dropout(0.2),
                    LSTM(64),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X, y, epochs=50, batch_size=32, verbose=0)
                
                # 5. Predict Future Day-by-Day with Technical Indicator Recalc
                df_forecast = df.copy()
                future_preds = []
                
                for _ in range(pred_days):
                    # Scale last 30 rows
                    input_data = scaler.transform(df_forecast[features].iloc[-30:])
                    input_data = input_data.reshape(1, 30, len(features))

                    # Predict next close
                    next_scaled = model.predict(input_data, verbose=0)[0, 0]

                    # Inverse transform only the predicted close
                    dummy_row = np.zeros((1, len(features)))
                    dummy_row[0, 0] = next_scaled
                    next_close = scaler.inverse_transform(dummy_row)[0, 0]
                    future_preds.append(next_close)

                    # Append new row with predicted close
                    next_date = df_forecast.index[-1] + timedelta(days=1)
                    new_row = pd.DataFrame([[np.nan]*len(df_forecast.columns)], columns=df_forecast.columns, index=[next_date])
                    new_row.at[next_date, 'Close'] = next_close
                    df_forecast = pd.concat([df_forecast, new_row])

                    # Recalculate technical indicators after adding prediction
                    df_forecast = add_technical_indicators(df_forecast)
                    if df_forecast is None:
                        st.error("Failed to recalculate indicators during forecast loop")
                        return

                # 6. Build Forecast DataFrame
                forecast_df = pd.DataFrame({
                    "Date": df_forecast.index[-pred_days:],
                    "Predicted Close": future_preds
                })

                # 7. Generate Signals
                signal, reasons = generate_signals(df, forecast_df)
                
                # 8. Display Results
                st.success("🎯 Forecast Complete!")

                # Trading Signal
                if signal == "BUY":
                    st.success(f"✅ SIGNAL: {signal}")
                elif signal == "SELL":
                    st.error(f"❌ SIGNAL: {signal}")
                else:
                    st.warning(f"🔄 SIGNAL: {signal}")
                
                st.subheader("Reasons:")
                for reason in reasons:
                    st.write(f"- {reason}")
                
                # Key Metrics
                col1, col2, col3 = st.columns(3)
                current_price = float(df['Close'].iloc[-1])
                predicted_price = float(forecast_df['Predicted Close'].values[0])
                price_diff_pct = ((predicted_price / current_price) - 1) * 100
                
                col1.metric("Current Price", f"₹{current_price:.2f}")
                col2.metric("Predicted Price", f"₹{predicted_price:.2f}", f"{price_diff_pct:.2f}%")
                col3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                
                # Forecast Table Display
                forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.strftime("%d-%m-%Y")
                forecast_df["Predicted Close"] = forecast_df["Predicted Close"].round(2)

                gb = GridOptionsBuilder.from_dataframe(forecast_df)
                gb.configure_default_column(resizable=True, wrapText=False, autoHeight=True)
                gb.configure_column("Date", width=100, cellStyle={"textAlign": "center"})
                gb.configure_column("Predicted Close", width=120, type=["numericColumn"], cellStyle={"textAlign": "center"})
                grid_options = gb.build()

                table_height = min(len(forecast_df), 8) * 38 + 50

                st.subheader("Forecast Details:")
                st.markdown("<div style='max-width: 300px; margin: auto;'>", unsafe_allow_html=True)

                AgGrid(
                    forecast_df,
                    gridOptions=grid_options,
                    height=table_height,
                    theme="balham",
                    fit_columns_on_grid_load=False
                )

                st.markdown("""
                <style>
                .ag-theme-balham .ag-cell {
                    padding: 4px !important;
                    font-size: 13px;
                }
                </style>
                """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
                                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    run_ai_prediction()
