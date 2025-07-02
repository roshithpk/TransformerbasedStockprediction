# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange
import ai_prediction_transformer

if "page" not in st.session_state:
    st.session_state.page = "main"

st.set_page_config(page_title="📊 Indian Swing Trade Scanner", layout="wide")
st.title("📈 Indian Swing Trade Scanner (5-10 Days)")

@st.cache_data
def load_stocks():
    df = pd.read_csv("stocks.csv")
    return df

stock_df = load_stocks()

st.sidebar.header("🔧 Filters")
min_volume = st.sidebar.slider("Min Volume (x 5-Day Avg)", 1.0, 5.0, 1.5)
rsi_low = st.sidebar.slider("Min RSI", 10, 50, 30)
rsi_high = st.sidebar.slider("Max RSI", 50, 90, 75)
min_price = st.sidebar.slider("Min Price (₹)", 10, 1000, 100)
max_price = st.sidebar.slider("Max Price (₹)", 1000, 10000, 3000)
breakout_required = st.sidebar.checkbox("📈 Current Price > Last 2 Days' Closes", value=True)
trend_required = st.sidebar.checkbox("🟢 Price Above 20 EMA", value=True)

st.subheader("📂 Select Stock Category to Scan")
categories = ["All"] + sorted(stock_df["Category"].dropna().unique())
selected_category = st.selectbox("Category", categories, index=0)

if selected_category == "All":
    filtered_df = stock_df
else:
    filtered_df = stock_df[stock_df["Category"] == selected_category]

filtered_tickers = filtered_df["Ticker"].dropna().unique().tolist()

def scan_stock(ticker):
    try:
        data = yf.download(ticker, period="1mo", progress=False, auto_adjust=False)
        if data.empty or len(data) < 20:
            return None

        data = data.dropna()
        close_prices = pd.Series(data["Close"].values.flatten(), dtype=float)
        volumes = pd.Series(data["Volume"].values.flatten(), dtype=float)

        if close_prices.empty or volumes.empty:
            return None

        ema_20 = EMAIndicator(close=close_prices, window=20).ema_indicator()
        rsi = RSIIndicator(close=close_prices, window=14).rsi()

        latest_close = close_prices.iloc[-1]
        latest_volume = volumes.iloc[-1]
        avg_volume_5d = volumes.rolling(window=5).mean().iloc[-1]
        latest_rsi = rsi.iloc[-1]

        breakout_ok = latest_close > close_prices.iloc[-2] and latest_close > close_prices.iloc[-3] if breakout_required else True
        volume_ok = latest_volume > avg_volume_5d * min_volume
        trend_ok = latest_close > ema_20.iloc[-1] if trend_required else True
        rsi_ok = rsi_low < latest_rsi < rsi_high
        price_ok = min_price <= latest_close <= max_price

        if all([volume_ok, trend_ok, rsi_ok, breakout_ok, price_ok]):
            return {
                "Stock": ticker.replace(".NS", ""),
                "Price (₹)": f"₹{latest_close:.2f}",
                "Volume (x)": f"{latest_volume / avg_volume_5d:.1f}",
                "RSI": f"{latest_rsi:.1f}",
                "Trend": "🟢" if latest_close > ema_20.iloc[-1] else "🔴",
                "Why Buy?": "🔥 2-Day Momentum + Volume Surge"
            }
    except Exception as e:
        st.error(f"Error scanning {ticker}: {str(e)}")
    return None

if st.button("🔍 Scan Selected Stocks"):
    with st.spinner("Scanning selected stocks..."):
        results = []
        for ticker in filtered_tickers:
            result = scan_stock(ticker)
            if result:
                results.append(result)
    if results:
        st.success(f"✅ Found {len(results)} potential swing trades.")
        st.dataframe(pd.DataFrame(results), hide_index=True)
    else:
        st.warning("⚠️ No stocks matched the criteria. Adjust your filters and try again.")

st.markdown("---")
st.subheader("🔎 Analyze a Specific Stock")
user_stock = st.text_input("Enter NSE Stock Symbol (e.g., INFY)")

if user_stock:
    full_ticker = user_stock.upper().strip() + ".NS"
    try:
        data = yf.download(full_ticker, period="1mo", progress=False)
        if not data.empty and len(data) > 14:
            data = data.dropna()
            close_prices = pd.Series(data["Close"].values.flatten(), index=data.index)
            volumes = pd.Series(data["Volume"].values.flatten(), index=data.index)

            ema_20 = EMAIndicator(close=close_prices, window=20).ema_indicator()
            rsi = RSIIndicator(close=close_prices, window=14).rsi()

            latest_close = close_prices.iloc[-1]
            latest_volume = volumes.iloc[-1]
            avg_volume_5d = volumes.rolling(window=5).mean().iloc[-1]
            latest_rsi = rsi.iloc[-1]
            trend = "🟢" if latest_close > ema_20.iloc[-1] else "🔴"

            remarks = []
            if latest_close <= close_prices.iloc[-2] or latest_close <= close_prices.iloc[-3]:
                remarks.append("Price not above last 2 days")
            if latest_volume < avg_volume_5d * min_volume:
                remarks.append("Low volume (vs 5-day avg)")
            if not (rsi_low < latest_rsi < rsi_high):
                remarks.append("RSI not in range")
            if latest_close < min_price or latest_close > max_price:
                remarks.append("Price not in range")

            st.markdown("#### 🔬 Result:")
            result = {
                "Stock": user_stock.upper(),
                "Price (₹)": f"₹{latest_close:.2f}",
                "Volume (x)": f"{latest_volume / avg_volume_5d:.1f}",
                "RSI": f"{latest_rsi:.1f}",
                "Trend": trend,
                "Remarks?": "✅ Good for Swing Trade" if not remarks else "❌ " + ", ".join(remarks)
            }
            st.dataframe(pd.DataFrame([result]), hide_index=True)
        else:
            st.error("❌ Not enough data for analysis.")
    except Exception as e:
        st.error(f"Error fetching data for {user_stock.upper()}: {str(e)}")

# --- AI SECTION ---
st.markdown("---")
st.subheader("🤖 AI Forecast")
ai_prediction_transformer.run_ai_prediction()

# --- FOOTER ---
st.markdown("---")
st.caption("⚡ Developed by Roshith •  Please feed your comments to roshith77@gmail.com")
