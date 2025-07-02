import streamlit as st
from data_utils import get_stock_data, prepare_data
from model import build_transformer_model
from visualize import plot_attention_heatmap
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

st.title("📈 Transformer-Based Stock Predictor")

symbol = st.text_input("Enter Stock Symbol (e.g., INFY)", value="INFY")
predict_days = st.slider("Forecast days", 5, 15, 7)

if st.button("Predict"):
    with st.spinner("Fetching and processing data..."):
        df = get_stock_data(symbol)
        features = ['Close']  # You can expand to RSI, EMA, etc.

        scaled_data, scaler = prepare_data(df[features])
        X, y = [], []
        window = 30
        for i in range(window, len(scaled_data)):
            X.append(scaled_data[i-window:i])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)

    model = build_transformer_model(input_shape=X.shape[1:])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    st.success("Model trained. Predicting future...")
    predictions, attention_scores = [], []
    last_sequence = X[-1]

    for _ in range(predict_days):
        input_seq = last_sequence[np.newaxis, ...]
        pred = model(input_seq, training=False).numpy()[0, 0]
        predictions.append(pred)

        att = model.get_layer("attention").output.numpy()
        attention_scores.append(att[0])

        # Shift window and append
        last_sequence = np.append(last_sequence[1:], [[pred]], axis=0)

    predicted_prices = scaler.inverse_transform(
        np.concatenate([np.array(predictions).reshape(-1, 1)] + [np.zeros_like(np.array(predictions).reshape(-1, 1))]* (scaled_data.shape[1] - 1), axis=1)
    )[:, 0]

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=predict_days)
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted Close": predicted_prices})

    st.line_chart(pred_df.set_index("Date"))
    plot_attention_heatmap(attention_scores, window)
