import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Cấu hình trang Streamlit
st.set_page_config(page_title="Dự Báo Giá Cổ Phiếu LSTM", page_icon="📈", layout="wide")

st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu bằng Deep Learning (LSTM)")
st.markdown("Dự báo giá đóng cửa cổ phiếu theo các khoảng thời gian tùy chỉnh (Ngày / Tháng / Năm).")

# ---------------------------------------------------------
# CẤU HÌNH BÊN THANH ĐIỀU HƯỚNG (SIDEBAR)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình đầu vào")

ticker = st.sidebar.text_input("Mã cổ phiếu (VD: VNM.VN, VIC.VN, AAPL):", value="VNM.VN").strip().upper()

col_start, col_end = st.sidebar.columns(2)
start_date = col_start.date_input("Từ ngày:", value=datetime.date(2018, 1, 1))
end_date = col_end.date_input("Đến ngày:", value=datetime.date.today())

st.sidebar.subheader("🎯 Khoảng thời gian dự báo")
predict_amount = st.sidebar.number_input("Số lượng:", min_value=1, value=10, step=1)
predict_unit = st.sidebar.selectbox("Đơn vị:", options=["Ngày", "Tháng", "Năm"])

# Quy đổi thời gian sang số ngày giao dịch (1 tháng ~ 22 ngày giao dịch, 1 năm ~ 252 ngày giao dịch)
if predict_unit == "Ngày":
    future_days = int(predict_amount)
elif predict_unit == "Tháng":
    future_days = int(predict_amount * 22)
else:  # Năm
    future_days = int(predict_amount * 252)

st.sidebar.info(f"👉 Tổng số ngày giao dịch sẽ dự báo: **{future_days} ngày**")

epochs = st.sidebar.slider("Số lượt huấn luyện (Epochs):", min_value=5, max_value=50, value=10, step=5)
batch_size = st.sidebar.selectbox("Batch Size:", options=[16, 32, 64], index=1)

btn_predict = st.sidebar.button("🚀 Bắt đầu Dự báo", use_container_width=True)

# ---------------------------------------------------------
# HÀM TẠO CHUỖI DỮ LIỆU (SEQUENCE CREATION)
# ---------------------------------------------------------
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# ---------------------------------------------------------
# XỬ LÝ CHÍNH KHÍ BẤM NÚT DỰ BÁO
# ---------------------------------------------------------
if btn_predict:
    # 1. FIX SEED (Ổn định kết quả)
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    with st.spinner(f"Đang tải dữ liệu cổ phiếu {ticker}..."):
        df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error(f"Không tìm thấy dữ liệu cho mã cổ phiếu **{ticker}** trong khoảng thời gian đã chọn!")
    else:
        st.success(f"Tải thành công **{len(df)}** phiên giao dịch của **{ticker}**!")

        # Chuẩn bị dữ liệu Close
        if isinstance(df.columns, pd.MultiIndex):
            data = df['Close'][[ticker]].values if ticker in df['Close'].columns else df['Close'].values
        else:
            data = df[['Close']].values

        # 2. CHIA TRAIN / TEST (80% / 20%)
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]

        # Chuẩn hóa MinMax
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Tạo chuỗi Lookback window = 60
        window_size = 60
        X_train, y_train = create_sequences(train_scaled, window_size)

        test_combined = np.vstack((train_scaled[-window_size:], test_scaled))
        X_test, y_test = create_sequences(test_combined, window_size)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Tách Validation (10% từ tập Train)
        val_size = int(0.1 * len(X_train))
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]

        # 3. XÂY DỰNG MÔ HÌNH LSTM
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # 4. HUẤN LUYỆN MÔ HÌNH
        with st.spinner("Đang huấn luyện mô hình LSTM... Vui lòng chờ trong giây lát!"):
            history = model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

        # 5. DỰ BÁO TRÊN TẬP TEST VÀ ĐÁNH GIÁ
        test_pred = model.predict(X_test)
        test_pred = scaler.inverse_transform(test_pred)
        y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
        mae = mean_absolute_error(y_test_real, test_pred)
        r2 = r2_score(y_test_real, test_pred)

        # Hiển thị chỉ số đánh giá
        st.subheader("📊 Đánh Giá Mô Hình Trên Tập Kiểm Thử (Test Set)")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("RMSE (Root Mean Sq Error)", f"{rmse:,.2f}")
        col_m2.metric("MAE (Mean Absolute Error)", f"{mae:,.2f}")
        col_m3.metric("R² Score", f"{r2:.4f}")

        # 6. DỰ BÁO TƯƠNG LAI
        last_60_days = np.vstack((train_scaled, test_scaled))[-window_size:]
        last_60_days = last_60_days.reshape(1, window_size, 1)

        future_predictions = []
        with st.spinner(f"Đang tiến hành dự báo {future_days} ngày tiếp theo..."):
            for _ in range(future_days):
                next_pred = model.predict(last_60_days, verbose=0)
                future_predictions.append(next_pred[0, 0])
                last_60_days = np.append(
                    last_60_days[:, 1:, :],
                    next_pred.reshape(1, 1, 1),
                    axis=1
                )

        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = scaler.inverse_transform(future_predictions)

        # ---------------------------------------------------------
        # HIỂN THỊ BIỂU ĐỒ (DỰA TRÊN FILE MẪU)
        # ---------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["📉 Biểu đồ Dự báo Tổng hợp", "📉 Biểu đồ Loss", "📋 Bảng Giá Dự Báo"])

        with tab1:
            st.subheader(f"Biểu đồ dự báo giá {ticker} ({predict_amount} {predict_unit.lower()} tương đương {future_days} ngày tới)")
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(y_test_real, label="Giá thật", color="blue")
            ax1.plot(test_pred, label="Dự báo test", color="orange")

            future_x = range(len(y_test_real), len(y_test_real) + future_days)
            ax1.plot(future_x, future_predictions, marker='o', color='green', markersize=3, label=f"Dự báo {predict_amount} {predict_unit.lower()} tới ({future_days} ngày)")

            ax1.set_title(f"Dự Báo Giá Cổ Phiếu {ticker}", fontsize=14)
            ax1.set_xlabel("Số phiên giao dịch")
            ax1.set_ylabel("Giá cổ phiếu")
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig1)

        with tab2:
            st.subheader("Lịch sử Loss trong quá trình huấn luyện")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(history.history['loss'], label='Train Loss', color='blue')
            ax2.plot(history.history['val_loss'], label='Validation Loss', color='red')
            ax2.set_title("Loss Train vs Validation", fontsize=14)
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Loss")
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig2)

        with tab3:
            st.subheader(f"Chi tiết giá dự báo {future_days} ngày giao dịch tiếp theo")
            df_future = pd.DataFrame({
                "Ngày dự báo (Thứ tự)": [f"Ngày +{i+1}" for i in range(future_days)],
                "Giá dự báo": future_predictions.flatten()
            })
            st.dataframe(df_future, use_container_width=True)

else:
    st.info("👈 Hãy chọn thông số bên thanh công cụ và nhấn **Bắt đầu Dự báo** để khởi chạy chương trình.")
