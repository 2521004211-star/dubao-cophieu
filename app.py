
import random
import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 1. CẤU HÌNH ỨNG DỤNG
# =========================================================
st.set_page_config(
    page_title="Stock Forecast AI | LSTM",
    page_icon="📈",
    layout="wide",
)

st.title("📈 DỰ BÁO GIÁ CỔ PHIẾU BẰNG LSTM")
st.caption(
    "Phân tích dữ liệu lịch sử và dự báo giá đóng cửa bằng Deep Learning."
)

# =========================================================
# 2. HÀM XỬ LÝ DỮ LIỆU
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(ticker, start_date, end_date):
    """Tải dữ liệu Yahoo Finance, cache tối đa 1 giờ."""
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Chuẩn hóa tên cột khi yfinance trả về MultiIndex.
    if isinstance(df.columns, pd.MultiIndex):
        close_columns = [
            col for col in df.columns
            if "Close" in [str(part) for part in col]
        ]

        if not close_columns:
            return pd.DataFrame()

        # Chọn đúng cột Close của ticker nếu có.
        exact = [
            col for col in close_columns
            if ticker in [str(part) for part in col]
        ]

        selected_column = exact[0] if exact else close_columns[0]
        close = df[selected_column]
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()

        close = df["Close"]

    # Luôn trả về một DataFrame có cột Close.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    result = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")})
    result.index = pd.to_datetime(df.index)
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index().dropna()
    result = result[np.isfinite(result["Close"])]
    result = result[result["Close"] > 0]

    return result


def create_sequences(data, window_size=60):
    """Tạo chuỗi đầu vào gồm window_size phiên trước."""
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )


def build_lstm_model(window_size):
    """Khởi tạo mô hình LSTM."""
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.1),
        LSTM(50),
        Dense(1),
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
    )

    return model


def forecast_future(model, last_window, scaler, steps):
    """Dự báo đệ quy từng bước trên thang đo đã chuẩn hóa."""
    current_window = last_window.copy()
    predictions_scaled = []

    for _ in range(steps):
        next_value = model.predict(current_window, verbose=0)
        predictions_scaled.append(float(next_value[0, 0]))

        current_window = np.concatenate(
            [
                current_window[:, 1:, :],
                next_value.reshape(1, 1, 1),
            ],
            axis=1,
        )

    predictions_scaled = np.asarray(
        predictions_scaled, dtype=np.float32
    ).reshape(-1, 1)

    return scaler.inverse_transform(predictions_scaled).flatten()


# =========================================================
# 3. SIDEBAR - CẤU HÌNH ĐẦU VÀO
# =========================================================
st.sidebar.header("⚙️ Cấu hình dự báo")

ticker = st.sidebar.text_input(
    "Mã cổ phiếu",
    value="VNM.VN",
    help="Ví dụ: VNM.VN, VIC.VN, AAPL, MSFT.",
).strip().upper()

today = datetime.date.today()

col_start, col_end = st.sidebar.columns(2)

start_date = col_start.date_input(
    "Từ ngày",
    value=datetime.date(2018, 1, 1),
    max_value=today,
)

end_date = col_end.date_input(
    "Đến ngày",
    value=today,
    max_value=today,
)

st.sidebar.subheader("🎯 Thời gian dự báo")

predict_amount = st.sidebar.number_input(
    "Số lượng",
    min_value=1,
    max_value=10,
    value=1,
    step=1,
)

predict_unit = st.sidebar.selectbox(
    "Đơn vị",
    ["Ngày", "Tháng", "Năm"],
)

# Giới hạn số bước để tránh huấn luyện và dự báo quá lâu.
if predict_unit == "Ngày":
    future_days = int(predict_amount)
elif predict_unit == "Tháng":
    future_days = int(predict_amount * 22)
else:
    future_days = int(predict_amount * 252)

st.sidebar.info(
    f"Số bước dự báo: **{future_days:,} phiên**"
)

epochs = st.sidebar.slider(
    "Số epochs tối đa",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
)

batch_size = st.sidebar.selectbox(
    "Batch size",
    [16, 32, 64],
    index=1,
)

window_size = st.sidebar.selectbox(
    "Số phiên lịch sử đầu vào",
    [30, 60, 90],
    index=1,
)

run_forecast = st.sidebar.button(
    "🚀 Bắt đầu dự báo",
    type="primary",
    use_container_width=True,
)

st.sidebar.caption(
    "Lưu ý: kết quả mô hình chỉ mang tính tham khảo, "
    "không phải khuyến nghị đầu tư."
)

# =========================================================
# 4. THỰC HIỆN DỰ BÁO
# =========================================================
if run_forecast:

    # Kiểm tra dữ liệu đầu vào.
    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    if start_date >= end_date:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        st.stop()

    # Tính kết quả có thể tái lập tương đối.
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        with st.spinner(f"Đang tải dữ liệu {ticker}..."):
            df = download_stock_data(
                ticker,
                start_date.isoformat(),
                end_date.isoformat(),
            )

        if df.empty:
            st.error(
                f"Không tìm thấy dữ liệu giá đóng cửa của {ticker}. "
                "Hãy kiểm tra mã cổ phiếu và khoảng thời gian."
            )
            st.stop()

        data = df["Close"].to_numpy(dtype=np.float32).reshape(-1, 1)

        # Cần đủ dữ liệu cho train, validation và test.
        if len(data) < 100:
            st.error(
                f"Chỉ có {len(data)} phiên hợp lệ. "
                "Hãy chọn khoảng thời gian dài hơn "
                "để có ít nhất 100 phiên dữ liệu."
            )
            st.stop()

        train_size = int(len(data) * 0.8)

        if train_size <= window_size + 2:
            st.error(
                "Dữ liệu huấn luyện không đủ cho cửa sổ đầu vào. "
                "Hãy tăng khoảng thời gian hoặc giảm số phiên lịch sử."
            )
            st.stop()

        if len(data) - train_size < 2:
            st.error("Tập kiểm thử quá nhỏ.")
            st.stop()

        # -------------------------------------------------
        # 4.1 CHIA DỮ LIỆU THEO THỜI GIAN
        # -------------------------------------------------
        train_data = data[:train_size]
        test_data = data[train_size:]

        # Chỉ fit scaler trên tập train để tránh rò rỉ dữ liệu.
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        X_train, y_train = create_sequences(
            train_scaled, window_size
        )

        if len(X_train) < 10:
            st.error(
                "Không đủ chuỗi dữ liệu để huấn luyện. "
                "Hãy chọn cửa sổ ngắn hơn hoặc khoảng thời gian dài hơn."
            )
            st.stop()

        # Validation là phần cuối của chuỗi train.
        val_size = max(1, int(len(X_train) * 0.1))

        if len(X_train) - val_size < 2:
            st.error("Tập train sau khi tách validation quá nhỏ.")
            st.stop()

        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]

        # Tạo test sequences với window lịch sử ngay trước test.
        test_combined = np.vstack([
            train_scaled[-window_size:],
            test_scaled,
        ])

        X_test, y_test = create_sequences(
            test_combined, window_size
        )

        X_train_final = X_train_final.reshape(
            -1, window_size, 1
        )
        X_val = X_val.reshape(-1, window_size, 1)
        X_test = X_test.reshape(-1, window_size, 1)

        # -------------------------------------------------
        # 4.2 HUẤN LUYỆN MÔ HÌNH
        # -------------------------------------------------
        tf.keras.backend.clear_session()

        model = build_lstm_model(window_size)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )

        progress = st.progress(0, text="Đang chuẩn bị huấn luyện...")

        class StreamlitProgress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress.progress(
                    min((epoch + 1) / epochs, 1.0),
                    text=f"Đã huấn luyện epoch {epoch + 1}/{epochs}",
                )

        with st.spinner("Mô hình LSTM đang học dữ liệu..."):
            history = model.fit(
                X_train_final,
                y_train_final,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                callbacks=[
                    early_stop,
                    StreamlitProgress(),
                ],
                verbose=0,
            )

        progress.empty()

        # -------------------------------------------------
        # 4.3 ĐÁNH GIÁ TRÊN TẬP TEST
        # -------------------------------------------------
        test_pred_scaled = model.predict(X_test, verbose=0)

        test_pred = scaler.inverse_transform(
            test_pred_scaled
        ).flatten()

        y_test_real = scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        rmse = float(np.sqrt(
            mean_squared_error(y_test_real, test_pred)
        ))
        mae = float(
            mean_absolute_error(y_test_real, test_pred)
        )
        r2 = (
            float(r2_score(y_test_real, test_pred))
            if len(y_test_real) >= 2
            else float("nan")
        )

        # -------------------------------------------------
        # 4.4 DỰ BÁO TƯƠNG LAI
        # -------------------------------------------------
        # Dùng toàn bộ lịch sử có sẵn để tạo cửa sổ cuối.
        all_scaled = scaler.transform(data)
        last_window = all_scaled[-window_size:].reshape(
            1, window_size, 1
        )

        with st.spinner(
            f"Đang dự báo {future_days:,} bước tiếp theo..."
        ):
            future_predictions = forecast_future(
                model,
                last_window,
                scaler,
                future_days,
            )

        # Ngày làm việc ước tính; không phản ánh chính xác
        # lịch nghỉ riêng của từng sàn chứng khoán.
        future_dates = pd.bdate_range(
            start=df.index[-1] + pd.offsets.BDay(1),
            periods=future_days,
        )

        # -------------------------------------------------
        # 5. GIAO DIỆN KẾT QUẢ
        # -------------------------------------------------
        st.success(
            f"Đã xử lý {len(df):,} phiên dữ liệu của {ticker}."
        )

        st.subheader("📌 Tổng quan dữ liệu")

        latest_price = float(data[-1, 0])
        previous_price = (
            float(data[-2, 0]) if len(data) >= 2 else latest_price
        )
        change_pct = (
            (latest_price / previous_price - 1) * 100
            if previous_price != 0
            else 0.0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Giá đóng cửa gần nhất",
            f"{latest_price:,.2f}",
            f"{change_pct:+.2f}%",
        )
        c2.metric("Số phiên dữ liệu", f"{len(df):,}")
        c3.metric("RMSE tập test", f"{rmse:,.3f}")
        c4.metric("MAE tập test", f"{mae:,.3f}")

        st.metric("R² tập test", f"{r2:.4f}")

        st.caption(
            "Các chỉ số RMSE và MAE được tính theo đơn vị giá của mã cổ phiếu. "
            "R² có thể âm nếu mô hình dự báo kém hơn đường chuẩn trung bình."
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Giá thực tế & dự báo",
            "🧠 Quá trình huấn luyện",
            "📋 Bảng giá dự báo",
            "📊 Dữ liệu lịch sử",
        ])

        with tab1:
            st.subheader(f"Giá đóng cửa và dự báo: {ticker}")

            fig, ax = plt.subplots(figsize=(13, 6))

            test_dates = df.index[train_size:]

            ax.plot(
                test_dates,
                y_test_real,
                label="Giá thực tế (test)",
                linewidth=1.8,
            )
            ax.plot(
                test_dates,
                test_pred,
                label="Dự báo trên test",
                linewidth=1.6,
            )
            ax.plot(
                future_dates,
                future_predictions,
                label="Dự báo tương lai",
                linewidth=2,
            )

            ax.axvline(
                df.index[-1],
                linestyle="--",
                alpha=0.7,
                label="Ngày dữ liệu cuối",
            )

            ax.set_xlabel("Ngày")
            ax.set_ylabel("Giá đóng cửa")
            ax.set_title(f"Dự báo giá cổ phiếu {ticker}")
            ax.legend()
            ax.grid(True, alpha=0.25)
            fig.autofmt_xdate()
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.info(
                "Ngày tương lai trên biểu đồ là ngày làm việc ước tính. "
                "Mô hình dự báo theo từng bước, vì vậy sai số có thể "
                "tích lũy khi dự báo xa."
            )

        with tab2:
            st.subheader("Loss trong quá trình huấn luyện")

            fig_loss, ax_loss = plt.subplots(figsize=(10, 5))

            ax_loss.plot(
                history.history["loss"],
                label="Train loss",
            )
            ax_loss.plot(
                history.history["val_loss"],
                label="Validation loss",
            )

            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Mean squared error")
            ax_loss.set_title("Train loss vs Validation loss")
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.25)
            fig_loss.tight_layout()

            st.pyplot(fig_loss)
            plt.close(fig_loss)

            st.write(
                f"Mô hình dừng sau **{len(history.history['loss'])} epochs** "
                "hoặc khi đạt điều kiện Early Stopping."
            )

        with tab3:
            st.subheader("Bảng giá dự báo tương lai")

            df_future = pd.DataFrame({
                "Ngày ước tính": future_dates,
                "Phiên dự báo": np.arange(1, future_days + 1),
                "Giá dự báo": future_predictions,
            })

            st.dataframe(
                df_future,
                use_container_width=True,
                hide_index=True,
            )

            csv_data = df_future.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="⬇️ Tải bảng dự báo CSV",
                data=csv_data,
                file_name=f"{ticker}_forecast.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with tab4:
            st.subheader("Dữ liệu giá đóng cửa lịch sử")

            st.line_chart(df["Close"])

            st.dataframe(
                df.tail(20).sort_index(ascending=False),
                use_container_width=True,
            )

    except Exception as exc:
        st.error(
            "Ứng dụng gặp lỗi trong quá trình tải dữ liệu, "
            "huấn luyện hoặc dự báo."
        )
        st.exception(exc)

else:
    st.info(
        "👈 Chọn mã cổ phiếu, khoảng thời gian và thông số ở thanh bên, "
        "sau đó nhấn **Bắt đầu dự báo**."
    )

    st.markdown(
        """
        ### Hướng dẫn nhanh
        1. Nhập mã cổ phiếu, ví dụ `VNM.VN` hoặc `AAPL`.
        2. Chọn khoảng thời gian có dữ liệu lịch sử.
        3. Chọn thời gian dự báo và số phiên lịch sử đầu vào.
        4. Nhấn **Bắt đầu dự báo** để xem kết quả.
        """
    )
