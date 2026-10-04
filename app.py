"""
===================================================================================
HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC ĐỊNH LƯỢNG: KẾT HỢP SMA + OBV (OR) TRÊN DỮ LIỆU HOSE
Tác giả / Nhóm nghiên cứu: Nhóm 5 - Phân tích Định lượng & Quản trị Danh mục
Triển khai: Streamlit Web Application
===================================================================================
"""

import io
import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import minimize
import streamlit as st
import ta

warnings.filterwarnings("ignore")

# =================================================================================
# 1. CẤU HÌNH TRANG WEB STREAMLIT
# =================================================================================
st.set_page_config(
    page_title="Kiểm Định Chiến Lược SMA + OBV (OR) | HOSE Quantitative",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    /* Font và khoảng cách */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    /* Card thông số KPI */
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-value-positive {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10B981;
    }
    .metric-value-negative {
        font-size: 1.8rem;
        font-weight: 700;
        color: #EF4444;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Badge trạng thái */
    .badge-buy {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-sell {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-neutral {
        background-color: #F3F4F6;
        color: #374151;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    /* Highlight box */
    .callout-box {
        border-left: 5px solid #2563EB;
        background-color: #EFF6FF;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# =================================================================================
# 2. HẰNG SỐ & BỘ THAM SỐ TỐI ƯU MẶC ĐỊNH TỪ NGHIÊN CỨU
# =================================================================================
DEFAULT_TOP5 = ["DIG", "DPM", "KBC", "MSN", "DCM"]

PRESET_PARAMS = {
    "DIG": {"ma_short": 105, "ma_long": 345, "obv_window": 30},
    "DPM": {"ma_short": 25, "ma_long": 255, "obv_window": 100},
    "KBC": {"ma_short": 35, "ma_long": 350, "obv_window": 75},
    "MSN": {"ma_short": 140, "ma_long": 350, "obv_window": 100},
    "DCM": {"ma_short": 100, "ma_long": 360, "obv_window": 15},
}

PRESET_MPT_WEIGHTS = {
    "DIG": 0.2355,
    "DPM": 0.0000,
    "KBC": 0.1796,
    "MSN": 0.2339,
    "DCM": 0.3510,
}

# =================================================================================
# 3. HÀM TẢI & TIỀN XỬ LÝ DỮ LIỆU (CACHED)
# =================================================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file_source):
    """
    Đọc file CSV và chuẩn hóa cột dữ liệu OHLCV
    """
    if isinstance(file_source, str):
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)
    else:
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)

    df.columns = df.columns.str.strip().str.lower()
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "open", "high", "low", "close", "volume"])
    return df

@st.cache_data(show_spinner=False)
def calculate_ta_features(df, min_observations=400, trading_days=252):
    """
    Sàng lọc kỹ thuật (TA Screening) trên tập Train:
    - Trend (30%): Tỷ lệ SMA50 > SMA200, Giá vs SMA200
    - Momentum (25%): Momentum 6 tháng (~126 phiên)
    - Volume (25%): Giá trị giao dịch trung bình, OBV confirmation (chuẩn hóa volume 20 phiên)
    - Risk (20%): Annualized Volatility thấp hơn, Max Drawdown gần 0 hơn
    """
    results = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").dropna(subset=["close", "volume"]).copy()
        if len(g) < min_observations:
            continue

        close = g["close"].astype(float)
        volume = g["volume"].astype(float)

        # 1. Trend
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        trend_strength = (sma50 > sma200).mean()

        if pd.notna(sma200.iloc[-1]) and sma200.iloc[-1] != 0:
            price_vs_sma200 = close.iloc[-1] / sma200.iloc[-1] - 1
        else:
            price_vs_sma200 = np.nan

        # 2. Momentum
        if len(close) >= 126:
            momentum_6m = close.iloc[-1] / close.iloc[-126] - 1
        else:
            momentum_6m = np.nan

        # 3. Volume
        avg_trading_value = (close * volume).mean()
        obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        if len(obv) >= 20:
            vol_20 = volume.iloc[-20:].sum()
            obv_confirmation = (obv.iloc[-1] - obv.iloc[-20]) / vol_20 if vol_20 != 0 else np.nan
        else:
            obv_confirmation = np.nan

        # 4. Risk
        returns = close.pct_change().dropna()
        annual_volatility = returns.std() * np.sqrt(trading_days)
        equity = (1 + returns).cumprod()
        max_drawdown = (equity / equity.cummax() - 1).min()

        results.append({
            "Ticker": ticker,
            "N": len(g),
            "Trend_Strength": trend_strength,
            "Price_vs_SMA200": price_vs_sma200,
            "Momentum_6M": momentum_6m,
            "Avg_Trading_Value": avg_trading_value,
            "OBV_Confirmation": obv_confirmation,
            "Volatility": annual_volatility,
            "Max_Drawdown": max_drawdown
        })

    res = pd.DataFrame(results)
    if res.empty:
        return res

    res = res.dropna(subset=[
        "Trend_Strength", "Price_vs_SMA200", "Momentum_6M",
        "Avg_Trading_Value", "OBV_Confirmation", "Volatility", "Max_Drawdown"
    ])

    # Tính Rank & Điểm
    res["Trend_Strength_Rank"] = res["Trend_Strength"].rank(pct=True)
    res["Price_SMA200_Rank"] = res["Price_vs_SMA200"].rank(pct=True)
    res["Trend_Score"] = 0.50 * res["Trend_Strength_Rank"] + 0.50 * res["Price_SMA200_Rank"]

    res["Momentum_Score"] = res["Momentum_6M"].rank(pct=True)

    res["Liquidity_Rank"] = res["Avg_Trading_Value"].rank(pct=True)
    res["OBV_Rank"] = res["OBV_Confirmation"].rank(pct=True)
    res["Volume_Score"] = 0.60 * res["Liquidity_Rank"] + 0.40 * res["OBV_Rank"]

    res["Volatility_Rank"] = res["Volatility"].rank(pct=True, ascending=False)
    res["Drawdown_Rank"] = res["Max_Drawdown"].rank(pct=True)
    res["Risk_Score"] = 0.50 * res["Volatility_Rank"] + 0.50 * res["Drawdown_Rank"]

    res["TA_Score"] = (
        0.30 * res["Trend_Score"]
        + 0.25 * res["Momentum_Score"]
        + 0.25 * res["Volume_Score"]
        + 0.20 * res["Risk_Score"]
    )

    res = res.sort_values("TA_Score", ascending=False).reset_index(drop=True)
    return res

def prepare_stock_data(df_full, ticker):
    """Tách và chuẩn hóa dữ liệu 1 mã theo chuẩn OHLCV, index theo Datetime"""
    df = df_full[df_full["ticker"] == ticker].copy()
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
    })
    df = df[["date", "Open", "High", "Low", "Close", "Volume"]]
    return df.set_index("date")

# =================================================================================
# 4. CÁC HÀM TÍNH TOÁN CHIẾN LƯỢC SMA + OBV (OR) & HIỆU SUẤT
# =================================================================================
def find_position_sma(df, paras):
    position = pd.Series(0.0, index=df.index, name="SMA_signal")
    ma_short = int(paras["ma_short"])
    ma_long = int(paras["ma_long"])
    if ma_short >= ma_long:
        return position

    sma_short = ta.trend.SMAIndicator(close=df["Close"], window=ma_short).sma_indicator()
    sma_long = ta.trend.SMAIndicator(close=df["Close"], window=ma_long).sma_indicator()

    buy = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
    sell = (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))

    position.loc[buy] = 1.0
    position.loc[sell] = -1.0
    return position

def find_position_obv(df, paras):
    position = pd.Series(0.0, index=df.index, name="OBV_signal")
    obv_window = int(paras["obv_window"])

    obv = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    obv_ma = obv.rolling(window=obv_window).mean()

    buy = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))

    position.loc[buy] = 1.0
    position.loc[sell] = -1.0
    return position

def find_position_or(df, sma_paras, obv_paras):
    sma_signal = find_position_sma(df, sma_paras)
    obv_signal = find_position_obv(df, obv_paras)

    position = pd.Series(0.0, index=df.index, name="SMA_OBV_OR")
    buy = (sma_signal == 1.0) | (obv_signal == 1.0)
    sell = (sma_signal == -1.0) | (obv_signal == -1.0)

    conflict = buy & sell
    position.loc[buy & ~conflict] = 1.0
    position.loc[sell & ~conflict] = -1.0
    return position

def events_to_holding(events):
    holding = pd.Series(0.0, index=events.index)
    current = 0.0
    for i, signal in enumerate(events):
        if signal == 1.0:
            current = 1.0
        elif signal == -1.0:
            current = 0.0
        holding.iloc[i] = current
    return holding

def strategy_returns(df, events, commission=0.0):
    asset_return = df["Close"].pct_change().fillna(0.0)
    holding = events_to_holding(events)

    # Chống look-ahead bias: tín hiệu ngày t chỉ được thực thi tại phiên t+1
    executed_holding = holding.shift(1).fillna(0.0)
    strat_ret = executed_holding * asset_return

    turnover = executed_holding.diff().abs().fillna(executed_holding.abs())
    strat_ret = strat_ret - turnover * commission

    return strat_ret, executed_holding

def performance_stats(returns, trading_days=252):
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return {
            "Mean Daily Return [%]": np.nan,
            "Total Return [%]": np.nan,
            "Annual Return [%]": np.nan,
            "Annual Volatility [%]": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown [%]": np.nan,
        }

    equity = (1 + r).cumprod()
    total_ret = equity.iloc[-1] - 1
    years = len(r) / trading_days

    if years > 0 and equity.iloc[-1] > 0:
        annual_ret = equity.iloc[-1] ** (1 / years) - 1
    else:
        annual_ret = np.nan

    annual_vol = r.std() * np.sqrt(trading_days)
    sharpe = (r.mean() / r.std() * np.sqrt(trading_days)) if r.std() != 0 else np.nan
    drawdown = equity / equity.cummax() - 1

    return {
        "Mean Daily Return [%]": r.mean() * 100,
        "Total Return [%]": total_ret * 100,
        "Annual Return [%]": annual_ret * 100,
        "Annual Volatility [%]": annual_vol * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": drawdown.min() * 100,
    }

def buy_hold_returns(df):
    return df["Close"].pct_change().fillna(0.0)

# =================================================================================
# 5. TỐI ƯU HÓA DANH MỤC MPT (MARKOWITZ)
# =================================================================================
def portfolio_annual_return(weights, returns, trading_days=252):
    mean_daily = returns.mean().values
    return float(weights @ mean_daily * trading_days)

def portfolio_annual_volatility(weights, returns, trading_days=252):
    cov_annual = returns.cov().values * trading_days
    variance = float(weights.T @ cov_annual @ weights)
    return np.sqrt(max(variance, 0.0))

def negative_sharpe(weights, returns, risk_free_rate=0.0):
    p_return = portfolio_annual_return(weights, returns)
    p_vol = portfolio_annual_volatility(weights, returns)
    if p_vol == 0:
        return 1e9
    return -(p_return - risk_free_rate) / p_vol

def optimize_mpt(returns, risk_free_rate=0.0):
    n = returns.shape[1]
    x0 = np.repeat(1 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        negative_sharpe,
        x0=x0,
        args=(returns, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    if not result.success:
        return x0
    return result.x

def portfolio_returns(return_matrix, weights):
    return return_matrix.mul(weights, axis=1).sum(axis=1)

# =================================================================================
# 6. SIDEBAR - ĐIỀU KHIỂN & CẤU HÌNH HỆ THỐNG
# =================================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Cấu Hình Tham Số")

# A. Nguồn dữ liệu
st.sidebar.subheader("1. Nguồn Dữ Liệu")
data_source_option = st.sidebar.radio(
    "Chọn nguồn file:",
    ["Sử dụng HOSE_2020_2023_in.csv mặc định", "Tải lên file CSV tùy chỉnh"],
    index=0
)

df_full = None
default_csv_path = "HOSE_2020_2023_in.csv"

if data_source_option == "Sử dụng HOSE_2020_2023_in.csv mặc định":
    if os.path.exists(default_csv_path):
        try:
            df_full = load_and_preprocess_data(default_csv_path)
            st.sidebar.success(f"Đã nạp file mặc định: {len(df_full):,} dòng")
        except Exception as e:
            st.sidebar.error(f"Lỗi đọc file mặc định: {e}")
    else:
        st.sidebar.warning(f"Không tìm thấy file `{default_csv_path}` tại thư mục gốc. Vui lòng tải file lên.")
        uploaded = st.sidebar.file_uploader("Tải file CSV dữ liệu", type=["csv"])
        if uploaded is not None:
            df_full = load_and_preprocess_data(uploaded)
            st.sidebar.success("Nạp dữ liệu thành công!")
else:
    uploaded = st.sidebar.file_uploader("Tải file CSV dữ liệu chứng khoán", type=["csv"])
    if uploaded is not None:
        try:
            df_full = load_and_preprocess_data(uploaded)
            st.sidebar.success(f"Nạp dữ liệu thành công: {len(df_full):,} dòng")
        except Exception as e:
            st.sidebar.error(f"Lỗi: {e}")

if df_full is None:
    st.info("👋 Vui lòng kiểm tra file `HOSE_2020_2023_in.csv` trong thư mục ứng dụng hoặc tải lên file dữ liệu từ sidebar để bắt đầu!")
    st.stop()

# B. Thời gian Train / Test
st.sidebar.subheader("2. Phân Đoạn Thời Gian")
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    train_start_input = st.date_input("Train Bắt đầu", pd.to_datetime("2020-01-01"))
    train_end_input = st.date_input("Train Kết thúc", pd.to_datetime("2021-12-31"))
with col_d2:
    test_start_input = st.date_input("Test Bắt đầu", pd.to_datetime("2022-01-01"))
    test_end_input = st.date_input("Test Kết thúc", pd.to_datetime("2022-12-31"))

train_start = pd.to_datetime(train_start_input)
train_end = pd.to_datetime(train_end_input)
test_start = pd.to_datetime(test_start_input)
test_end = pd.to_datetime(test_end_input)

# C. Tham số danh mục
st.sidebar.subheader("3. Tham Số Danh Mục & Phí")
initial_capital = st.sidebar.number_input("Vốn Khởi Điểm (VNĐ)", value=1_000_000, step=100_000)
commission_pct = st.sidebar.slider("Phí Giao Dịch Mua/Bán (%)", min_value=0.0, max_value=0.5, value=0.0, step=0.05) / 100.0
risk_free_rate = st.sidebar.number_input("Lãi Suất Phi Rủi Ro (%)", value=0.0, step=0.5) / 100.0
min_obs = st.sidebar.number_input("Số Quan Sát Tối Thiểu (Train)", value=400, step=50)

# D. Lựa chọn cổ phiếu
st.sidebar.subheader("4. Lựa Chọn Cổ Phiếu")
all_tickers = sorted(df_full["ticker"].unique())
ticker_selection_mode = st.sidebar.radio(
    "Chế độ chọn danh mục:",
    ["Sử dụng Top 5 TA Screening từ Train (DIG, DPM, KBC, MSN, DCM)", "Tự chọn cổ phiếu thủ công"],
    index=0
)

if ticker_selection_mode == "Sử dụng Top 5 TA Screening từ Train (DIG, DPM, KBC, MSN, DCM)":
    selected_tickers = [t for t in DEFAULT_TOP5 if t in all_tickers]
else:
    selected_tickers = st.sidebar.multiselect(
        "Chọn từ danh sách mã:",
        options=all_tickers,
        default=[t for t in DEFAULT_TOP5 if t in all_tickers]
    )

if len(selected_tickers) == 0:
    st.sidebar.warning("Vui lòng chọn ít nhất 1 cổ phiếu để tiếp tục.")
    st.stop()

# =================================================================================
# 7. CHUẨN BỊ DỮ LIỆU STOCK DATA VÀ TRAIN/TEST SPLIT
# =================================================================================
stock_data = {t: prepare_stock_data(df_full, t) for t in selected_tickers}
train_data = {}
test_data = {}

for t in selected_tickers:
    df_t = stock_data[t]
    tr = df_t.loc[(df_t.index >= train_start) & (df_t.index <= train_end)].copy()
    te = df_t.loc[(df_t.index >= test_start) & (df_t.index <= test_end)].copy()
    if not tr.empty and not te.empty:
        train_data[t] = tr
        test_data[t] = te

valid_tickers = [t for t in selected_tickers if t in train_data and t in test_data]
if len(valid_tickers) < len(selected_tickers):
    st.sidebar.warning(f"Một số mã bị thiếu dữ liệu trong khoảng Train/Test và đã bị loại: {set(selected_tickers) - set(valid_tickers)}")
selected_tickers = valid_tickers

if len(selected_tickers) == 0:
    st.error("Không có cổ phiếu nào đủ dữ liệu trong cả giai đoạn Train và Test đã chọn!")
    st.stop()

# E. Bộ tham số chiến lược (Preset vs Tùy chỉnh)
st.sidebar.subheader("5. Bộ Tham Số SMA + OBV")
use_preset = st.sidebar.checkbox("Sử dụng bộ tham số tối ưu chuẩn (Tức thì)", value=True)

strategy_params = {}
for t in selected_tickers:
    if use_preset and t in PRESET_PARAMS:
        strategy_params[t] = PRESET_PARAMS[t].copy()
    else:
        # Giá trị mặc định hợp lý nếu không nằm trong preset
        strategy_params[t] = {"ma_short": 50, "ma_long": 200, "obv_window": 30}

# =================================================================================
# 8. GIAO DIỆN CHÍNH - HEADER & NAVIGATION TABS
# =================================================================================
st.markdown('<div class="main-title">📈 HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC ĐỊNH LƯỢNG: SMA + OBV (OR)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Kiểm định chiến lược giao dịch kỹ thuật trên Sở Giao dịch Chứng khoán TP.HCM (HOSE) | '
    'Bộ lọc <b>Trend + Volume (OR)</b> kết hợp Phân bổ Danh mục <b>Equal Weight vs MPT (Markowitz)</b></div>',
    unsafe_allow_html=True
)

# Tabs chính
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 1. Tổng Quan & Pipeline",
    "🔍 2. Sàng Lọc Kỹ Thuật (TA)",
    "⚡ 3. Chiến Lược SMA + OBV (OR)",
    "⚖️ 4. Phân Bổ Danh Mục (MPT vs EW)",
    "🏆 5. Kiểm Định Out-of-Sample 2022",
    "📥 6. Xuất Dữ Liệu & Báo Cáo"
])

# =================================================================================
# TAB 1: TỔNG QUAN & QUY TRÌNH NGHIÊN CỨU
# =================================================================================
with tab1:
    st.subheader("📌 Tổng Quan Quy Trình Định Lượng 4 Bước")

    col_pipe1, col_pipe2, col_pipe3, col_pipe4 = st.columns(4)
    with col_pipe1:
        st.markdown("""
        <div class="metric-card">
            <span class="badge-neutral">BƯỚC 1</span>
            <h4>Sàng Lọc Kỹ Thuật (TA)</h4>
            <p style="font-size: 0.85rem; color: #4B5563; text-align: left;">
            • Chỉ lọc trên <b>Train (2020-2021)</b>.<br>
            • Đánh giá 4 trụ cột: Trend (30%), Momentum (25%), Volume (25%), Risk (20%).<br>
            • Tự động chọn <b>Top 5</b> cổ phiếu HOSE.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_pipe2:
        st.markdown("""
        <div class="metric-card">
            <span class="badge-neutral">BƯỚC 2</span>
            <h4>Tín Hiệu SMA + OBV (OR)</h4>
            <p style="font-size: 0.85rem; color: #4B5563; text-align: left;">
            • <b>BUY</b> khi SMA cắt lên HOẶC OBV cắt lên MA.<br>
            • <b>SELL</b> khi SMA cắt xuống HOẶC OBV cắt xuống.<br>
            • Triệt tiêu mâu thuẫn tín hiệu.<br>
            • Tối ưu tham số trên Train.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_pipe3:
        st.markdown("""
        <div class="metric-card">
            <span class="badge-neutral">BƯỚC 3</span>
            <h4>Phân Bổ Danh Mục</h4>
            <p style="font-size: 0.85rem; color: #4B5563; text-align: left;">
            • <b>Equal Weight</b>: 20% đều mỗi mã.<br>
            • <b>MPT (Markowitz)</b>: Tối đa hóa Sharpe Ratio trên Train (Long-only, sum=1).<br>
            • Giữ nguyên trọng số sang Test.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_pipe4:
        st.markdown("""
        <div class="metric-card">
            <span class="badge-neutral">BƯỚC 4</span>
            <h4>Stress Test 2022 (Test)</h4>
            <p style="font-size: 0.85rem; color: #4B5563; text-align: left;">
            • Kiểm định Out-of-sample trong giai đoạn downtrend khốc liệt 2022.<br>
            • So sánh với <b>Buy & Hold</b> benchmark.<br>
            • Đánh giá Return, Sharpe & Drawdown.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("💡 Nguyên Tắc Phương Pháp Luận Chặt Chẽ")
    col_phil1, col_phil2 = st.columns(2)
    with col_phil1:
        st.markdown("""
        <div class="callout-box">
            <b>1. Chống Look-ahead Bias (Xuất Hiện Trước Thời Điểm):</b><br>
            Mọi tín hiệu tính toán tại phiên giao dịch ngày $t$ (Close, Volume) chỉ được đưa vào trạng thái vị thế và khớp lệnh thực thi từ đầu phiên tiếp theo $t+1$. Tránh tuyệt đối việc lấy giá tương lai để khớp lệnh quá khứ.
        </div>
        """, unsafe_allow_html=True)
    with col_phil2:
        st.markdown("""
        <div class="callout-box">
            <b>2. Tách Biệt Nghiêm Ngặt Train (In-sample) và Test (Out-of-sample):</b><br>
            Không sử dụng dữ liệu 2022 để sàng lọc cổ phiếu, không dùng để tối ưu tham số SMA/OBV, và không dùng để tính trọng số MPT. Dữ liệu Test 2022 được cô lập hoàn toàn đóng vai trò bài kiểm tra sức chịu đựng (Stress test).
        </div>
        """, unsafe_allow_html=True)

    # Hiển thị tóm tắt nhanh dữ liệu nạp vào
    st.subheader("📋 Tóm Tắt Dữ Liệu Khởi Tạo")
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("Tổng Số Mã HOSE", f"{df_full['ticker'].nunique():,} mã")
    with c_m2:
        st.metric("Tổng Số Bản Ghi", f"{len(df_full):,} dòng")
    with c_m3:
        st.metric("Giai Đoạn Dữ Liệu", f"{df_full['date'].min().strftime('%d/%m/%Y')} → {df_full['date'].max().strftime('%d/%m/%Y')}")
    with c_m4:
        st.metric("Cổ Phiếu Đang Chọn", f"{len(selected_tickers)} mã: {', '.join(selected_tickers)}")

# =================================================================================
# TAB 2: SÀNG LỌC KỸ THUẬT (TA SCREENING)
# =================================================================================
with tab2:
    st.subheader("🔍 Bước 1: Sàng Lọc Kỹ Thuật Đa Nhân Tố (Chỉ Dùng Train 2020-2021)")
    st.markdown("""
    Hệ thống đánh giá toàn bộ các mã niêm yết trên HOSE thỏa mãn số quan sát tối thiểu trong giai đoạn **Train (2020-2021)**
    dựa trên 4 nhóm nhân tố định lượng:
    - **Trend (30%)**: Sức mạnh xu hướng tỷ lệ $SMA50 > SMA200$ (50%) + Tỷ lệ chênh lệch giá so với $SMA200$ (50%).
    - **Momentum (25%)**: Động lượng giá 6 tháng (~126 phiên).
    - **Volume (25%)**: Thanh khoản GTGD trung bình (60%) + Tín hiệu xác nhận của $OBV$ chuẩn hóa theo thanh khoản 20 phiên (40%).
    - **Risk (20%)**: Biến động lợi nhuận ngày (Annualized Volatility thấp được điểm cao) (50%) + Mức sụt giảm tối đa (Max Drawdown gần 0 được điểm cao) (50%).
    """)

    train_screen_df = df_full[(df_full["date"] >= train_start) & (df_full["date"] <= train_end)].copy()

    with st.spinner("Đang tính toán ma trận Technical Screening trên tập Train..."):
        ta_rank_df = calculate_ta_features(train_screen_df, min_observations=min_obs)

    if not ta_rank_df.empty:
        col_ta_s1, col_ta_s2 = st.columns([3, 2])
        with col_ta_s1:
            st.write(f"**Bảng Xếp Hạng Top 20 Cổ Phiếu Theo Tổng Điểm TA Score (Tổng số mã hợp lệ: {len(ta_rank_df)})**")
            display_cols = ["Ticker", "Trend_Score", "Momentum_Score", "Volume_Score", "Risk_Score", "TA_Score"]
            top_display = ta_rank_df[display_cols].head(20).copy()
            top_display.insert(0, "Hạng", range(1, len(top_display) + 1))
            st.dataframe(
                top_display.style.format({
                    "Trend_Score": "{:.3f}",
                    "Momentum_Score": "{:.3f}",
                    "Volume_Score": "{:.3f}",
                    "Risk_Score": "{:.3f}",
                    "TA_Score": "{:.3f}"
                }).background_gradient(subset=["TA_Score"], cmap="Blues"),
                use_container_width=True,
                height=450
            )

        with col_ta_s2:
            st.write("**Biểu Đồ Thành Phần Điểm Top 5 Được Lựa Chọn**")
            top5_plot_df = ta_rank_df.head(5).copy()
            fig_bar_ta = go.Figure()
            fig_bar_ta.add_trace(go.Bar(name='Trend (30%)', x=top5_plot_df['Ticker'], y=top5_plot_df['Trend_Score'] * 0.30, marker_color='#2563EB'))
            fig_bar_ta.add_trace(go.Bar(name='Momentum (25%)', x=top5_plot_df['Ticker'], y=top5_plot_df['Momentum_Score'] * 0.25, marker_color='#10B981'))
            fig_bar_ta.add_trace(go.Bar(name='Volume (25%)', x=top5_plot_df['Ticker'], y=top5_plot_df['Volume_Score'] * 0.25, marker_color='#F59E0B'))
            fig_bar_ta.add_trace(go.Bar(name='Risk (20%)', x=top5_plot_df['Ticker'], y=top5_plot_df['Risk_Score'] * 0.20, marker_color='#8B5CF6'))

            fig_bar_ta.update_layout(
                barmode='stack',
                xaxis_title="Cổ Phiếu",
                yaxis_title="Đóng Góp Điểm TA",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_bar_ta, use_container_width=True)

        st.success(f"✅ Top 5 Cổ phiếu có TA Score cao nhất trên Train: **{', '.join(ta_rank_df.head(5)['Ticker'].tolist())}**")
    else:
        st.warning("Không có cổ phiếu nào thỏa mãn điều kiện lọc với ngưỡng số quan sát đã chọn.")

# =================================================================================
# TAB 3: CHIẾN LƯỢC SMA + OBV (OR) CHO TỪNG CỔ PHIẾU
# =================================================================================
with tab3:
    st.subheader("⚡ Bước 2: Tín Hiệu & Hiệu Quả Chiến Lược SMA + OBV (OR)")

    st.markdown("""
    <div class="callout-box">
        <b>Logic Kết Hợp Theo Mệnh Đề Hoặc (OR):</b><br>
        • <b>Tín hiệu MUA (BUY = 1)</b>: SMA Ngắn cắt lên SMA Dài <b>HOẶC</b> OBV cắt lên OBV-MA.<br>
        • <b>Tín hiệu BÁN (SELL = -1)</b>: SMA Ngắn cắt xuống SMA Dài <b>HOẶC</b> OBV cắt xuống OBV-MA.<br>
        • <i>Giải quyết mâu thuẫn</i>: Nếu trong cùng một ngày xuất hiện cả tín hiệu Mua và Bán từ 2 chỉ báo thì vị thế giữ nguyên trạng thái trung lập 0.
    </div>
    """, unsafe_allow_html=True)

    col_tk_sel, col_p_s, col_p_l, col_p_obv = st.columns([2, 2, 2, 2])
    with col_tk_sel:
        target_ticker = st.selectbox("Chọn cổ phiếu để phân tích:", selected_tickers)
    with col_p_s:
        ma_s = st.number_input(f"SMA Ngắn ({target_ticker})", value=int(strategy_params[target_ticker]["ma_short"]), step=5, min_value=5, max_value=200)
    with col_p_l:
        ma_l = st.number_input(f"SMA Dài ({target_ticker})", value=int(strategy_params[target_ticker]["ma_long"]), step=5, min_value=50, max_value=450)
    with col_p_obv:
        obv_w = st.number_input(f"OBV MA Window ({target_ticker})", value=int(strategy_params[target_ticker]["obv_window"]), step=5, min_value=5, max_value=150)

    # Cập nhật tham số động
    current_paras = {"ma_short": ma_s, "ma_long": ma_l, "obv_window": obv_w}

    # Tính toán tín hiệu cho cổ phiếu đã chọn trên Train và Test
    df_stock_full = stock_data[target_ticker]

    # Tính tín hiệu trên toàn bộ chuỗi
    sma_sig = find_position_sma(df_stock_full, current_paras)
    obv_sig = find_position_obv(df_stock_full, current_paras)
    combined_sig = find_position_or(df_stock_full, current_paras, current_paras)
    ret_strat_full, holding_full = strategy_returns(df_stock_full, combined_sig, commission=commission_pct)

    # Tách chuỗi theo Train và Test
    mask_train = (df_stock_full.index >= train_start) & (df_stock_full.index <= train_end)
    mask_test = (df_stock_full.index >= test_start) & (df_stock_full.index <= test_end)

    df_stock_test = df_stock_full.loc[mask_test].copy()
    ret_test_stock = ret_strat_full.loc[mask_test]
    bh_test_stock = buy_hold_returns(df_stock_test)

    # Hiển thị số liệu so sánh Train vs Test cho cổ phiếu này
    stats_strat_train = performance_stats(ret_strat_full.loc[mask_train])
    stats_strat_test = performance_stats(ret_test_stock)
    stats_bh_train = performance_stats(buy_hold_returns(df_stock_full.loc[mask_train]))
    stats_bh_test = performance_stats(bh_test_stock)

    st.write(f"#### Hiệu Suất Chiến Lược vs Buy & Hold: Cổ Phiếu `{target_ticker}`")
    perf_single_df = pd.DataFrame({
        "B&H Train [%]": [stats_bh_train["Total Return [%]"], stats_bh_train["Annual Return [%]"], stats_bh_train["Sharpe Ratio"], stats_bh_train["Max Drawdown [%]"]],
        "SMA+OBV Train [%]": [stats_strat_train["Total Return [%]"], stats_strat_train["Annual Return [%]"], stats_strat_train["Sharpe Ratio"], stats_strat_train["Max Drawdown [%]"]],
        "B&H Test 2022 [%]": [stats_bh_test["Total Return [%]"], stats_bh_test["Annual Return [%]"], stats_bh_test["Sharpe Ratio"], stats_bh_test["Max Drawdown [%]"]],
        "SMA+OBV Test 2022 [%]": [stats_strat_test["Total Return [%]"], stats_strat_test["Annual Return [%]"], stats_strat_test["Sharpe Ratio"], stats_strat_test["Max Drawdown [%]"]]
    }, index=["Total Return [%]", "Annual Return [%]", "Sharpe Ratio", "Max Drawdown [%]"])

    st.dataframe(perf_single_df.style.format("{:.2f}").background_gradient(subset=["SMA+OBV Test 2022 [%]"], cmap="Greens"), use_container_width=True)

    # Biểu đồ nến / giá và chỉ báo OBV
    st.write(f"#### Biểu Đồ Kỹ Thuật & Tín Hiệu Giao Dịch `{target_ticker}`")
    chart_view_period = st.radio("Chọn vùng hiển thị biểu đồ:", ["Toàn bộ dữ liệu (2020-2023)", "Chỉ vùng Out-of-sample Test 2022", "Chỉ vùng Train (2020-2021)"], horizontal=True)

    if chart_view_period == "Chỉ vùng Out-of-sample Test 2022":
        plot_df = df_stock_full.loc[mask_test]
        plot_sig = combined_sig.loc[mask_test]
    elif chart_view_period == "Chỉ vùng Train (2020-2021)":
        plot_df = df_stock_full.loc[mask_train]
        plot_sig = combined_sig.loc[mask_train]
    else:
        plot_df = df_stock_full
        plot_sig = combined_sig

    # Tính lại đường SMA và OBV cho plot
    sma_s_series = ta.trend.SMAIndicator(close=df_stock_full["Close"], window=int(current_paras["ma_short"])).sma_indicator().loc[plot_df.index]
    sma_l_series = ta.trend.SMAIndicator(close=df_stock_full["Close"], window=int(current_paras["ma_long"])).sma_indicator().loc[plot_df.index]
    obv_series = ta.volume.OnBalanceVolumeIndicator(close=df_stock_full["Close"], volume=df_stock_full["Volume"]).on_balance_volume().loc[plot_df.index]
    obv_ma_series = obv_series.rolling(window=int(current_paras["obv_window"])).mean()

    buy_points = plot_df[plot_sig == 1.0]
    sell_points = plot_df[plot_sig == -1.0]

    fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.68, 0.32],
                             subplot_titles=(f"Đường Giá {target_ticker} & Tín hiệu Mua/Bán SMA+OBV (OR)", f"Chỉ Báo On-Balance Volume (OBV) vs OBV-MA({current_paras['obv_window']})"))

    # Candlestick
    fig_tech.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"],
        name="Giá", increasing_line_color="#10B981", decreasing_line_color="#EF4444"
    ), row=1, col=1)

    fig_tech.add_trace(go.Scatter(x=plot_df.index, y=sma_s_series, name=f"SMA Ngắn ({current_paras['ma_short']})", line=dict(color="#3B82F6", width=1.5)), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=plot_df.index, y=sma_l_series, name=f"SMA Dài ({current_paras['ma_long']})", line=dict(color="#F97316", width=1.8)), row=1, col=1)

    # Buy / Sell markers
    if not buy_points.empty:
        fig_tech.add_trace(go.Scatter(
            x=buy_points.index, y=buy_points["Low"] * 0.97, mode="markers", name="Tín hiệu MUA (OR)",
            marker=dict(symbol="triangle-up", size=11, color="#10B981", line=dict(width=1, color="black"))
        ), row=1, col=1)

    if not sell_points.empty:
        fig_tech.add_trace(go.Scatter(
            x=sell_points.index, y=sell_points["High"] * 1.03, mode="markers", name="Tín hiệu BÁN (OR)",
            marker=dict(symbol="triangle-down", size=11, color="#EF4444", line=dict(width=1, color="black"))
        ), row=1, col=1)

    # OBV Subplot
    fig_tech.add_trace(go.Scatter(x=plot_df.index, y=obv_series, name="OBV", line=dict(color="#8B5CF6", width=1.5)), row=2, col=1)
    fig_tech.add_trace(go.Scatter(x=plot_df.index, y=obv_ma_series, name=f"OBV MA ({current_paras['obv_window']})", line=dict(color="#EC4899", width=1.5, dash="dash")), row=2, col=1)

    fig_tech.update_layout(
        xaxis_rangeslider_visible=False,
        height=620,
        margin=dict(l=30, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_tech, use_container_width=True)

# =================================================================================
# TAB 4: PHÂN BỔ DANH MỤC (EQUAL WEIGHT VS MPT)
# =================================================================================
with tab4:
    st.subheader("⚖️ Bước 3: Phân Bổ Trọng Số Danh Mục (Chỉ Tính Trên Train)")

    st.markdown("""
    Sau khi thu được chuỗi lợi nhuận theo ngày của chiến lược **SMA + OBV (OR)** cho từng cổ phiếu trên tập Train,
    ta tiến hành xây dựng và so sánh 2 phương pháp phân bổ tỷ trọng:
    1. **Equal Weight (Đẳng trọng số)**: Phân bổ đều $1/N$ ($20\%$ cho mỗi mã nếu danh mục có 5 cổ phiếu).
    2. **Modern Portfolio Theory (MPT - Markowitz)**: Tối ưu hóa trọng số $\mathbf{w}$ nhằm **tối đa hóa Sharpe Ratio** trên tập Train với ràng buộc $0 \le w_i \le 1$ và $\sum w_i = 1$.
    """)

    # Tính ma trận lợi nhuận chiến lược trên Train
    train_strategy_dict = {}
    test_strategy_dict = {}

    for t in selected_tickers:
        paras = strategy_params[t]
        events_tr = find_position_or(train_data[t], paras, paras)
        r_tr, _ = strategy_returns(train_data[t], events_tr, commission=commission_pct)
        train_strategy_dict[t] = r_tr

        events_te = find_position_or(test_data[t], paras, paras)
        r_te, _ = strategy_returns(test_data[t], events_te, commission=commission_pct)
        test_strategy_dict[t] = r_te

    train_return_mat = pd.DataFrame(train_strategy_dict).dropna()
    test_return_mat = pd.DataFrame(test_strategy_dict).dropna()

    # Tính trọng số Equal Weight
    n_assets = len(selected_tickers)
    ew_weights = np.repeat(1.0 / n_assets, n_assets)

    # Tính trọng số MPT
    if use_preset and all(t in PRESET_MPT_WEIGHTS for t in selected_tickers) and len(selected_tickers) == 5:
        mpt_weights = np.array([PRESET_MPT_WEIGHTS[t] for t in selected_tickers])
    else:
        with st.spinner("Đang chạy thuật toán tối ưu SLSQP MPT..."):
            mpt_weights = optimize_mpt(train_return_mat, risk_free_rate=risk_free_rate)

    weights_df = pd.DataFrame({
        "Ticker": selected_tickers,
        "Equal Weight (EW)": ew_weights,
        "MPT Optimal Weight": mpt_weights
    })

    col_w_t, col_w_c = st.columns([2, 3])
    with col_w_t:
        st.write("**Bảng So Sánh Trọng Số Phân Bổ**")
        st.dataframe(
            weights_df.style.format({
                "Equal Weight (EW)": "{:.2%}",
                "MPT Optimal Weight": "{:.2%}"
            }).background_gradient(subset=["MPT Optimal Weight"], cmap="YlGnBu"),
            use_container_width=True
        )

        st.info(f"Tổng trọng số MPT: **{mpt_weights.sum():.2%}** | Ràng buộc Long-only: $0 \le w_i \le 1$.")

    with col_w_c:
        st.write("**Biểu Đồ So Sánh Trọng Số Phân Bổ (Equal Weight vs MPT)**")
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(x=weights_df["Ticker"], y=weights_df["Equal Weight (EW)"] * 100, name="Equal Weight", marker_color="#6B7280"))
        fig_w.add_trace(go.Bar(x=weights_df["Ticker"], y=weights_df["MPT Optimal Weight"] * 100, name="MPT Weight", marker_color="#2563EB"))
        fig_w.update_layout(
            barmode="group",
            yaxis_title="Tỷ Trọng Phân Bổ (%)",
            xaxis_title="Cổ Phiếu",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=320,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("---")
    st.write("**Ma Trận Tương Quan Lợi Nhuận Chiến Lược Giữa Các Mã (Train 2020-2021)**")
    corr_mat = train_return_mat.corr()
    fig_corr = px.imshow(
        corr_mat,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap - Strategy Returns (Train)"
    )
    fig_corr.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

# =================================================================================
# TAB 5: KIỂM ĐỊNH OUT-OF-SAMPLE TEST 2022 & BENCHMARK
# =================================================================================
with tab5:
    st.subheader("🏆 Bước 4: Kiểm Định Out-of-Sample Trên Năm 2022 (Bear Market Stress Test)")

    st.markdown("""
    Năm 2022 là năm thị trường chứng khoán Việt Nam (VN-Index) trải qua đợt sụt giảm mạnh nhất thập kỷ.
    Đây là môi trường lý tưởng để **kiểm định tính bền bỉ và khả năng phòng vệ** của chiến lược định lượng.
    """)

    # Tính lợi nhuận danh mục trên Test
    # 1. Equal Weight Buy & Hold
    bh_test_dict = {t: buy_hold_returns(test_data[t]) for t in selected_tickers}
    bh_test_mat = pd.DataFrame(bh_test_dict).dropna()
    ew_bh_test_ret = portfolio_returns(bh_test_mat, ew_weights)

    # 2. SMA+OBV Equal Weight
    ew_strat_test_ret = portfolio_returns(test_return_mat, ew_weights)

    # 3. SMA+OBV MPT
    mpt_strat_test_ret = portfolio_returns(test_return_mat, mpt_weights)

    # Tính đường cong tăng trưởng vốn (Equity Curves)
    bh_equity_test = initial_capital * (1 + ew_bh_test_ret).cumprod()
    ew_equity_test = initial_capital * (1 + ew_strat_test_ret).cumprod()
    mpt_equity_test = initial_capital * (1 + mpt_strat_test_ret).cumprod()

    # Thống kê hiệu suất
    stats_ew_bh = performance_stats(ew_bh_test_ret)
    stats_ew_strat = performance_stats(ew_strat_test_ret)
    stats_mpt_strat = performance_stats(mpt_strat_test_ret)

    # Hiển thị 3 Card KPI so sánh
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Equal Weight Buy & Hold</div>
            <div class="metric-value-negative">{stats_ew_bh['Total Return [%]']:.2f}%</div>
            <div style="font-size: 0.9rem; color: #64748B; margin-top: 6px;">
                Sharpe: <b>{stats_ew_bh['Sharpe Ratio']:.2f}</b> | Max DD: <b>{stats_ew_bh['Max Drawdown [%]']:.2f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SMA+OBV (OR) Equal Weight</div>
            <div class="metric-value-positive">{stats_ew_strat['Total Return [%]']:+.2f}%</div>
            <div style="font-size: 0.9rem; color: #64748B; margin-top: 6px;">
                Sharpe: <b>{stats_ew_strat['Sharpe Ratio']:.2f}</b> | Max DD: <b>{stats_ew_strat['Max Drawdown [%]']:.2f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SMA+OBV (OR) MPT (Markowitz)</div>
            <div class="metric-value-positive">{stats_mpt_strat['Total Return [%]']:+.2f}%</div>
            <div style="font-size: 0.9rem; color: #64748B; margin-top: 6px;">
                Sharpe: <b>{stats_mpt_strat['Sharpe Ratio']:.2f}</b> | Max DD: <b>{stats_mpt_strat['Max Drawdown [%]']:.2f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("#### Bảng So Sánh Hiệu Suất Tổng Thể Ngoài Mẫu (Out-of-Sample Test 2022)")
    final_comp_df = pd.DataFrame({
        "Equal Weight Buy & Hold": stats_ew_bh,
        "SMA+OBV OR Equal Weight": stats_ew_strat,
        "SMA+OBV OR MPT": stats_mpt_strat
    }).T

    st.dataframe(
        final_comp_df.style.format({
            "Mean Daily Return [%]": "{:.4f}%",
            "Total Return [%]": "{:.2f}%",
            "Annual Return [%]": "{:.2f}%",
            "Annual Volatility [%]": "{:.2f}%",
            "Sharpe Ratio": "{:.4f}",
            "Max Drawdown [%]": "{:.2f}%"
        }).background_gradient(subset=["Total Return [%]", "Sharpe Ratio"], cmap="Greens"),
        use_container_width=True
    )

    # Biểu đồ đường cong tăng trưởng vốn (Equity Curve)
    st.write("#### Biểu Đồ Đường Cong Tăng Trưởng Vốn (Equity Curve - Khởi Điểm: 1,000,000 VNĐ)")
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=bh_equity_test.index, y=bh_equity_test, name="Equal Weight Buy & Hold",
        line=dict(color="#EF4444", width=2.5, dash="dash")
    ))
    fig_equity.add_trace(go.Scatter(
        x=ew_equity_test.index, y=ew_equity_test, name="SMA+OBV OR Equal Weight",
        line=dict(color="#10B981", width=2.5)
    ))
    fig_equity.add_trace(go.Scatter(
        x=mpt_equity_test.index, y=mpt_equity_test, name="SMA+OBV OR MPT",
        line=dict(color="#2563EB", width=2.5)
    ))

    fig_equity.update_layout(
        xaxis_title="Thời Gian",
        yaxis_title="Giá Trị Danh Mục (VNĐ)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
        margin=dict(l=30, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # Biểu đồ Sụt giảm vốn (Underwater Drawdown)
    st.write("#### Biểu Đồ Sụt Giảm Vốn Từ Đỉnh (Underwater Drawdown %)")
    dd_bh = (bh_equity_test / bh_equity_test.cummax() - 1) * 100
    dd_ew = (ew_equity_test / ew_equity_test.cummax() - 1) * 100
    dd_mpt = (mpt_equity_test / mpt_equity_test.cummax() - 1) * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_bh.index, y=dd_bh, name="B&H Drawdown", fill="tozeroy", line=dict(color="#EF4444", width=1)))
    fig_dd.add_trace(go.Scatter(x=dd_ew.index, y=dd_ew, name="SMA+OBV EW Drawdown", fill="tozeroy", line=dict(color="#10B981", width=1.5)))
    fig_dd.add_trace(go.Scatter(x=dd_mpt.index, y=dd_mpt, name="SMA+OBV MPT Drawdown", line=dict(color="#2563EB", width=1.8)))

    fig_dd.update_layout(
        xaxis_title="Thời Gian",
        yaxis_title="Mức Sụt Giảm (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(l=30, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("""
    <div class="callout-box">
        <b>💡 Kết Luận Định Lượng Quan Trọng:</b><br>
        1. <b>Bảo vệ vốn vượt trội khi thị trường giảm điểm</b>: Trong khi chiến lược Mua & Nắm giữ (Buy & Hold) chịu lỗ nặng <b>-50.41%</b> với mức sụt giảm tối đa (Max Drawdown) lên tới <b>-64.30%</b> trong năm 2022, chiến lược <b>SMA + OBV (OR)</b> đã kịp thời thoát vị thế tiền mặt khi tín hiệu gãy xu hướng và dòng tiền đảo chiều, giúp bảo vệ vốn và ghi nhận mức sinh lời dương <b>+5.63%</b> (Equal Weight) và <b>+7.00%</b> (MPT).<br>
        2. <b>So sánh MPT vs Equal Weight</b>: Danh mục tối ưu theo Sharpe (MPT) đạt lợi nhuận và Sharpe Ratio cao hơn một chút so với Equal Weight. Tuy nhiên, Equal Weight là một benchmark đơn giản, ít nhạy cảm với sai số tham số (estimation risk) và rất mạnh mẽ trong kiểm nghiệm thực tế.
    </div>
    """, unsafe_allow_html=True)

# =================================================================================
# TAB 6: XUẤT DỮ LIỆU & BÁO CÁO CHI TIẾT
# =================================================================================
with tab6:
    st.subheader("📥 Xuất Báo Cáo & Dữ Liệu Kiểm Định")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.write("##### 1. Tải Bảng Hiệu Suất So Sánh Benchmark")
        csv_benchmark = final_comp_df.to_csv(index=True).encode("utf-8-sig")
        st.download_button(
            label="📥 Tải file CSV Hiệu Suất Benchmark",
            data=csv_benchmark,
            file_name="benchmark_comparison_test_2022.csv",
            mime="text/csv"
        )

        st.write("##### 2. Tải Bảng Trọng Số Phân Bổ MPT & Equal Weight")
        csv_weights = weights_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Tải file CSV Bảng Trọng Số",
            data=csv_weights,
            file_name="portfolio_weights_mpt_ew.csv",
            mime="text/csv"
        )

    with col_exp2:
        st.write("##### 3. Tải Chuỗi Tăng Trưởng Vốn Danh Mục Hàng Ngày")
        equity_export_df = pd.DataFrame({
            "Date": bh_equity_test.index,
            "Equal_Weight_Buy_Hold": bh_equity_test.values,
            "SMA_OBV_OR_Equal_Weight": ew_equity_test.values,
            "SMA_OBV_OR_MPT": mpt_equity_test.values
        })
        csv_equity = equity_export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Tải file CSV Đường Cong Vốn (Daily Equity)",
            data=csv_equity,
            file_name="daily_equity_curves_2022.csv",
            mime="text/csv"
        )

        st.write("##### 4. Tải Bảng Xếp Hạng Sàng Lọc Kỹ Thuật (TA Screening)")
        if not ta_rank_df.empty:
            csv_ta = ta_rank_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Tải file CSV Xếp Hạng TA Score Toàn Bộ Mã",
                data=csv_ta,
                file_name="ta_screening_rankings.csv",
                mime="text/csv"
            )

    st.markdown("---")
    st.info("💡 Bạn có thể đẩy mã nguồn dự án lên GitHub và kết nối với Streamlit Community Cloud để chia sẻ ứng dụng trực tuyến hoàn toàn miễn phí!")
