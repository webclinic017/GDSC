import streamlit as st
import yfinance as yf
import backtrader as bt
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 定義 RSI 策略
class RSIStrategy(bt.Strategy):
    params = (
        ('printlog', True),
        ('rsi_period', None),
        ('rsi_overbought', None),
        ('rsi_oversold', None),
        ('trade_amount', None),  # 每次交易的固定投入金額
    )

    def __init__(self):
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.data.close, period=self.params.rsi_period)

    def next(self):
        if not self.position:  # 沒有持倉
            if self.rsi < self.params.rsi_oversold:
                size = self.params.trade_amount // self.data.close[0]
                self.buy(size=size)  # RSI 低於超賣區域，買入
        else:
            if self.rsi > self.params.rsi_overbought:
                self.sell(size=self.position.size)  # RSI 高於超買區域，賣出

# Streamlit 應用程式
st.title("RSI 股票交易策略")

st.markdown("""
## RSI 策略解釋

RSI（相對強弱指數）是一種動量指標，用於衡量股票價格變動的速度和變動的幅度。RSI 的值在 0 到 100 之間波動，通常用來識別超買和超賣狀態。

### 策略邏輯
1. **RSI 計算**：
   - RSI 是根據一定的週期（例如 14 天）計算的。
   - 當 RSI 的值低於某個閾值（例如 30）時，表示股票可能被超賣，這是一個買入信號。
   - 當 RSI 的值高於某個閾值（例如 70）時，表示股票可能被超買，這是一個賣出信號。

2. **交易決策**：
   - 當 RSI 低於超賣區域（例如 30）時，策略會買入股票。
   - 當 RSI 高於超買區域（例如 70）時，策略會賣出股票。

### 使用方法
1. 輸入股票符號、開始和結束日期。
2. 設置 RSI 的週期、超買和超賣閾值。
3. 設置初始現金和每次交易的固定投入金額。
4. 設置交易手續費。
5. 點擊“開始回測”按鈕，運行回測策略並顯示結果。
""")

# 使用者輸入參數
symbol = st.text_input("股票符號", "AAPL")
start_date = st.date_input("開始日期", datetime(2020, 1, 1))
end_date = st.date_input("結束日期", datetime.today())
rsi_period = st.slider("RSI 週期", 1, 50, 14)
rsi_overbought = st.slider("RSI 超買區域", 50, 100, 70)
rsi_oversold = st.slider("RSI 超賣區域", 0, 50, 30)
initial_cash = st.slider("初始现金", min_value=0, max_value=10000000, step=10000, value=10000)
trade_amount = st.slider("每次交易金额", min_value=0, max_value=50000, step=1000, value=1000)
commission = st.slider('交易手續费 (%)', min_value=0.0, max_value=0.5, step=0.0005, format="%.4f", value=0.001)

if st.button("開始回測"):
    # 獲取股票數據
    data = yf.download(symbol, start=start_date, end=end_date)
    data = bt.feeds.PandasData(dataname=data)

    # 創建回測引擎
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(RSIStrategy, rsi_period=rsi_period, rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold, trade_amount=trade_amount)
    cerebro.broker.set_cash(initial_cash)
    cerebro.broker.setcommission(commission=commission / 100)  # 設置手續費

    # 運行回測
    st.write("開始回測...")
    initial_portfolio_value = cerebro.broker.getvalue()
    cerebro.run()
    final_portfolio_value = cerebro.broker.getvalue()

    # 顯示結果
    st.write(f"初始投資組合價值: ${initial_portfolio_value:.2f}")
    st.write(f"最終投資組合價值: ${final_portfolio_value:.2f}")
    st.write(f"盈虧: ${final_portfolio_value - initial_portfolio_value:.2f}")

    # 繪製回測結果
    fig = cerebro.plot(style='candlestick')[0][0]
    st.pyplot(fig)