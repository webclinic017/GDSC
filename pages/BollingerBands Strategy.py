import streamlit as st
import yfinance as yf
import backtrader as bt
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.use('Agg')

# 定義布林通道策略
class BollingerBandsStrategy(bt.Strategy):
    params = (
        ('period', None),
        ('devfactor', None),
        ('trade_amount', None),  # 每次交易的固定投入金額
    )

    def __init__(self):
        self.bollinger = bt.indicators.BollingerBands(
            self.data.close, period=self.params.period, devfactor=self.params.devfactor)

    def next(self):
        if not self.position:  # 沒有持倉
            if self.data.close < self.bollinger.lines.bot:
                size = self.params.trade_amount // self.data.close[0]
                self.buy(size=size)  # 價格低於下軌線，買入
        else:
            if self.data.close > self.bollinger.lines.top:
                self.sell(size=self.position.size)  # 價格高於上軌線，賣出

# Streamlit 應用程式
st.title("布林通道股票交易策略")

st.markdown("""
## 布林通道策略解釋

布林通道（Bollinger Bands）是一種技術指標，由中間的移動平均線和上下兩條標準差線組成，用於識別價格的波動範圍。

### 策略邏輯
1. **布林通道計算**：
   - 中間線是一定週期的簡單移動平均線（SMA）。
   - 上軌線是中間線加上一定倍數的標準差。
   - 下軌線是中間線減去一定倍數的標準差。

2. **交易決策**：
   - 當價格低於下軌線時，買入股票。
   - 當價格高於上軌線時，賣出股票。

### 使用方法
1. 輸入股票符號、開始和結束日期。
2. 設置布林通道的週期和標準差倍數。
3. 設置初始現金和每次交易的固定投入金額。
4. 設置交易手續費。
5. 點擊“開始回測”按鈕，運行回測策略並顯示結果。
""")

# 使用者輸入參數
symbol = st.text_input("股票符號", "AAPL")
start_date = st.date_input("開始日期", datetime(2020, 1, 1))
end_date = st.date_input("結束日期", datetime.today())
period = st.slider("布林通道週期", 1, 50, 20)
devfactor = st.slider("標準差倍數", 1.0, 5.0, 2.0)
initial_cash = st.slider("初始现金", min_value=0, max_value=10000000, step=10000, value=10000)
trade_amount = st.slider("每次交易金额", min_value=0, max_value=50000, step=1000, value=1000)
commission = st.slider('交易手續費 (%)', min_value=0.0, max_value=1.0, step=0.01, value=0.1)

if st.button("開始回測"):
    # 獲取股票數據
    data = yf.download(symbol, start=start_date, end=end_date)
    data = bt.feeds.PandasData(dataname=data)

    # 創建回測引擎
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(BollingerBandsStrategy, period=period, devfactor=devfactor, trade_amount=trade_amount)
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