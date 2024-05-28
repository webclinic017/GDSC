import streamlit as st
import yfinance as yf
import pandas as pd
import backtrader as bt
import matplotlib
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

# 定義MACD策略
class MACDStrategy(bt.Strategy):
    params = (
        ('printlog', True),
        ('fast', None),
        ('slow', None),
        ('signal', None),
    )

    def __init__(self):
        macd = bt.indicators.MACD(self.data.close, 
                                  period_me1=self.params.fast, 
                                  period_me2=self.params.slow, 
                                  period_signal=self.params.signal)
        self.macd = macd.macd
        self.signal = macd.signal
        self.crossover = bt.indicators.CrossOver(self.macd, self.signal)

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(
                    "SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" % (trade.pnl, trade.pnlcomm))

    def next(self):
        self.log("Close, %.2f" % self.data.close[0])
        if self.crossover > 0:  # 買入信號
            if not self.position:
                self.log("BUY CREATE, %.2f" % self.data.close[0])
                self.buy()
        elif self.crossover < 0:  # 賣出信號
            if self.position:
                self.log("SELL CREATE, %.2f" % self.data.close[0])
                self.sell()

    def stop(self):
        self.log("Ending Value %.2f" % (self.broker.getvalue()), doprint=True)

# 自定义Sizer，根據每次交易的金額計算股數
class FixedCashSizer(bt.Sizer):
    params = (('cash', None),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return self.params.cash // data.close[0]
        return self.broker.getposition(data).size

# Streamlit 用户界面
st.title("Backtrader with Streamlit")

# 策略說明
st.header("MACD 策略說明")
st.markdown("""
MACD策略是一種基於移動平均線聚合散度（MACD）指標的交易策略。MACD 通常由三部分組成：MACD 線、信號線和柱狀圖。
- **MACD線**：由兩條指數移動平均線（EMA）的差值計算得來，通常使用12日和26日的EMA。
- **信號線**：MACD線的9日EMA，用於生成交易信號。
- **柱狀圖**：MACD線和信號線之間的差值，用於視覺化兩者之間的關係。

**買入信號**：當MACD線從下方穿過信號線時，產生買入信號。
**賣出信號**：當MACD線從上方穿過信號線時，產生賣出信號。
""")

# 用户输入参数
symbol = st.text_input("股票符號", "AAPL")
start_date = st.date_input("開始日期", pd.to_datetime("2020-01-01"))
end_date = st.date_input("结束日期", pd.to_datetime("today"))

fast_ema = st.slider('快速EMA周期', min_value=1, max_value=50, value=12)
slow_ema = st.slider('慢速EMA周期', min_value=1, max_value=50, value=26)
signal_ema = st.slider('信號EMA周期', min_value=1, max_value=50, value=9)
commission = st.slider('交易手續费 (%)', min_value=0.0, max_value=0.5, step=0.0005, format="%.4f", value=0.001)
trade_cash = st.slider("每次交易金额", min_value=0, max_value=50000, step=1000, value=1000)
initial_cash = st.slider("初始现金", min_value=0, max_value=10000000, step=10000, value=10000)

# 获取数据
if st.button("開始回测"):
    df = yf.download(symbol, start=start_date, end=end_date)
    df.dropna(inplace=True)
    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        MACDStrategy,
        fast=fast_ema,
        slow=slow_ema,
        signal=signal_ema
    )
    cerebro.adddata(data)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission / 100)  # 转换为十进制形式
    cerebro.addsizer(FixedCashSizer, cash=trade_cash)  # 使用自定义的FixedCashSizer

    st.write("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    cerebro.run()
    final_value = cerebro.broker.getvalue()
    st.write("Final Portfolio Value: %.2f" % final_value)

    # 计算年化报酬率
    duration_days = (end_date - start_date).days
    duration_years = duration_days / 365.25
    cagr = ((final_value / initial_cash) ** (1 / duration_years)) - 1
    st.write("年化报酬率: %.2f%%" % (cagr * 100))

    # 绘制结果
    fig = cerebro.plot(style='candlestick')[0][0]  # 获取 Matplotlib 图形对象
    st.pyplot(fig)  # 将图形嵌入到 Streamlit 页面中