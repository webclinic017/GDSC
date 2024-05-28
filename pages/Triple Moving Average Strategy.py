import streamlit as st
import yfinance as yf
import pandas as pd
import backtrader as bt
import matplotlib
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
matplotlib.use('Agg')

# 自定义指标
class MySignal(bt.Indicator):
    lines = ("signal",)
    params = dict(short_period=None, median_period=None, long_period=None)

    def __init__(self):
        self.s_ma = bt.ind.SMA(period=self.p.short_period)
        self.m_ma = bt.ind.SMA(period=self.p.median_period)
        self.l_ma = bt.ind.SMA(period=self.p.long_period)
        self.signal1 = bt.And(self.m_ma > self.l_ma, self.s_ma > self.m_ma)
        self.buy_signal = bt.If((self.signal1 - self.signal1(-1)) > 0, 1, 0)
        self.sell_signal = bt.ind.CrossDown(self.s_ma, self.m_ma)
        self.lines.signal = bt.Sum(self.buy_signal, self.sell_signal * (-1))

# 自定义Sizer
class FixedAmountSizer(bt.Sizer):
    params = (("amount", None),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            size = self.p.amount // data.close[0]
            return size
        return self.broker.getposition(data).size

# 策略
class TestStrategy(bt.Strategy):
    params = dict(
        printlog=True,
        short_period=None,
        median_period=None,
        long_period=None,
        initial_cash=None,
    )

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.signal = MySignal(
            self.datas[0],
            short_period=self.params.short_period,
            median_period=self.params.median_period,
            long_period=self.params.long_period
        )
        self.s_ma = bt.ind.SMA(period=self.params.short_period)
        self.m_ma = bt.ind.SMA(period=self.params.median_period)
        self.l_ma = bt.ind.SMA(period=self.params.long_period)
        bt.indicators.MACDHisto(self.datas[0])

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
        self.log("Close, %.2f" % self.dataclose[0])
        if self.order:
            return
        if not self.position:
            if self.signal.lines.signal[0] == 1:
                self.log("BUY CREATE, %.2f" % self.dataclose[0])
                self.order = self.buy()
        else:
            if self.signal.lines.signal[0] == -1:
                self.log("SELL CREATE, %.2f" % self.dataclose[0])
                self.order = self.sell()

    def stop(self):
        self.log("Ending Value %.2f" % (self.broker.getvalue()), doprint=True)

# Streamlit 用户界面
st.title("Backtrader with Streamlit")

# 策略說明
st.header("三均線策略說明")
st.markdown("""
三均線策略是一種基於短期、中期和長期簡單移動平均線（SMA）交叉來生成買賣信號的交易策略。
- **短期均線（SMA短期）**：通常設置為5天，用於捕捉近期價格趨勢。
- **中期均線（SMA中期）**：通常設置為20天，用於反映較長時間段的價格趨勢。
- **長期均線（SMA長期）**：通常設置為60天，用於反映更長時間段的價格趨勢。

**買入信號**：當短期均線向上穿越中期均線，且中期均線高於長期均線時，產生買入信號。

**賣出信號**：當短期均線向下穿越中期均線時，產生賣出信號。
""")

# 用户输入参数
symbol = st.text_input("股票符號", "AAPL")
start_date = st.date_input("開始日期", pd.to_datetime("2020-01-01"))
end_date = st.date_input("结束日期", pd.to_datetime("today"))

short_period = st.slider("短期均線", 1, 30, 5)
median_period = st.slider("中期均線", 15, 100, 20)
long_period = st.slider("長期均線", 30, 200, 60)
commission = st.slider('交易手續费 (%)', min_value=0.0, max_value=0.5, step=0.0005, format="%.4f", value=0.001)
trade_amount = st.slider("每次交易金额", min_value=0, max_value=50000, step=1000, value=1000)
initial_cash = st.slider("初始现金", min_value=0, max_value=10000000, step=10000, value=10000)

# 获取数据
if st.button("開始回测"):
    df = yf.download(symbol, start=start_date, end=end_date)
    df.dropna(inplace=True)
    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        TestStrategy,
        short_period=short_period,
        median_period=median_period,
        long_period=long_period,
        initial_cash=initial_cash
    )
    cerebro.adddata(data)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission / 100)  # 转换为十进制形式
    cerebro.addsizer(FixedAmountSizer, amount=trade_amount)  # 使用自定义的FixedAmountSizer

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