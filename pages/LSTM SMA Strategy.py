import streamlit as st
import backtrader as bt
import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time  # 添加这一行
matplotlib.use('Agg')

# 函数：获取股票数据
def get_stock_data(code, start_date, end_date):
    df = yf.download(code, start=start_date, end=end_date)
    df = df.sort_index(ascending=True)
    df['SMA_10'] = df['Close'].rolling(window=short_period).mean()
    df['SMA_20'] = df['Close'].rolling(window=long_period).mean()
    df = df.dropna()
    return df

# 函数：将股票数据转换为模型训练数据集
def create_dataset(stock_data, window_size):
    X = []
    y = []
    scaler = MinMaxScaler()
    stock_data_normalized = scaler.fit_transform(stock_data.values)

    for i in range(len(stock_data) - window_size - 2):
        X.append(stock_data_normalized[i:i + window_size])
        if stock_data.iloc[i + window_size + 2]['Close'] > stock_data.iloc[i + window_size - 1]['Close']:
            y.append(1)
        else:
            y.append(0)

    X, y = np.array(X), np.array(y)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    return X, y, scaler

# 函数：创建DataLoader
def create_dataloader(X, y, batch_size):
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# LSTM 模型定义
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.bn(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# 定义策略
class LSTMStrategy(bt.Strategy):
    params = (
        ("window_size", 10),
        ("scaler", None),
        ("model", None),
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.sma10 = bt.indicators.SimpleMovingAverage(self.datas[0], period=short_period)
        self.sma20 = bt.indicators.SimpleMovingAverage(self.datas[0], period=long_period)
        self.counter = 1
        self.buyprice = None
        self.buycomm = None

    def log(self, txt, dt=None):
        pass  # 不再记录详细日志

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            self.bar_executed = len(self)

        self.order = None

    def notify_trade(self, trade):
        pass  # 不再记录详细日志

    def next(self):
        if self.counter < self.params.window_size:
            self.counter += 1
            return

        previous_features = [[self.data_close[-i], self.sma10[-i], self.sma20[-i]] for i in range(0, self.params.window_size)]
        X = torch.tensor(previous_features).view(1, self.params.window_size, -1).float()
        X = self.params.scaler.transform(X.numpy().reshape(-1, 3)).reshape(1, self.params.window_size, -1)

        # 將模型設置為評估模式
        self.params.model.eval()
        with torch.no_grad():
            prediction = self.params.model(torch.tensor(X).float())

        max_vals, max_idxs = torch.max(prediction, dim=1)
        predicted_trend = max_idxs.item()

        if predicted_trend == 1 and not self.position:
            self.order = self.buy()  # 买入股票
        elif predicted_trend == 0 and self.position:
            self.order = self.sell()
        elif self.position:
            # 這裡可以添加止損或止盈邏輯
            if self.data_close[0] < self.buyprice * 0.9:  # 假設止損點為買入價格的90%
                self.order = self.sell()
            elif self.data_close[0] > self.buyprice * 1.5:  # 假設止盈點為買入價格的150%
                self.order = self.sell()

# 定义训练LSTM模型的函数
def train_lstm():
    global input_size, hidden_size, num_layers, num_classes, scaler, lstm_model_ready, trained_model
    with st.spinner("Start training LSTM..."):
        start_time = time.time()
        max_training_time = 300  
        # 最大训练时间为300秒（5分钟）

        # 获取股票数据
        stock_data = get_stock_data(symbol, start_date, end_date)

        # 将股票数据转换为模型训练数据集
        window_size = 10
        X, y, scaler = create_dataset(stock_data[['Close', 'SMA_10', 'SMA_20']], window_size)

        # 定义批量大小和DataLoader
        batch_size = 64
        train_loader = create_dataloader(X, y, batch_size)

        # 模型参数定义
        input_size = 3  # 更新為特徵數
        hidden_size = 128
        num_layers = 2
        num_classes = 2

        # LSTM 模型初始化
        model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        num_epochs = 200

        # 训练模型
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch+1) % 10 == 0:
                st.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 保存训练好的模型到內存
        trained_model = model
        lstm_model_ready = True

# 定义backtrader相关的函数
def run_backtrader():
    global input_size, hidden_size, num_layers, num_classes, scaler, lstm_model_ready, trained_model
    with st.spinner("Running backtrader..."):
        while not lstm_model_ready:
            time.sleep(10)

        st.write("LSTM model loaded successfully.")

        # 获取股票数据
        stock_data = get_stock_data(symbol, start_date, end_date)

        # 创建Backtrader引擎
        cerebro = bt.Cerebro()

        # 设置初始资金
        cerebro.broker.set_cash(initial_cash)
        cerebro.broker.setcommission(commission=commission/100)

        # 添加策略并传递scaler和model
        cerebro.addstrategy(LSTMStrategy, scaler=scaler, model=trained_model)

        # 将数据添加到引擎中
        data = bt.feeds.PandasData(dataname=stock_data)
        cerebro.adddata(data)

        # 运行策略
        results = cerebro.run()

        # 打印最终的投资组合价值
        portvalue = cerebro.broker.getvalue()
        pnl = portvalue - initial_cash  # 初始资金为用户输入的值
        st.write(f'Final Portfolio Value: ${portvalue:.2f}')
        st.write(f'P/L: ${pnl:.2f}')

        # 計算投資報酬率（ROI）
        roi = (portvalue - initial_cash) / initial_cash * 100
        st.write(f'ROI: {roi:.2f}%')

        # 绘制回测结果
        fig = cerebro.plot(style='candlestick')[0][0]  # 获取 Matplotlib 图形对象
        st.pyplot(fig)  # 将图形嵌入到 Streamlit 页面中

# Streamlit 應用
st.title("LSTM 股票交易策略")

import streamlit as st

st.markdown("""
# 策略概述

這個策略使用了一個簡單的長短期移動平均線（SMA）和一個基於LSTM（長短期記憶）神經網絡的模型來進行股票交易。主要步驟如下：

1. **獲取股票數據**：從Yahoo Finance下載指定股票的歷史數據。
2. **計算移動平均線**：計算短期和長期的移動平均線（SMA）。
3. **創建訓練數據集**：將股票數據轉換為LSTM模型的訓練數據集。
4. **訓練LSTM模型**：使用訓練數據集來訓練LSTM模型。
5. **回測策略**：使用Backtrader回測引擎來運行交易策略，並評估其表現。

## 詳細步驟

### 1. 獲取股票數據
- 使用`yfinance`庫從Yahoo Finance下載股票數據。
- 計算短期和長期的移動平均線，並將其添加到數據框中。

### 2. 創建訓練數據集
- 將股票數據轉換為LSTM模型的訓練數據集。這裡使用了`MinMaxScaler`進行數據標準化。
- 根據窗口大小（`window_size`）創建特徵和標籤。特徵是窗口內的股票價格和移動平均線，標籤是窗口結束後的價格變動方向（上漲或下跌）。

### 3. 訓練LSTM模型
- 定義LSTM模型的結構，包括LSTM層、全連接層、激活函數、Dropout層和Batch Normalization層。
- 使用訓練數據集進行模型訓練，優化損失函數（交叉熵損失）並更新模型參數。

### 4. 回測策略
- 使用Backtrader回測引擎運行交易策略。
- 策略根據LSTM模型的預測結果進行交易決策。如果預測價格會上漲且目前沒有持倉，則買入股票；如果預測價格會下跌且目前有持倉，則賣出股票。
- 策略還包括止損和止盈邏輯：如果價格下跌超過10%或上漲超過50%，則賣出股票。

### 5. 結果展示
- 回測完成後，顯示最終的投資組合價值、盈虧（P/L）和投資報酬率（ROI）。
- 使用Matplotlib繪製回測結果的K線圖，並嵌入到Streamlit應用中展示。

## 使用方法

1. 在Streamlit應用中輸入股票符號、開始日期和結束日期。
2. 調整短期和長期移動平均線的參數、交易手續費、每次交易金額和初始現金。
3. 點擊“開始回測”按鈕，系統會自動訓練LSTM模型並運行回測策略，最終展示回測結果。

這個策略結合了技術分析（移動平均線）和機器學習（LSTM模型）的優勢，旨在提高交易決策的準確性和收益率。
""")

# 用户输入参数
symbol = st.text_input("股票符號", "AAPL")
start_date = st.date_input("開始日期", pd.to_datetime("2020-01-01"))
end_date = st.date_input("结束日期", pd.to_datetime("today"))

short_period = st.slider("短期均線", 1, 30, 5)
long_period = st.slider("長期均線", 30, 200, 60)
commission = st.slider('交易手續费 (%)', min_value=0.0, max_value=0.5, step=0.0005, format="%.4f", value=0.001)
trade_amount = st.slider("每次交易金额", min_value=0, max_value=50000, step=1000, value=1000)
initial_cash = st.slider("初始现金", min_value=0, max_value=10000000, step=10000, value=10000)

if st.button("開始回测"):
    lstm_model_ready = False
    trained_model = None
    
    # 執行訓練和回測
    train_lstm()
    run_backtrader()