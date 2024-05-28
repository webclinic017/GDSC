import streamlit as st
import pandas as pd
import yfinance as yf
import backtrader as bt
import datetime
import matplotlib.pyplot as plt
from prophet import Prophet
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import requests
from io import BytesIO

# Prophet 預測函數
def predict_stock(selected_stock, n_years):
    data = yf.download(selected_stock, start="2010-01-01", end=datetime.date.today().strftime("%Y-%m-%d"))
    data.reset_index(inplace=True)

    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=n_years * 365)
    forecast = m.predict(future)

    return data, forecast, m

class PeriodicInvestmentStrategy(bt.Strategy):
    params = (
        ('monthly_investment', None),  # 每期投資金額
        ('commission', None),  # 手續費
        ('investment_day', None),  # 投資日
        ('printlog', True),  # 是否打印交易日誌
    )

    def __init__(self, **kwargs):
        self.order = None
        self.add_timer(
            when=bt.Timer.SESSION_START,
            monthdays=[self.params.investment_day],  # 每月的特定日期投資
            monthcarry=True,  # 如果特定日期不是交易日，則延至下一個交易日
        )

        # 從kwargs中獲取初始資金
        self.initial_cash = kwargs.get('initial_cash', 10000)  # 初始資金設置為10000

    def notify_timer(self, timer, when, *args, **kwargs):
        self.log('進行定期投資')
        # 獲取當前價格
        price = self.data.close[0]
        # 計算購買數量
        investment_amount = self.params.monthly_investment / price
        # 執行購買
        self.order = self.buy(size=investment_amount)

    def log(self, txt, dt=None):
        ''' 日誌函數 '''
        dt = dt or self.datas[0].datetime.date(0)
        if self.params.printlog:
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                cost = order.executed.price * order.executed.size
                commission = cost * self.params.commission / 100  # 將百分比轉換為小數
                self.log('買入執行, 價格: %.2f, 成本: %.2f, 手續費: %.2f' %
                        (order.executed.price, cost, commission))

            elif order.issell():
                self.log('賣出執行, 價格: %.2f, 成本: %.2f, 手續費: %.2f' %
                        (order.executed.price,
                        order.executed.value,
                        order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('訂單 取消/保證金不足/拒絕')

        self.order = None

# 從 GitHub 加載圖片
image_url = 'https://raw.githubusercontent.com/j7808833/test_02/main/pic/Cyberpunk_bar_03.jpg'
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# 顯示圖片
st.image(image, use_column_width=True)

# Streamlit 頁面佈局
st.title('Backtest&Backtrader  Bar')

# 提示用戶輸入股票代碼，並使用逗號分隔
user_input = st.text_area("請輸入股票代碼，用逗號分隔，台股請記得在最後加上.TW", "AAPL, MSFT, GOOG, AMZN, 0050.TW")
# 將用戶輸入的股票代碼轉換為列表
stocks = [stock.strip() for stock in user_input.split(",")]
st.write("您輸入的股票代碼：", stocks)

# 股票選擇器和預測年限滑塊
selected_stock = st.selectbox('選擇股票進行預測和回測', stocks)
n_years = st.slider('預測年限:', 1, 3)

# 預測和顯示結果
if st.button('運行預測'):
    # 做預測並獲取數據、預測結果和 Prophet 模型
    data, forecast, m = predict_stock(selected_stock, n_years)
    st.write('預測數據:')
    st.write(forecast)

    st.write(f'{n_years} 年的預測圖')
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

# 添加滑塊來控制參數
initial_cash = st.slider('預算', min_value=0, max_value=10000000, step=10000, value=10000)
monthly_investment = st.slider('每月投資金額', min_value=0, max_value=50000, step=1000, value=1000)
commission = st.slider('手續費 (%)', min_value=0.0, max_value=1.0, step=0.0001, format="%.4f", value=0.001)
investment_day = st.slider('每月投資日', min_value=1, max_value=28, step=1, value=1)
n_years_backtest = st.slider('回測持續時間 (年)', min_value=1, max_value=10, step=1, value=5)

# 定義顯示結果的函數
def display_results(cash, value, initial_value, n_years):
    st.write(f"預算: ${initial_cash:.2f}")
    st.write(f"最終價值: ${value:.2f}")

    # 計算年回報率
    annual_return = ((value - cash) / (initial_cash - cash)) ** (1 / n_years) - 1
    annual_return *= 100  # 轉換為百分比形式
    st.write(f"年回報率: {annual_return:.2f}%")
    return annual_return

def get_drink_name(investment_ratio, commission, annual_return):
    if investment_ratio > 0.1:
        if commission < 0.15:
            if annual_return <= 2:
                return "Vodka_Soda"
            elif annual_return <= 5:
                return "Vodka_Martini"
            elif annual_return <= 10:
                return "Whiskey_Sour"
            else:
                return "Whiskey_Neat"
        else:
            if annual_return <= 2:
                return "Moscow_Mule"
            elif annual_return <= 5:
                return "Bloody_Mary"
            elif annual_return <= 10:
                return "Old_Fashioned"
            else:
                return "Manhattan"
    else:
        if commission < 0.15:
            if annual_return <= 2:
                return "Screwdriver"
            elif annual_return <= 5:
                return "Vodka_Collins"
            elif annual_return <= 10:
                return "Rob_Roy"
            else:
                return "Sazerac"
        else:
            if annual_return <= 2:
                return "Aperol_Spritz"
            elif annual_return <= 5:
                return "Cosmopolitan"
            elif annual_return <= 10:
                return "Boulevardier"
            else:
                return "Vieux_Carré"

# 定義調酒名稱和其對應的特性和依據
drinks_info = {
    "Vodka_Soda": {
        "報酬率": "低",
        "大小": "小額",
        "特性": "伏特加和蘇打水，酒精度低，口感清淡清爽。",
        "依據": "低風險，適合低回報的小額短期投資。"
    },
    "Vodka_Martini": {
        "報酬率": "中",
        "大小": "小額",
        "特性": "伏特加和乾苦艾酒，酒精度中等，口感適中，經典且稍微複雜。",
        "依據": "適合中等風險和回報的小額短期投資。"
    },
    "Whiskey_Sour": {
        "報酬率": "高",
        "大小": "小額",
        "特性": "威士忌、檸檬汁和糖漿，酒精度高，口感濃烈且有層次。",
        "依據": "對應高風險和高回報的小額短期投資。"
    },
    "Whiskey_Neat": {
        "報酬率": "極高",
        "大小": "小額",
        "特性": "純飲威士忌，酒精度非常高，口感非常濃烈直接。",
        "依據": "對應極高風險和極高回報的小額短期投資。"
    },
    "Moscow_Mule": {
        "報酬率": "低",
        "大小": "大額",
        "特性": "伏特加、薑汁啤酒和青檸汁，酒精度低，口感溫和，帶有薑味的清爽感。",
        "依據": "適合低風險且低回報的大額短期投資。"
    },
    "Bloody_Mary": {
        "報酬率": "中",
        "大小": "大額",
        "特性": "伏特加、番茄汁和各種調味料，酒精度中等，口感豐富且略帶鹹味。",
        "依據": "適合中等風險和回報的大額短期投資。"
    },
    "Old_Fashioned": {
        "報酬率": "高",
        "大小": "大額",
        "特性": "威士忌、苦味酒和糖，酒精度高，口感濃烈且複雜。",
        "依據": "適合高風險和高回報的大額短期投資。"
    },
    "Manhattan": {
        "報酬率": "極高",
        "大小": "大額",
        "特性": "威士忌、甜苦艾酒和苦味酒，酒精度非常高，口感非常濃烈複雜且富有層次。",
        "依據": "適合極高風險和極高回報的大額短期投資。"
    },
    "Screwdriver": {
        "報酬率": "低",
        "大小": "小額",
        "特性": "伏特加和橙汁，酒精度低，口感清新簡單。",
        "依據": "適合低風險低回報的小額長期投資。"
    },
    "Vodka_Collins": {
        "報酬率": "中",
        "大小": "小額",
        "特性": "伏特加、檸檬汁、糖漿和蘇打水，酒精度中等，口感清爽且略帶甜味。",
        "依據": "適合中等風險和回報的小額長期投資。"
    },
    "Rob_Roy": {
        "報酬率": "高",
        "大小": "小額",
        "特性": "威士忌、甜苦艾酒和苦味酒，酒精度高，口感濃烈且經典。",
        "依據": "適合高風險和高回報的小額長期投資。"
    },
    "Sazerac": {
        "報酬率": "極高",
        "大小": "小額",
        "特性": "威士忌、苦艾酒和苦味酒，酒精度非常高，口感非常濃烈複雜。",
        "依據": "適合極高風險和極高回報的小額長期投資。"
    },
    "Aperol_Spritz": {
        "報酬率": "低",
        "大小": "大額",
        "特性": "Aperol、蘇打水和香檳，酒精度低，口感溫和且清爽。",
        "依據": "適合低風險低回報的大額長期投資。"
    },
    "Cosmopolitan": {
        "報酬率": "中",
        "大小": "大額",
        "特性": "伏特加、柑橘利口酒、蔓越莓汁和青檸汁，酒精度中等，口感適中且帶有水果味。",
        "依據": "適合中等風險和回報的大額長期投資。"
    },
    "Boulevardier": {
        "報酬率": "高",
        "大小": "大額",
        "特性": "威士忌、甜苦艾酒和苦味酒，酒精度高，口感濃烈且複雜。",
        "依據": "適合高風險和高回報的大額長期投資。"
    },
    "Vieux_Carré": {
        "報酬率": "極高",
        "大小": "大額",
        "特性": "威士忌、干邑、甜苦艾酒和苦味酒，酒精度非常高，口感非常濃烈複雜。",
        "依據": "適合極高風險和極高回報的大額長期投資。"
    }
}

# 执行回测并显示结果
if st.button('Run Backtest'):
    # 初始化 Cerebro 引擎
    cerebro = bt.Cerebro()
    cerebro.addstrategy(PeriodicInvestmentStrategy, initial_cash=initial_cash, monthly_investment=monthly_investment, commission=commission, investment_day=investment_day)

    # 添加数据
    start_date = datetime.datetime.now() - relativedelta(years=n_years_backtest)  # 根据回测年限动态计算开始时间
    data = yf.download(selected_stock,
                    start=start_date,
                    end=datetime.datetime.now())
    cerebro.adddata(bt.feeds.PandasData(dataname=data))

    # 设置初始资本
    cerebro.broker.setcash(initial_cash)

    # 设置每笔交易的手续费
    cerebro.broker.setcommission(commission=commission)

    # 执行策略
    cerebro.run()

    # 获取初始总价值
    initial_value = cerebro.broker.get_value()

    # 获取当前现金余额和总价值
    cash = cerebro.broker.get_cash()
    value = cerebro.broker.get_value()

    # 显示结果
    display_results(cash, value, initial_value, n_years_backtest)

    # 绘制结果
    fig = cerebro.plot(style='plotly')[0][0]  # 获取 Matplotlib 图形对象
    st.pyplot(fig)  # 将图形嵌入到 Streamlit 页面中

    # 計算投資比例
    investment_ratio = monthly_investment / initial_cash if initial_cash != 0 else float('inf')

    # 計算年化回報率
    annual_return = ((value - initial_cash) / initial_cash + 1) ** (1 / n_years_backtest) - 1
    annual_return *= 100  # 轉換為百分比形式

    # 根據投資參數查找對應的調酒名稱
    drink_name = get_drink_name(investment_ratio, commission, annual_return)
        
    # 調酒圖片 URL 字典
    drink_images = {
        "Vodka_Soda": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_01_Vodka%20Soda.jpg",
        "Vodka_Martini": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_02_Vodka%20Martini.jpg",
        "Whiskey_Sour": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_03_Whiskey%20Sour.jpg",
        "Whiskey_Neat": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_04_Whiskey%20Neat.jpg",
        "Moscow_Mule": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_05_Moscow%20Mule.jpg",
        "Bloody_Mary": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_06_Bloody%20Mary.jpg",
        "Old_Fashioned": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_07_Old%20Fashioned.jpg",
        "Manhattan": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_08_Manhattan.jpg",
        "Screwdriver": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_09_Screwdriver.jpg",
        "Vodka_Collins": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_10_Vodka%20Collins.jpg",
        "Rob_Roy": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_11_Rob%20Roy.jpg",
        "Sazerac": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_12_Sazerac.jpg",
        "Aperol_Spritz": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_13_Aperol%20Spritz.jpg",
        "Cosmopolitan": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_14_Cosmopolitan.jpg",
        "Boulevardier": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_15_Boulevardier.jpg",
        "Vieux_Carré": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_16_Vieux%20Carr%C3%A9.jpg"
    }

    st.write(f"您的投資風格對應的調酒是: {drink_name}")

    # 顯示調酒圖片
    image_url = drink_images[drink_name]
    response = requests.get(image_url)
    drink_image = Image.open(BytesIO(response.content))
    st.image(drink_image, caption=drink_name, use_column_width=True)

    # 顯示特性和依據
    if drink_name in drinks_info:
        st.write("特性：", drinks_info[drink_name]["特性"])
        st.write("依據：", drinks_info[drink_name]["依據"])
    else:
        st.write("找不到對應的調酒信息。")
