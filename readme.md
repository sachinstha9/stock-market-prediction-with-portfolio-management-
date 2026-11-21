# Stock Market Prediction and Portfolio Management

## About the Project

This project is a stock market prediction and portfolio management application developed using Python, Machine Learning and Deep Learning.

The main aim of this project is to analyse historical stock market data, predict possible future stock movements, and help users understand and manage their investment portfolios.

The application uses real stock market data from Yahoo Finance and provides an interactive dashboard where users can explore stocks, train different prediction models, view predictions, manage a virtual portfolio, and test trading strategies.

## Features

### 1. Market Overview

- View stock market information.
- Check the latest available stock prices.
- View daily price changes.
- Explore stock price charts.
- Create and manage a personal watchlist.

### 2. Stock Analysis

- Search for stocks using their ticker symbols.
- View historical stock prices.
- Display moving averages (SMA 20, SMA 50 and SMA 200).
- Calculate technical indicators.
- View predicted stock direction.
- Check prediction probabilities.
- Analyse stock price movements using historical data.

### 3. Machine Learning and Deep Learning Models

The project includes five different models for stock market prediction. These models are used to analyse historical stock data and predict future stock movements.

**Machine Learning Models:**

- Logistic Regression
- Random Forest
- Gradient Boosting

**Deep Learning Models:**

- MLP (Multi-Layer Perceptron)
- LSTM (Long Short-Term Memory)

The MLP model is used to learn patterns from stock market features, while the LSTM model is designed to learn patterns from sequential stock market data.

Users can select different models, train them and compare their prediction performance.

The application also supports different prediction periods:

- 1 trading day
- 5 trading days
- 10 trading days
- 20 trading days

It supports two prediction modes:

- UP/DOWN
- UP/DOWN/SIDEWAYS

### 4. Portfolio Management

- Manage a virtual investment portfolio.
- Record buy and sell transactions.
- Track available virtual cash.
- View current stock holdings.
- Calculate estimated profit and loss.
- View portfolio allocation.
- Track investment performance.

### 5. Backtesting

- Test trading strategies using historical stock data.
- Compare strategy performance with a buy-and-hold approach.
- View total returns.
- Check maximum drawdown.
- Calculate the Sharpe ratio.
- View portfolio performance charts.

### 6. Prediction History

- Save previous stock predictions.
- View prediction records.
- Check prediction dates.
- Review selected models and forecast periods.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application and dashboard |
| Pandas | Data processing |
| NumPy | Numerical calculations |
| Scikit-learn | Machine learning models |
| TensorFlow / Keras | Building and training MLP and LSTM models |
| Yahoo Finance (yfinance) | Downloading stock market data |
| Plotly | Interactive charts |
| SQLite | Portfolio and prediction data storage |
| Joblib | Saving and loading trained models |

## Project Structure

```text
stock-market-prediction-with-portfolio-management/
│
├── artifacts/
│
├── data/
│
├── stockdash/
│   ├── config.py
│   ├── data.py
│   ├── modeling.py
│   ├── portfolio.py
│   └── backtest.py
│
├── tests/
│
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

### Folder and File Description

- `app.py` - Main Streamlit application.
- `train.py` - Script for training and validating prediction models.
- `stockdash/` - Contains the main project modules.
- `config.py` - Project configuration and stock settings.
- `data.py` - Downloads stock data and prepares features for prediction.
- `modeling.py` - Handles machine learning and deep learning model training and prediction.
- `portfolio.py` - Handles portfolio transactions and prediction history.
- `backtest.py` - Contains backtesting calculations and performance metrics.
- `data/` - Stores local application data.
- `artifacts/` - Stores trained models and related files.
- `tests/` - Contains project tests.
- `requirements.txt` - Contains the required Python libraries.

## Installation and Setup

Follow these steps to run the project on your computer.

### 1. Clone the Repository

```bash
git clone https://github.com/sachinstha9/stock-market-prediction-with-portfolio-management-.git
```

### 2. Open the Project Folder

```bash
cd stock-market-prediction-with-portfolio-management-
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**For Windows:**

```bash
venv\Scripts\activate
```

**For Linux and macOS:**

```bash
source venv/bin/activate
```

### 4. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser. If it does not open automatically, visit:

```text
http://localhost:8501
```

## How to Use the Application

1. Open the application.
2. Visit the Market Overview page to explore stock prices.
3. Go to Stock Analysis and enter a stock ticker.
4. Select a prediction period and analyse the stock.
5. Open Model Lab to select and train different machine learning or deep learning models.
6. Compare the performance of the available models.
7. Visit the Portfolio page to record virtual buy and sell transactions.
8. Use Backtesting to test trading strategies.
9. Visit Prediction History to check previously saved predictions.

## Data Source

This project uses Yahoo Finance through the `yfinance` Python library to download historical stock market data.

The application uses information such as:

- Open price
- High price
- Low price
- Close price
- Trading volume

It also calculates technical indicators such as:

- RSI
- MACD
- Moving averages
- Returns
- Volatility

These features are used by the prediction models to analyse stock market movements.

## Project Team

**Rehan Ahmed**

**Sachin Shrestha**

## Disclaimer

This project is developed for educational purposes as part of an academic project.

Stock market predictions are not always accurate because market prices can change due to many factors. The predictions and backtesting results should not be considered financial advice.

The portfolio feature is a virtual portfolio and does not perform real stock market transactions.