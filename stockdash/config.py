from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
MODEL_PATH = ARTIFACT_DIR / "stock_classifier.joblib"
DB_PATH = DATA_DIR / "portfolio.sqlite3"
FEATURES = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "sma_ratio_10", "sma_ratio_20", "sma_ratio_50", "ema_ratio_12",
    "volume_change", "volume_ratio_20", "volatility_10", "volatility_20",
    "high_low_range", "close_open_return",
]
DEFAULT_TICKERS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "ORCL"],
    "Financials": ["JPM", "BAC", "GS", "V", "MA"],
    "Healthcare": ["JNJ", "LLY", "PFE", "UNH", "MRK"],
    "Consumer": ["AMZN", "WMT", "COST", "MCD", "KO"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "NEE"],
    "Industrials": ["CAT", "GE", "HON", "UPS", "BA"],
    "ETFs / broad market": ["SPY", "QQQ", "IWM", "DIA", "EFA"],
}
