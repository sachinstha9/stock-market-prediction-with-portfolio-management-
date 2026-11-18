from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from stockdash.config import DEFAULT_TICKERS, FEATURES, MODEL_PATH, DB_PATH
from stockdash.portfolio import PortfolioDB
from stockdash.data import download_history, add_features, make_supervised
from stockdash.modeling import load_model, train_and_save, predict_latest, model_candidates

st.set_page_config(page_title='MarketLab | Stock Intelligence', page_icon='📈', layout='wide')
st.markdown('''<style>
/* MarketLab — restrained editorial layout */
.block-container {
    max-width: 1380px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}
.hero {
    position: relative;
    padding: 1.25rem 1.5rem 1.15rem;
    margin-bottom: 1.6rem;
    background: rgba(120, 132, 112, .075);
    border: 1px solid rgba(120, 132, 112, .22);
    border-left: 4px solid #788774;
    border-radius: 5px;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 650;
    letter-spacing: -0.7px;
    margin: 0 0 .3rem 0;
}
.hero p {
    margin: 0 0 .65rem 0;
    color: var(--text-color);
    opacity: .78;
    font-size: .98rem;
}
.small-muted {
    color: var(--text-color);
    opacity: .62;
    font-size: .82rem;
}
h2, h3 {
    letter-spacing: -.25px;
}
div[data-testid="stMetric"] {
    padding: .8rem .9rem;
    background: rgba(128, 128, 128, .045);
    border: 1px solid rgba(128, 128, 128, .18);
    border-radius: 6px;
}
div[data-testid="stMetricLabel"] {
    font-size: .82rem;
}
div[data-testid="stMetricValue"] {
    font-weight: 620;
}
div[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, .2);
}
div[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.4rem;
}
div[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: .28rem .45rem;
    border-radius: 4px;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 5px;
    min-height: 2.45rem;
    font-weight: 550;
}
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(128, 128, 128, .2);
    border-radius: 6px;
    overflow: hidden;
}
hr {
    margin: 1.35rem 0;
    border-color: rgba(128, 128, 128, .2);
}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1> MarketLab</h1><p>Multi-stock classification </p></div>',unsafe_allow_html=True)

db=PortfolioDB(DB_PATH)
@st.cache_data(ttl=900,show_spinner=False)

def get_prices(tickers,period='2y'):
    return download_history(list(tickers),period=period)

def money(v):
    return f"${v:,.2f}"
 
def safe_prices(ticker,period='2y'):
    return get_prices((ticker,),period)[ticker]

metadata=load_model()
page=st.sidebar.radio('Workspace',['Market overview','Stock analysis','Model lab','Portfolio','Backtesting','Prediction history'])
st.sidebar.caption(f"Saved model: {'available' if metadata else 'not trained yet'}")

if page=='Market overview':
    st.subheader('Market overview')

    tickers=[t for group in DEFAULT_TICKERS.values() for t in group]
    try:
        prices=get_prices(tuple(tickers),'6mo')
        cols=st.columns(4)
        for i,t in enumerate(list(prices)[:4]):
            p=prices[t]; change=(p.Close.iloc[-1]/p.Close.iloc[-2]-1)*100
            cols[i].metric(t,money(p.Close.iloc[-1]),f'{change:+.2f}%')
        selected=st.selectbox('Explore symbol',sorted(prices))
        p=prices[selected]
        fig=go.Figure(go.Candlestick(x=p.index,open=p.Open,high=p.High,low=p.Low,close=p.Close,name=selected))
        fig.update_layout(height=460,xaxis_rangeslider_visible=False,template='plotly_dark',margin=dict(l=10,r=10,t=25,b=10))
        st.plotly_chart(fig,width='stretch')
        st.caption('Quotes may be delayed, missing, adjusted, or unavailable for some symbols.')
    except Exception as e: st.error(f'Could not load market data: {e}')

    st.divider(); st.subheader('Watchlist')
    a,b=st.columns([3,1]); new=a.text_input('Add ticker',placeholder='e.g. NVDA').upper().strip()
    if b.button('Add to watchlist') and new: db.add_watch(new); st.rerun()
    watch=db.watchlist()
    if watch:
        try:
            wp=get_prices(tuple(watch),'6mo')
            st.dataframe(pd.DataFrame([{'Ticker':t,'Last close':float(v.Close.iloc[-1]),'1D %':float(v.Close.pct_change().iloc[-1]*100)} for t,v in wp.items()]),use_container_width=True,hide_index=True)
        except Exception as e: st.warning(str(e))
        remove=st.selectbox('Remove symbol',watch)
        if st.button('Remove'): db.remove_watch(remove); st.rerun()
    else: st.info('Add symbols above to create a persistent watchlist.')

elif page=='Stock analysis':
    st.subheader('Stock analysis & prediction')
    ticker=st.text_input('Ticker symbol','AAPL').upper().strip()
    horizon=st.select_slider('Prediction horizon (trading days)',options=[1,5,10,20],value=5)
    if st.button('Analyze',type='primary'):
        try:
            prices=safe_prices(ticker,'10y')
            features=add_features(prices)
            fig=go.Figure(); fig.add_trace(go.Scatter(x=prices.index,y=prices.Close,name='Adjusted close'))
            for n in [20,50,200]: fig.add_trace(go.Scatter(x=prices.index,y=prices.Close.rolling(n).mean(),name=f'SMA {n}'))
            fig.update_layout(template='plotly_dark',height=420)
            st.plotly_chart(fig,use_container_width=True)
            c1,c2,c3=st.columns(3); c1.metric('Last close',money(prices.Close.iloc[-1])); c2.metric('1D return',f"{prices.Close.pct_change().iloc[-1]*100:+.2f}%"); c3.metric('Annualized 20D vol',f"{features.volatility_20.iloc[-1]*100:.1f}%")
            if metadata:
                if metadata['horizon']!=horizon: st.warning(f"Saved model predicts {metadata['horizon']} trading days ahead, not {horizon}. Train a matching horizon in Model lab.")
                else:
                    pred=predict_latest(metadata,prices)
                    label={1:'UP',0:'DOWN',-1:'DOWN',}.get(pred['predicted_class'],'SIDEWAYS')
                    st.markdown(f'### Model estimate: **{label}**')
                    st.caption(f"As of {pd.Timestamp(pred['date']).date()} close {money(pred['close'])}; class probabilities are model estimates.")
                    st.bar_chart(pd.DataFrame({'Class':[('UP' if k=='1' else 'DOWN' if k in ('0','-1') else 'SIDEWAYS') for k in pred['probabilities']], 'Probability':list(pred['probabilities'].values())}).set_index('Class'))
                    db.record_prediction(ticker,horizon,pred['predicted_class'],pred['probabilities'])
            else: st.info('No saved model exists yet. Open Model lab and train on real historical data.')
            with st.expander('Latest technical features'): st.dataframe(features[FEATURES].tail(5),use_container_width=True)
        except Exception as e: st.error(f'Analysis failed: {e}')

elif page=='Model lab':
    st.subheader('Model training & validation')
    st.write('Train and compare classical machine-learning models and a feed-forward deep-learning neural network. Models are evaluated with chronological walk-forward folds; training downloads real data and may take several minutes.')
    horizon=st.selectbox('Target horizon',[1,5,10,20],index=1)
    mode=st.radio('Target classes',['UP/DOWN','UP/DOWN/SIDEWAYS'],horizontal=True)
    period=st.selectbox('History period',['5y','10y'],index=1)
    model_options=list(model_candidates().keys())
    selection=st.selectbox(
        'Model selection',
        ['Automatic (compare all models)'] + model_options,
        help='Automatic compares every model using chronological walk-forward validation. Choose one model to train and validate only that option.'
    )

    threshold=st.slider('Minimum walk-forward balanced accuracy required to replace saved model',0.35,0.70,0.48,0.01)
    if st.button('Download, train & validate',type='primary'):
        with st.spinner('Downloading market history and evaluating models...'):
            tickers=[t for group in DEFAULT_TICKERS.values() for t in group]
            raw=download_history(tickers,period=period
            dataset=make_supervised(raw,horizon,mode)

            st.write(f"Usable symbols: {len(raw)} · supervised rows: {len(dataset):,} · dates: {dataset.index.nunique():,}")

            chosen_models = None if selection.startswith('Automatic') else [selection]

            result=train_and_save(
                dataset, horizon, mode, threshold, MODEL_PATH,
                selected_models=chosen_models
            )

            st.dataframe(result['scores'].drop(columns=['fold_details']),use_container_width=True,hide_index=True)

            st.info(result['reason'])

            if result['saved']: 
                st.success(f"Saved model: {result['model_name']}")
                st.rerun()
            
    if metadata:
        st.markdown('Current saved artifact')
        st.json({k:v for k,v in metadata.items() if k not in ('model','features')})
        st.caption(f'Artifact path: {MODEL_PATH}')
    else: 
        st.warning('No model artifact detected. predictions remain disabled until a real training run succeeds.')

elif page == "Portfolio":
    st.subheader('Virtual portfolio')

    cash = db.cash()
    tx = db.transactions()
    holdings = db.holdings()

    c1, c2, c3 = st.columns(3)
    c1.metric('Virtual cash', money(cash))
    c2.metric('Open positions', len(holdings))
    c3.metric('Transactions', len(tx))

    with st.expander('Record buy / sell transaction', expanded=True):
        with st.form('trade_form'):
            ticker = st.text_input('Ticker', 'AAPL').upper().strip()
            side = st.radio('Side', ['BUY', 'SELL'], horizontal=True)
            qty = st.number_input('Shares', min_value=0.0001, value=1.0,
                                step=1.0, format='%.4f')
            price = st.number_input('Execution price', min_value=0.01,
                                    value=100.0, step=1.0)
            fees = st.number_input('Fees', min_value=0.0, value=0.0, step=0.5)

            submit = st.form_submit_button('Record transaction')

        if submit:
            try:
                db.trade(ticker, side, qty, price, fees)
                st.success('Transaction recorded.')
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if not holdings.empty:
        try:
            live = get_prices(tuple(holdings.ticker), '5d')

            holdings['last_price'] = holdings.ticker.map(
                {t: float(p.Close.iloc[-1]) for t, p in live.items()}
            )

            holdings['market_value'] = holdings.quantity * holdings.last_price
            holdings['unrealized_pnl'] = holdings.market_value - holdings.cost_basis
            holdings['allocation_%'] = (
                holdings.market_value / holdings.market_value.sum() * 100
            )

            st.markdown('#### Holdings')
            st.dataframe(holdings, use_container_width=True, hide_index=True)

            st.metric('Unrealised P&L', money(holdings.unrealized_pnl.sum()))

            fig = go.Figure(
                go.Pie(
                    labels=holdings.ticker,
                    values=holdings.market_value,
                    hole=0.55
                )
            )
            fig.update_layout(template='plotly_dark')

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f'Live holdings valuation unavailable: {e}')

    st.markdown('#### Transaction ledger')
    st.dataframe(tx, use_container_width=True, hide_index=True)

    st.caption(
        'Virtual account only. Realised P&L is an estimate based on recorded '
        'trades and does not include tax or corporate-action handling.'
    )

elif page=='Backtesting':
    st.subheader('Out-of-sample backtesting')
    st.warning('Simplified long/flat strategy; results depend on data quality and assumptions. No live trading.')

    ticker = st.text_input('Ticker', 'SPY').upper().strip()

    if st.button('Run backtest', type='primary'):

        if not metadata:
            st.error('Train a model first.')

        else:
            try:
                prices = safe_prices(ticker, '5y')

                if metadata['horizon'] != 1:
                    st.error('This daily execution backtest requires a 1-day model. Train horizon=1 for this page.')

                elif not metadata.get('oos_predictions'):
                    st.error('This artifact has no saved out-of-sample predictions. Retrain to create an OOS validation ledger.')

                else:
                    oos = pd.DataFrame(metadata['oos_predictions'])
                    oos = oos[oos.ticker == ticker].copy()

                    if oos.empty:
                        st.error(f'No saved out-of-sample predictions for {ticker}. Train with this ticker included.')

                    else:
                        oos['date'] = pd.to_datetime(oos.date).dt.tz_localize(None)
                        oos = oos[oos.date.isin(prices.index)]

                        if oos.empty:
                            st.error('Saved out-of-sample dates do not overlap downloaded price history.')

                        else:
                            prices = prices.loc[oos.date.min():oos.date.max()]
                            signals = pd.Series(0, index=prices.index, dtype=int)
                            pred_by_date = oos.drop_duplicates('date').set_index('date')
                            common_dates = signals.index.intersection(pred_by_date.index)

                            signals.loc[common_dates] = (
                                pred_by_date.loc[common_dates, 'predicted']
                                .astype(int).eq(1).astype(int).values
                            )

                            result, metrics = backtest_signals(prices, signals)

                            c1, c2, c3 = st.columns(3)

                            c1.metric(
                                'OOS strategy return',
                                f"{metrics['strategy']['total_return'] * 100:.2f}%"
                            )

                            c2.metric(
                                'Max drawdown',
                                f"{metrics['strategy']['max_drawdown'] * 100:.2f}%"
                            )

                            c3.metric(
                                'Sharpe',
                                f"{metrics['strategy']['sharpe']:.2f}"
                            )

                            st.write(
                                'Buy-and-hold comparison over the same OOS date range',
                                metrics['buy_hold']
                            )

                            st.caption(
                                'Uses saved walk-forward out-of-fold classifications. '
                                'Signal at close, execution at next close, exposure begins '
                                'after execution; costs/slippage are approximated. '
                                'Probability threshold control is not applied to this '
                                'OOS class-only ledger.'
                            )

                            fig = go.Figure()

                            fig.add_trace(go.Scatter(
                                x=result.index,
                                y=result.equity,
                                name='OOS strategy'
                            ))

                            fig.add_trace(go.Scatter(
                                x=result.index,
                                y=result.buy_hold,
                                name='Buy & hold'
                            ))

                            fig.update_layout(
                                template='plotly_dark',
                                height=420
                            )

                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(result.tail(20), use_container_width=True)

            except Exception as e:
                st.error(f'Backtest failed: {e}')
elif page=='Prediction history':
    st.subheader('Prediction history')

    hist=db.predictions()

    if hist.empty:
          st.info('No predictions recorded yet. Run a stock analysis with a saved model.')
    else:
        st.dataframe(hist,use_container_width=True,hide_index=True)
        st.caption('Actual outcomes are not auto-filled: evaluation requires waiting until the selected forecast horizon has elapsed and matching the corresponding market data.')
