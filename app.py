"""Mobile-friendly gold research dashboard. PAPER ONLY. No broker integration."""
import io
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from trader import fetch_prices, backtest, metrics, signals

st.set_page_config(page_title='Gold AutoTrader | Paper Only', page_icon='🟡', layout='centered')
st.markdown('''<style>div.block-container{padding-top:1.5rem;max-width:800px} .stMetric{background:#17212d10;padding:12px;border-radius:12px} </style>''',unsafe_allow_html=True)
st.title('🟡 Gold AutoTrader')
st.warning('SIMULATION ONLY · No real-money trading · No guarantee of profit')
st.caption('Daily gold ETC strategy research. Prices must be in GBP per unit; never upload prices in GBp without converting to GBP.')

with st.sidebar:
    st.header('Strategy controls')
    preset=st.selectbox('Preset', ['Cautious','Balanced','Active'], index=0)
    config={'Cautious':(.25,.05),'Balanced':(.35,.08),'Active':(.50,.10)}
    alloc,halt=config[preset]
    alloc=st.slider('Maximum position (% of account)',5,50,int(alloc*100),5)/100
    halt=st.slider('Drawdown pause (%)',3,15,int(halt*100),1)/100
    fee=st.number_input('Fixed commission per order (£)',min_value=0.0,max_value=10.0,value=0.0,step=0.5)
    spread=st.number_input('Estimated one-way trading cost (%)',min_value=0.0,max_value=5.0,value=0.2,step=0.1)/100
    st.caption('These presets change position size and loss limits, not the prediction algorithm. A drawdown halt cannot guarantee a loss ceiling.')

st.subheader('Market data')
source=st.radio('Data source',['Download research data','Upload historical CSV'],horizontal=True)
df=None
if source=='Upload historical CSV':
    f=st.file_uploader('Upload Date,Close CSV (GBP per unit)',type=['csv'])
    if f:
        try:
            raw=pd.read_csv(f)
            if not {'Date','Close'}.issubset(raw.columns): raise ValueError('CSV needs Date and Close columns')
            raw['Date']=pd.to_datetime(raw['Date'],errors='raise',utc=True).dt.tz_localize(None)
            raw['Close']=pd.to_numeric(raw['Close'],errors='raise')
            df=raw.set_index('Date')[['Close']].sort_index().dropna()
            df=df[~df.index.duplicated(keep='last')]
            if len(df)<220 or (df.Close<=0).any(): raise ValueError('Need 220+ valid daily rows with positive GBP prices')
        except Exception as e: st.error(f'Invalid data: {e}')
else:
    ticker=st.text_input('Research ticker (verify GBP price units)',value='SGLN.L')
    st.caption('External Yahoo Finance research feed may be delayed, unavailable, or quote London securities in GBp. Confirm units independently before using.')
    if st.button('Fetch historical prices',type='primary'):
        try:
            with st.spinner('Downloading research prices...'):
                st.session_state['prices']=fetch_prices(ticker=ticker,period='10y')
                st.session_state['ticker']=ticker
        except Exception as e: st.error(f'Data download failed: {e}')
    df=st.session_state.get('prices')
    if df is not None:
        st.info(f"Loaded {len(df)} sessions for {st.session_state.get('ticker','selected ticker')}. Verify price units before testing.")

if df is not None:
    st.line_chart(df['Close'],height=220)
    st.caption(f"Latest observation: {df.index[-1].date()} · {float(df.Close.iloc[-1]):.4f} in the feed's price units. This is not a live quote.")
    st.subheader('Historical backtest')
    if st.button('Run backtest',type='primary'):
        try:
            curve,trades=backtest(df,cash=100.0,cost=spread,fixed_fee=fee,max_allocation=alloc,max_drawdown=halt)
            m=metrics(curve,trades)
            benchmark=(df.Close/float(df.Close.iloc[0]))*100
            st.session_state['test']=(curve,trades,m,benchmark,{'preset':preset,'allocation':alloc,'halt':halt,'fee':fee,'spread':spread,'last_date':str(df.index[-1].date())})
        except Exception as e: st.error(f'Backtest failed: {e}')
    if 'test' in st.session_state:
        curve,trades,m,benchmark,settings=st.session_state['test']
        st.caption('Backtest results reflect the settings at the time of the last run: '+str(settings))
        a,b,c=st.columns(3)
        a.metric('Final simulated value',f"£{m['ending_gbp']:.2f}")
        b.metric('Strategy return',f"{m['total_return_pct']:+.2f}%")
        c.metric('Worst drawdown',f"{m['max_drawdown_pct']:.2f}%")
        st.metric('Number of orders',m['trade_count'])
        st.line_chart(pd.DataFrame({'Strategy (£)':curve.equity,'Buy & hold (£)':benchmark.reindex(curve.index)}),height=260)
        st.caption('Illustrative backtest only: assumes fractional units, next-close fills, constant costs and no taxes. Buy & hold benchmark excludes its trading costs. No out-of-sample validation.')
        st.subheader('Historical orders')
        if trades.empty: st.info('No trades were generated. Waiting is an acceptable result.')
        else: st.dataframe(trades.iloc[::-1],hide_index=True,use_container_width=True)
        st.download_button('Download backtest results (CSV)',curve.to_csv().encode(),file_name='gold_backtest_equity.csv',mime='text/csv')
    st.subheader('Paper trading preview')
    signal=signals(df)
    today=int(signal.signal.iloc[-1]); prior=int(signal.signal.iloc[-2])
    st.info(('Current closing signal: BUY eligible' if today else 'Current closing signal: WAIT / EXIT')+' · This signal is only considered at the next session, not traded immediately.')
    st.caption('Previous-session signal: '+('eligible for a simulated buy' if prior else 'no new buy')+'. This page does not run unattended. For scheduled paper trading, deploy the separate daily worker after verifying market data freshness.')
else:
    st.info('Load historical market data to begin. No example performance is invented.')

st.divider()
st.caption('Research tool, not financial advice. Do not use live trading until data licensing, price currency, commissions, order constraints, execution safeguards and forward tests have been verified.')
