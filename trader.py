"""Gold AutoTrader v1: research, backtest and persistent paper trading ONLY.
No live broker interface or real-money order placement exists in this project.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_TICKER = 'SGLN.L'  # Example LSE gold ETC; verify currency, fees and eligibility.
STARTING_CASH = 100.0
MAX_ALLOCATION = 0.25
COST_RATE = 0.002  # illustrative one-way spread/slippage + percentage costs
FIXED_FEE = 0.0  # set to actual broker minimum commission before interpreting results
MAX_DRAWDOWN = 0.10


def fetch_prices(ticker=DEFAULT_TICKER, period='10y', csv=None):
    if csv:
        df = pd.read_csv(csv)
        if 'Date' not in df or 'Close' not in df:
            raise ValueError('CSV must contain Date and Close columns')
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        df = df.set_index('Date')
    else:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval='1d', auto_adjust=True,
                         progress=False, multi_level_index=False)
        if df.empty:
            raise ValueError('No data returned. Try --csv with Date,Close columns.')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    df = df[['Close']].copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep='last')]
    if len(df) < 220:
        raise ValueError('Need at least 220 trading-day observations.')
    if (df.Close <= 0).any():
        raise ValueError('Invalid nonpositive prices.')
    return df


def signals(df):
    out = df.copy()
    out['sma50'] = out.Close.rolling(50).mean()
    out['sma200'] = out.Close.rolling(200).mean()
    out['momentum63'] = out.Close.pct_change(63)
    out['signal'] = ((out.sma50 > out.sma200) &
                     (out.Close > out.sma200) &
                     (out.momentum63 > 0)).astype(int)
    return out


def backtest(df, cash=STARTING_CASH, cost=COST_RATE, fixed_fee=FIXED_FEE,
             max_allocation=MAX_ALLOCATION, max_drawdown=MAX_DRAWDOWN):
    """Signal from yesterday's CLOSE; trade at today's CLOSE as a next-bar proxy.
    No overnight lookahead, but next-close execution is only an approximation.
    Fractional shares assumed; verify broker support before live use.
    """
    data = signals(df)
    shares, peak, stopped = 0.0, cash, False
    rows, trades = [], []
    previous_signal = 0
    for date, row in data.iterrows():
        price = float(row.Close)
        equity = cash + shares * price
        peak = max(peak, equity)
        if equity <= peak * (1-max_drawdown):
            stopped = True
        target = int(previous_signal) if not stopped else 0
        if target == 0 and shares > 0:
            proceeds = shares * price * (1-cost) - fixed_fee
            if proceeds >= 0:
                cash += proceeds
                trades.append({'date': str(date.date()), 'side': 'SELL', 'price': price,
                               'shares': shares, 'fee_and_slippage': shares*price*cost+fixed_fee})
                shares = 0.0
        elif target == 1 and shares == 0:
            equity = cash
            spend = min(cash, equity*max_allocation)
            if spend > fixed_fee:
                qty = (spend-fixed_fee)/(price*(1+cost))
                if qty > 0:
                    cash -= qty*price*(1+cost)+fixed_fee
                    shares = qty
                    trades.append({'date': str(date.date()), 'side': 'BUY', 'price': price,
                                   'shares': shares, 'fee_and_slippage': shares*price*cost+fixed_fee})
        equity = cash + shares*price
        rows.append({'Date': date, 'Close': price, 'signal_from_prior_day': target,
                     'cash': cash, 'shares': shares, 'equity': equity, 'halted': stopped})
        previous_signal = int(row.signal)
    result = pd.DataFrame(rows).set_index('Date')
    result['peak'] = result.equity.cummax()
    result['drawdown'] = result.equity/result.peak-1
    return result, pd.DataFrame(trades)


def metrics(curve, trades, start=STARTING_CASH):
    years = max((curve.index[-1]-curve.index[0]).days/365.25, 1/365.25)
    daily = curve.equity.pct_change().dropna()
    return {'starting_gbp': round(start,2), 'ending_gbp': round(float(curve.equity.iloc[-1]),2),
            'total_return_pct': round((curve.equity.iloc[-1]/start-1)*100,2),
            'annualised_return_pct': round(((curve.equity.iloc[-1]/start)**(1/years)-1)*100,2),
            'max_drawdown_pct': round(float(curve.drawdown.min()*100),2),
            'sharpe_approx': round(float(np.sqrt(252)*daily.mean()/daily.std()),2) if daily.std()>0 else None,
            'trade_count': len(trades), 'halted': bool(curve.halted.iloc[-1])}


def paper_step(df, state_path, cost=COST_RATE):
    """Idempotent daily paper-trade step; never places a real order."""
    data = signals(df)
    date = str(data.index[-1].date())
    if len(data) < 2:
        raise ValueError('Not enough data')
    price = float(data.Close.iloc[-1])
    # Today's signal is only usable for next session; previous close drives today's order.
    target = int(data.signal.iloc[-2])
    path = Path(state_path)
    state = json.loads(path.read_text()) if path.exists() else {
        'cash': STARTING_CASH, 'shares': 0.0, 'peak': STARTING_CASH,
        'halted': False, 'last_date': None, 'trades': []}
    if state['last_date'] is not None and date <= state['last_date']:
        return state
    equity = state['cash']+state['shares']*price
    state['peak'] = max(state['peak'], equity)
    if equity <= state['peak']*(1-MAX_DRAWDOWN):
        state['halted'] = True
    if state['halted']:
        target = 0
    if target == 0 and state['shares']>0:
        qty = state['shares']
        state['cash'] += qty*price*(1-cost)-FIXED_FEE
        state['shares'] = 0.0
        state['trades'].append({'date': date, 'side': 'SELL', 'price': price, 'shares': qty})
    elif target == 1 and state['shares']==0:
        spend = min(state['cash'], equity*MAX_ALLOCATION)
        if spend > FIXED_FEE:
            qty = (spend-FIXED_FEE)/(price*(1+cost))
            state['cash'] -= qty*price*(1+cost)+FIXED_FEE
            state['shares'] = qty
            state['trades'].append({'date': date, 'side': 'BUY', 'price': price, 'shares': qty})
    state['equity'] = round(state['cash']+state['shares']*price, 4)
    state['last_date'] = date
    state['next_session_signal'] = int(data.signal.iloc[-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(path)
    return state


def main():
    p = argparse.ArgumentParser(description='Gold AutoTrader: no live orders')
    p.add_argument('command', choices=['backtest','paper'])
    p.add_argument('--ticker', default=DEFAULT_TICKER)
    p.add_argument('--csv', help='CSV with Date and Close columns, GBP prices')
    p.add_argument('--period', default='10y')
    p.add_argument('--out', default='results')
    p.add_argument('--fixed-fee', type=float, default=0.0)
    a = p.parse_args()
    if a.fixed_fee < 0:
        p.error('Fee cannot be negative')
    df = fetch_prices(a.ticker, a.period, a.csv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.command == 'backtest':
        curve, trades = backtest(df, fixed_fee=a.fixed_fee)
        curve.to_csv(out/'equity_curve.csv')
        trades.to_csv(out/'trades.csv', index=False)
        m = metrics(curve, trades)
        (out/'metrics.json').write_text(json.dumps(m, indent=2))
        print(json.dumps(m, indent=2))
    else:
        if a.fixed_fee != 0:
            p.error('--fixed-fee is only supported for backtest; paper mode uses FIXED_FEE in code')
        state = paper_step(df, out/'paper_state.json')
        print(json.dumps({k:v for k,v in state.items() if k!='trades'}, indent=2))

if __name__ == '__main__':
    main()
