import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title='Crypto Opportunity Scanner', page_icon='📈', layout='wide', initial_sidebar_state='expanded')

BINANCE_URLS = ['https://api.binance.com', 'https://data-api.binance.vision']
COINGECKO_MARKETS = 'https://api.coingecko.com/api/v3/coins/markets'
COINGECKO_OHLC = 'https://api.coingecko.com/api/v3/coins/{id}/ohlc'
COINGECKO_SIMPLE = 'https://api.coingecko.com/api/v3/simple/price'
HEADERS = {'User-Agent': 'Mozilla/5.0'}
TIMEFRAME_TO_CG_DAYS = {'15m': 1, '1h': 7, '4h': 30, '1d': 90}
DEFAULT_COINS = [
    {'symbol': 'BTCUSDT', 'coin': 'BTC', 'price': 0.0, 'change': 0.0, 'volume': 0.0, 'source': 'fallback', 'cg_id': 'bitcoin'},
    {'symbol': 'ETHUSDT', 'coin': 'ETH', 'price': 0.0, 'change': 0.0, 'volume': 0.0, 'source': 'fallback', 'cg_id': 'ethereum'},
    {'symbol': 'SOLUSDT', 'coin': 'SOL', 'price': 0.0, 'change': 0.0, 'volume': 0.0, 'source': 'fallback', 'cg_id': 'solana'},
]

@st.cache_data(ttl=900)
def load_coingecko_market_map():
    params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 250, 'page': 1, 'sparkline': 'false', 'price_change_percentage': '24h'}
    r = requests.get(COINGECKO_MARKETS, params=params, timeout=20, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    by_symbol, top = {}, []
    for item in data:
        sym = item.get('symbol', '').upper()
        coin = {
            'cg_id': item.get('id'), 'coin': sym, 'name': item.get('name', sym),
            'price': float(item.get('current_price') or 0), 'change': float(item.get('price_change_percentage_24h') or 0),
            'volume': float(item.get('total_volume') or 0), 'symbol': f'{sym}USDT', 'source': 'coingecko'
        }
        if sym and sym not in by_symbol:
            by_symbol[sym] = coin
        top.append(coin)
    return by_symbol, top

@st.cache_data(ttl=900)
def get_top_crypto_pairs(limit=100):
    for base in BINANCE_URLS:
        try:
            r = requests.get(f'{base}/api/v3/ticker/24hr', timeout=20, headers=HEADERS)
            r.raise_for_status()
            data = r.json()
            rows = []
            for item in data:
                sym = item.get('symbol', '')
                if not sym.endswith('USDT'):
                    continue
                rows.append({'symbol': sym, 'coin': sym.replace('USDT', ''), 'price': float(item.get('lastPrice') or 0), 'change': float(item.get('priceChangePercent') or 0), 'volume': float(item.get('quoteVolume') or 0), 'source': 'binance', 'cg_id': None})
            rows.sort(key=lambda x: x['volume'], reverse=True)
            return rows[:limit], 'binance'
        except Exception:
            pass
    try:
        _, top = load_coingecko_market_map()
        return top[:limit], 'coingecko'
    except Exception:
        return DEFAULT_COINS, 'fallback'

@st.cache_data(ttl=300)
def get_spot_price_from_coingecko(coin_symbol):
    cg_map, _ = load_coingecko_market_map()
    coin = cg_map.get(coin_symbol.upper())
    if not coin:
        return None
    params = {'ids': coin['cg_id'], 'vs_currencies': 'usd', 'include_24hr_change': 'true'}
    r = requests.get(COINGECKO_SIMPLE, params=params, timeout=20, headers=HEADERS)
    r.raise_for_status()
    data = r.json().get(coin['cg_id'], {})
    return {'price': float(data.get('usd') or 0), 'change': float(data.get('usd_24h_change') or 0), 'cg_id': coin['cg_id']}

@st.cache_data(ttl=180)
def get_crypto_data(symbol='BTCUSDT', interval='1h', limit=250):
    for base in BINANCE_URLS:
        try:
            r = requests.get(f'{base}/api/v3/klines', params={'symbol': symbol, 'interval': interval, 'limit': limit}, timeout=20, headers=HEADERS)
            r.raise_for_status()
            data = r.json()
            if data and not isinstance(data, dict):
                df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','num_trades','taker_buy_base','taker_buy_quote','ignore'])
                for col in ['open','high','low','close','volume']:
                    df[col] = df[col].astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df, 'binance'
        except Exception:
            pass
    try:
        cg_map, _ = load_coingecko_market_map()
        coin = cg_map.get(symbol.replace('USDT', '').upper())
        if coin and interval in TIMEFRAME_TO_CG_DAYS:
            r = requests.get(COINGECKO_OHLC.format(id=coin['cg_id']), params={'vs_currency': 'usd', 'days': TIMEFRAME_TO_CG_DAYS[interval]}, timeout=20, headers=HEADERS)
            r.raise_for_status()
            raw = r.json()
            if raw:
                df = pd.DataFrame(raw, columns=['timestamp','open','high','low','close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['volume'] = np.nan
                if len(df) > limit:
                    df = df.tail(limit)
                df.set_index('timestamp', inplace=True)
                return df, 'coingecko'
    except Exception:
        pass
    return pd.DataFrame(), 'none'

@st.cache_data(ttl=3600)
def get_economic_calendar():
    try:
        r = requests.get('https://nfs.faireconomy.media/ff_calendar_thisweek.json', timeout=20, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        rows = []
        for item in data:
            raw_date = item.get('date', '')
            try:
                dt = datetime.strptime(raw_date, '%Y-%m-%dT%H:%M:%S%z')
                date_str, time_str = dt.strftime('%Y-%m-%d'), dt.strftime('%H:%M')
            except Exception:
                date_str, time_str = raw_date[:10] if len(raw_date) >= 10 else raw_date, '00:00'
            rows.append({'date': date_str, 'time': time_str, 'country': item.get('country', ''), 'event': item.get('title', ''), 'impact': (item.get('impact') or 'Low').upper(), 'prev': item.get('previous') or '-', 'forecast': item.get('forecast') or '-', 'actual': item.get('actual') or ''})
        return rows
    except Exception:
        today = datetime.now()
        return [
            {'date': (today + timedelta(days=1)).strftime('%Y-%m-%d'), 'time': '14:30', 'country': 'US', 'event': 'Non-Farm Payrolls', 'impact': 'HIGH', 'prev': '-', 'forecast': '-', 'actual': ''},
            {'date': (today + timedelta(days=3)).strftime('%Y-%m-%d'), 'time': '14:30', 'country': 'US', 'event': 'CPI Inflation Rate', 'impact': 'HIGH', 'prev': '-', 'forecast': '-', 'actual': ''},
            {'date': (today + timedelta(days=8)).strftime('%Y-%m-%d'), 'time': '14:15', 'country': 'EU', 'event': 'ECB Rate Decision', 'impact': 'HIGH', 'prev': '-', 'forecast': '-', 'actual': ''},
        ]

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal, macd - signal

def calculate_bollinger(series, period=20, std_mult=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + (std * std_mult), sma - (std * std_mult), sma

def calculate_atr(df, period=14):
    tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df, ema_periods):
    df = df.copy()
    for p in ema_periods:
        df[f'EMA_{p}'] = calculate_ema(df['close'], p)
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD_line'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df['close'])
    df['BB_upper'], df['BB_lower'], df['BB_middle'] = calculate_bollinger(df['close'])
    df['ATR'] = calculate_atr(df)
    return df

class SimpleScanner:
    def evaluate(self, df, ema_periods):
        score, reasons = 0, []
        last_close = df['close'].iloc[-1]
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else np.nan
        macd = df['MACD_line'].iloc[-1] if 'MACD_line' in df.columns else np.nan
        macd_signal = df['MACD_signal'].iloc[-1] if 'MACD_signal' in df.columns else np.nan
        if ema_periods:
            above = sum(1 for p in ema_periods if f'EMA_{p}' in df.columns and last_close > df[f'EMA_{p}'].iloc[-1])
            score += int((above / len(ema_periods)) * 40)
            reasons.append(f'Prezzo sopra {above}/{len(ema_periods)} EMA')
        if not np.isnan(rsi):
            if 50 <= rsi <= 68:
                score += 20; reasons.append('RSI favorevole')
            elif rsi < 35:
                score += 10; reasons.append('RSI in area di rimbalzo')
            elif rsi > 75:
                score -= 10; reasons.append('RSI surriscaldato')
        if not np.isnan(macd) and not np.isnan(macd_signal):
            if macd > macd_signal:
                score += 20; reasons.append('MACD bullish')
            else:
                score -= 5; reasons.append('MACD debole')
        delta = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100 if len(df) > 1 else 0
        if delta > 0:
            score += 10; reasons.append('Momentum ultima candela positivo')
        signal = 'LONG' if score >= 55 else 'WATCH' if score >= 35 else 'NEUTRAL'
        return {'score': max(0, min(100, score)), 'signal': signal, 'reasons': reasons}

def create_chart(df, ema_periods, show_bbands=True):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.58, 0.20, 0.22])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Prezzo'), row=1, col=1)
    colors = {9: '#00d4ff', 20: '#f6c344', 50: '#ff7b72', 200: '#7ee787'}
    for p in ema_periods:
        col = f'EMA_{p}'
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1.6, color=colors.get(p, '#cccccc'))), row=1, col=1)
    if show_bbands:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', line=dict(width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', line=dict(width=1, dash='dot'), fill='tonexty', fillcolor='rgba(100,100,255,0.08)'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['volume'].fillna(0), name='Volume'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#c678dd')), row=3, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red', row=3, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_layout(template='plotly_dark', height=780, margin=dict(l=25, r=25, t=40, b=20), xaxis_rangeslider_visible=False)
    return fig

def main():
    st.title('📈 Crypto Opportunity Scanner')
    st.caption('Deploy-safe version con fallback automatico Binance -> CoinGecko e UI resiliente.')
    pairs, market_source = get_top_crypto_pairs(120)
    if market_source == 'binance':
        st.success('Dati mercato caricati da Binance.')
    elif market_source == 'coingecko':
        st.warning('Binance non disponibile dal deploy: uso CoinGecko come fallback.')
    else:
        st.warning('Fonti mercato temporaneamente limitate: avvio app con fallback minimo.')
    with st.sidebar:
        st.header('Configurazione')
        options = [f"{x['coin']} ({x['symbol']})" for x in pairs]
        selected = st.selectbox('Coin', options, index=0)
        selected_row = next((x for x in pairs if f"{x['coin']} ({x['symbol']})" == selected), pairs[0])
        interval = st.selectbox('Timeframe', ['15m', '1h', '4h', '1d'], index=1)
        limit = st.slider('Candele', 100, 400, 220, step=20)
        ema_periods = st.multiselect('EMA', [9, 20, 50, 200], default=[20, 50, 200])
        show_bbands = st.checkbox('Bollinger Bands', value=True)
        if st.button('Aggiorna'):
            st.cache_data.clear(); st.rerun()
    symbol = selected_row['symbol']
    df, price_source = get_crypto_data(symbol, interval, limit)
    if df.empty:
        st.error('Impossibile caricare storico prezzi da Binance e CoinGecko.')
        return
    if price_source == 'coingecko':
        st.info(f"Storico prezzi di {selected_row['coin']} caricato da CoinGecko.")
    elif price_source == 'binance':
        st.caption('Storico prezzi caricato da Binance.')
    df = add_indicators(df, ema_periods or [20, 50])
    result = SimpleScanner().evaluate(df, ema_periods or [20, 50])
    current_price = float(df['close'].iloc[-1])
    previous_price = float(df['close'].iloc[-2]) if len(df) > 1 else current_price
    delta_pct = ((current_price / previous_price) - 1) * 100 if previous_price else 0
    fallback_spot = None
    if price_source != 'binance':
        try:
            fallback_spot = get_spot_price_from_coingecko(selected_row['coin'])
        except Exception:
            fallback_spot = None
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Coin', selected_row['coin'])
    k2.metric('Prezzo', f'${current_price:,.4f}', f'{delta_pct:+.2f}%')
    k3.metric('Segnale', result['signal'])
    k4.metric('AI Score', f"{result['score']}/100")
    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.plotly_chart(create_chart(df, ema_periods or [20, 50], show_bbands=show_bbands), use_container_width=True)
    with c2:
        if result['signal'] == 'LONG':
            st.success(f"Setup favorevole su {selected_row['coin']}")
        elif result['signal'] == 'WATCH':
            st.warning(f"{selected_row['coin']} da monitorare")
        else:
            st.info(f"Nessun setup forte su {selected_row['coin']}")
        st.subheader('Motivi')
        for reason in result['reasons']:
            st.write(f'- {reason}')
        st.subheader('Sorgenti')
        st.write(f'- Mercato: {market_source}')
        st.write(f'- Storico prezzi: {price_source}')
        if fallback_spot:
            st.write(f"- Spot fallback: ${fallback_spot['price']:,.4f}")
    st.subheader('Calendario macro')
    events_df = pd.DataFrame(get_economic_calendar())
    if not events_df.empty:
        st.dataframe(events_df.head(12), use_container_width=True)
    st.subheader('Ultime candele')
    show_cols = [c for c in ['open','high','low','close','volume','RSI','MACD_line','MACD_signal','ATR'] if c in df.columns]
    st.dataframe(df.tail(20)[show_cols].round(4), use_container_width=True)

if __name__ == '__main__':
    main()
