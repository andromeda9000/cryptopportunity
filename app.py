import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

st.set_page_config(page_title='Cripto Opportunity', page_icon='🧠', layout='wide', initial_sidebar_state='expanded')

BINANCE_URLS = ['https://api.binance.com', 'https://data-api.binance.vision']
COINGECKO_MARKETS = 'https://api.coingecko.com/api/v3/coins/markets'
COINGECKO_OHLC = 'https://api.coingecko.com/api/v3/coins/{id}/ohlc'
FG_URL = 'https://api.alternative.me/fng/'
VIX_YF = 'https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX'
HEADERS = {'User-Agent': 'Mozilla/5.0'}
TIMEFRAME_TO_CG_DAYS = {'1h': 7, '4h': 30, '1d': 90}
STATUS_COLORS = {'LONG': '#22c55e', 'SHORT': '#ef4444', 'WATCH': '#eab308', 'NO TRADE': '#3b82f6'}
STATUS_ORDER = ['LONG', 'SHORT', 'WATCH', 'NO TRADE']


@st.cache_data(ttl=1800)
def load_cg_map():
    params = {
        'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 250,
        'page': 1, 'sparkline': 'false', 'price_change_percentage': '24h'
    }
    r = requests.get(COINGECKO_MARKETS, params=params, timeout=20, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    mapping, top = {}, []
    for item in data:
        sym = item.get('symbol', '').upper()
        row = {
            'symbol': f'{sym}USDT',
            'coin': sym,
            'volume': float(item.get('total_volume') or 0),
            'price': float(item.get('current_price') or 0),
            'change': float(item.get('price_change_percentage_24h') or 0),
            'cg_id': item.get('id')
        }
        if sym and sym not in mapping:
            mapping[sym] = row
        top.append(row)
    return mapping, top


@st.cache_data(ttl=300)
def get_top_crypto_pairs(limit=100):
    for base in BINANCE_URLS:
        try:
            r = requests.get(f'{base}/api/v3/ticker/24hr', timeout=15, headers=HEADERS)
            r.raise_for_status()
            data = r.json()
            rows = [
                {
                    'symbol': x['symbol'],
                    'coin': x['symbol'].replace('USDT', ''),
                    'volume': float(x['quoteVolume']),
                    'price': float(x['lastPrice']),
                    'change': float(x['priceChangePercent']),
                    'cg_id': None,
                }
                for x in data if x['symbol'].endswith('USDT')
            ]
            rows.sort(key=lambda x: x['volume'], reverse=True)
            return rows[:limit], 'binance'
        except Exception:
            pass
    try:
        _, top = load_cg_map()
        top.sort(key=lambda x: x['volume'], reverse=True)
        return top[:limit], 'coingecko'
    except Exception:
        return [], 'none'


@st.cache_data(ttl=60)
def get_crypto_data(symbol='BTCUSDT', interval='1h', limit=250):
    for base in BINANCE_URLS:
        try:
            r = requests.get(f'{base}/api/v3/klines', params={'symbol': symbol, 'interval': interval, 'limit': limit}, timeout=15, headers=HEADERS)
            r.raise_for_status()
            data = r.json()
            if not data or isinstance(data, dict):
                continue
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qa_vol', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = df[c].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df, 'binance'
        except Exception:
            pass
    try:
        cg_map, _ = load_cg_map()
        coin = cg_map.get(symbol.replace('USDT', '').upper())
        if coin and interval in TIMEFRAME_TO_CG_DAYS:
            r = requests.get(COINGECKO_OHLC.format(id=coin['cg_id']), params={'vs_currency': 'usd', 'days': TIMEFRAME_TO_CG_DAYS[interval]}, timeout=20, headers=HEADERS)
            r.raise_for_status()
            data = r.json()
            if data:
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['volume'] = np.nan
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                if len(df) > limit:
                    df = df.tail(limit)
                df.set_index('timestamp', inplace=True)
                return df, 'coingecko'
    except Exception:
        pass
    return pd.DataFrame(), 'none'


@st.cache_data(ttl=600)
def get_fear_greed():
    try:
        r = requests.get(FG_URL, params={'limit': 1, 'format': 'json'}, timeout=10, headers=HEADERS)
        d = r.json()['data'][0]
        return int(d['value']), d['value_classification']
    except Exception:
        return None, 'N/A'


@st.cache_data(ttl=600)
def get_vix():
    try:
        r = requests.get(VIX_YF, params={'range': '1d', 'interval': '1d'}, timeout=10, headers=HEADERS)
        meta = r.json()['chart']['result'][0]['meta']
        return meta.get('regularMarketPrice')
    except Exception:
        return None


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    f = series.ewm(span=fast, adjust=False).mean()
    s = series.ewm(span=slow, adjust=False).mean()
    m = f - s
    sig = m.ewm(span=signal, adjust=False).mean()
    h = m - sig
    return m, sig, h


def atr(df, period=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def adx(df, period=14):
    h, l, c = df['high'], df['low'], df['close']
    up = h.diff()
    dn = -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.rolling(period).mean()
    pdi = 100 * (pdm.rolling(period).mean() / atr_.replace(0, np.nan))
    mdi = 100 * (mdm.rolling(period).mean() / atr_.replace(0, np.nan))
    dx = 100 * ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan))
    return dx.rolling(period).mean(), pdi, mdi


def bollinger(series, period=20, mult=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid + mult * std, mid - mult * std, mid


def supertrend(df, period=10, mult=3.0):
    a = atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    fup = upper.copy()
    flo = lower.copy()
    st = pd.Series(np.nan, index=df.index)
    d = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        fup.iloc[i] = min(upper.iloc[i], fup.iloc[i - 1]) if df['close'].iloc[i - 1] <= fup.iloc[i - 1] else upper.iloc[i]
        flo.iloc[i] = max(lower.iloc[i], flo.iloc[i - 1]) if df['close'].iloc[i - 1] >= flo.iloc[i - 1] else lower.iloc[i]
        d.iloc[i] = 1 if df['close'].iloc[i] > fup.iloc[i - 1] else -1 if df['close'].iloc[i] < flo.iloc[i - 1] else d.iloc[i - 1]
        st.iloc[i] = flo.iloc[i] if d.iloc[i] == 1 else fup.iloc[i]
    return st, d


def compute(df):
    df = df.copy()
    df['EMA20'] = ema(df['close'], 20)
    df['EMA50'] = ema(df['close'], 50)
    df['EMA200'] = ema(df['close'], 200)
    df['RSI'] = rsi(df['close'])
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = macd(df['close'])
    df['ATR'] = atr(df)
    df['ADX'], df['DI+'], df['DI-'] = adx(df)
    df['BB_UP'], df['BB_DN'], df['BB_MID'] = bollinger(df['close'])
    df['ST'], df['ST_DIR'] = supertrend(df)
    return df


def score(df):
    last = df.iloc[-1]
    close = last['close']
    rsi_v = last['RSI']
    macd_v = last['MACD']
    sig_v = last['MACD_SIGNAL']
    hist_v = last['MACD_HIST']
    adx_v = last['ADX']
    atr_v = last['ATR']
    st_dir = last['ST_DIR']
    ls, ss = 0, 0
    lr, sr = [], []
    if last['EMA20'] > last['EMA50'] > last['EMA200']:
        ls += 25; lr.append('EMA rialziste')
    elif last['EMA20'] < last['EMA50'] < last['EMA200']:
        ss += 25; sr.append('EMA ribassiste')
    if close > last['EMA20']:
        ls += 10; lr.append('Prezzo sopra EMA20')
    else:
        ss += 10; sr.append('Prezzo sotto EMA20')
    if rsi_v >= 55:
        ls += 10; lr.append('RSI > 55')
    elif rsi_v <= 45:
        ss += 10; sr.append('RSI < 45')
    if macd_v > sig_v and hist_v > 0:
        ls += 15; lr.append('MACD bullish')
    elif macd_v < sig_v and hist_v < 0:
        ss += 15; sr.append('MACD bearish')
    if pd.notna(adx_v) and adx_v >= 18:
        if last['DI+'] > last['DI-']:
            ls += 10; lr.append('ADX trend up')
        elif last['DI-'] > last['DI+']:
            ss += 10; sr.append('ADX trend down')
    if st_dir == 1:
        ls += 15; lr.append('SuperTrend bullish')
    elif st_dir == -1:
        ss += 15; sr.append('SuperTrend bearish')
    vol_ma = df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df.columns else np.nan
    if pd.notna(last['volume']) and pd.notna(vol_ma) and last['volume'] > vol_ma:
        ls += 5; ss += 5
    regime_ok = pd.notna(adx_v) and adx_v >= 18
    if pd.notna(vol_ma) and pd.notna(last['volume']):
        regime_ok = regime_ok and last['volume'] > 0.8 * vol_ma
    if not regime_ok:
        return 'NO TRADE', 0, ['Regime debole'], close, adx_v, rsi_v, atr_v
    if ls >= ss + 15 and ls >= 50:
        return 'LONG', min(100, ls), lr, close, adx_v, rsi_v, atr_v
    if ss >= ls + 15 and ss >= 50:
        return 'SHORT', min(100, ss), sr, close, adx_v, rsi_v, atr_v
    return 'WATCH', max(ls, ss), ['Bias non allineato'], close, adx_v, rsi_v, atr_v


def levels(price, atr_v, direction):
    atr_v = atr_v if atr_v and np.isfinite(atr_v) else price * 0.01
    if direction == 'LONG':
        return price - 1.5 * atr_v, price + 1.5 * atr_v, price + 2.5 * atr_v
    if direction == 'SHORT':
        return price + 1.5 * atr_v, price - 1.5 * atr_v, price - 2.5 * atr_v
    return None, None, None


def badge(final):
    return final


def chart_explain(row):
    final = row['final']
    if final == 'LONG':
        return ['Prezzo sopra EMA20/50/200', 'Momentum positivo sopra zero', 'RSI e MACD favorevoli', 'SuperTrend rialzista e ADX presente']
    if final == 'SHORT':
        return ['Prezzo sotto EMA20/50/200', 'Momentum negativo sotto zero', 'RSI e MACD deboli', 'SuperTrend ribassista e ADX presente']
    if final == 'WATCH':
        return ['Alcune conferme ma non allineate', 'Trend presente ma non abbastanza forte', 'Momentum misto', 'Aspetta breakout o pullback migliore']
    return ['Regime troppo debole', 'Volume scarso o ADX basso', 'Direzionalità non chiara', 'Meglio attendere']


def render_chart(df, row, interval='1h'):
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=[0.54, 0.12, 0.11, 0.11, 0.12], subplot_titles=['Prezzo', 'Volume', 'Momentum', 'RSI', 'MACD'])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], increasing_line_color='#2dd4bf', decreasing_line_color='#fb7185', increasing_fillcolor='#2dd4bf', decreasing_fillcolor='#fb7185', whiskerwidth=0.4, name='Price'), row=1, col=1)
    for col, color, width in [('EMA20', '#f59e0b', 2.4), ('EMA50', '#60a5fa', 1.8), ('EMA200', '#fda4af', 1.8)]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=color, width=width)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_UP'], name='BB Upper', line=dict(color='rgba(148,163,184,0.42)', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_DN'], name='BB Lower', line=dict(color='rgba(148,163,184,0.42)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(148,163,184,0.05)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ST'], name='SuperTrend', line=dict(color='#22c55e', width=2.6)), row=1, col=1)
    last = df.iloc[-1]
    x_last = df.index[-1]
    fig.add_trace(go.Scatter(x=[x_last], y=[last['close']], mode='markers+text', text=[row['final']], textposition='top center', marker=dict(size=14, color=STATUS_COLORS[row['final']], line=dict(color='white', width=1)), name='Signal'), row=1, col=1)
    price = row['price']
    atrv = row['price'] * 0.01 if row['sl'] is None else abs(row['price'] - row['sl']) / 1.5
    sl, tp1, tp2 = levels(price, atrv, row['final'])
    for lv, name, color, dash in [(price, 'Entry', '#f59e0b', 'solid'), (sl, 'SL', '#ef4444', 'dash'), (tp1, 'TP1', '#22c55e', 'dot'), (tp2, 'TP2', '#60a5fa', 'dot')]:
        if lv is not None and np.isfinite(lv):
            fig.add_hline(y=lv, line_color=color, line_width=1.5, line_dash=dash, row=1, col=1)
            fig.add_annotation(x=x_last, y=lv, text=name, showarrow=False, xshift=36, font=dict(size=10, color=color), row=1, col=1)
    volume_plot = df['volume'].fillna(0)
    vol_colors = ['#2dd4bf' if c >= o else '#fb7185' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df.index, y=volume_plot, name='Volume', marker_color=vol_colors, opacity=0.68), row=2, col=1)
    momentum = (df['close'] - df['EMA20']) / df['EMA20'] * 100
    fig.add_trace(go.Scatter(x=df.index, y=momentum, name='Momentum %', line=dict(color='#a78bfa', width=1.8), fill='tozeroy', fillcolor='rgba(167,139,250,0.08)'), row=3, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.20)', row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#d8b4fe', width=2)), row=4, col=1)
    for level, color in [(70, 'rgba(251,113,133,.45)'), (30, 'rgba(45,212,191,.45)'), (50, 'rgba(255,255,255,.20)')]:
        fig.add_hline(y=level, line_dash='dash', line_color=color, row=4, col=1)
    hist_colors = ['#2dd4bf' if v >= 0 else '#fb7185' for v in df['MACD_HIST']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], name='MACD Hist', marker_color=hist_colors, opacity=0.5), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#60a5fa', width=1.6)), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], name='Signal', line=dict(color='#f59e0b', width=1.6)), row=5, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.20)', row=5, col=1)
    fig.update_layout(template='plotly_dark', height=1180, margin=dict(l=18, r=18, t=72, b=18), legend=dict(orientation='h', y=1.03, x=0.01, font=dict(size=11), bgcolor='rgba(0,0,0,0)'), title=dict(text=f'{row["pair"]} · {interval}', font=dict(size=18)), hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(rangeslider_visible=False, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.10)', zeroline=False)
    return fig


def main():
    st.markdown('₿Cripto OpportunityΞGrafico chiaro e professionale · entry/SL/TP visualizzati · lettura del motore spiegata', unsafe_allow_html=True)
    fg_val, fg_lbl = get_fear_greed()
    vix_val = get_vix()
    top, source_top = get_top_crypto_pairs(100)
    if source_top != 'binance':
        st.info(f'Sorgente mercato: {source_top}')
    if not top:
        st.error('Impossibile caricare il mercato crypto.')
        st.stop()
    big25, alt75 = top[:25], top[25:100]
    stablecoins = {'USDT', 'USDC', 'DAI', 'FDUSD', 'TUSD', 'USDE', 'PYUSD', 'FRAX', 'LUSD', 'GUSD', 'USDD', 'USDP', 'USTC', 'EURS', 'EURC', 'RLUSD', 'BUSD', 'USD1'}
    allowed_paired = {'EURUSDT'}

    def is_excluded(sym):
        base = sym.replace('USDT', '')
        if sym in allowed_paired:
            return False
        if base in stablecoins:
            return True
        return False

    def is_low_vol(sym):
        base = sym.replace('USDT', '')
        low_vol = {'USDC', 'FDUSD', 'TUSD', 'DAI', 'USDE', 'PYUSD', 'GUSD', 'EURC', 'EURS', 'USDP', 'BUSD', 'USDD', 'LUSD'}
        return base in low_vol

    with st.sidebar:
        st.markdown('### Setup')
        group = st.radio('Gruppo', ['Top 25', 'Altre 75'], index=0)
        vol_filter = st.radio('Filtro vol', ['Solo volatili', 'Anche low vol'], index=0)
        timeframe_options = ['5m', '15m', '1h', '4h', '12h', '1d']
        if source_top != 'binance':
            timeframe_options = ['1h', '4h', '1d']
        interval = st.selectbox('Timeframe', timeframe_options, index=0 if source_top != 'binance' else 2)
        show_states = st.multiselect('Mostra stati', STATUS_ORDER, default=STATUS_ORDER)
        scan = st.button('Scansiona', type='primary', use_container_width=True)
        st.caption('Stable escluse sempre, tranne EUR/USDT.')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Fear & Greed', f'{fg_val if fg_val is not None else "N/A"}', fg_lbl)
    c2.metric('VIX', f'{vix_val:.1f}' if vix_val else 'N/A')
    c3.metric('Top 25', len(big25))
    c4.metric('Altre 75', len(alt75))

    if not scan and 'scanner_results' not in st.session_state:
        st.info('Premi Scansiona per vedere i risultati.')
        return

    if scan or 'scanner_results' not in st.session_state or st.session_state.get('scanner_group') != group or st.session_state.get('scanner_interval') != interval or st.session_state.get('scanner_vol_filter') != vol_filter:
        raw_list = big25 if group == 'Top 25' else alt75
        scan_list = [x for x in raw_list if not is_excluded(x['symbol'])]
        if vol_filter == 'Solo volatili':
            scan_list = [x for x in scan_list if not is_low_vol(x['symbol'])]
        results = []
        progress = st.progress(0)
        for i, item in enumerate(scan_list, 1):
            try:
                df, price_source = get_crypto_data(item['symbol'], interval, 250)
                if df.empty or len(df) < 80:
                    continue
                df = compute(df)
                final, conf, reasons, price, adx_v, rsi_v, atr_v = score(df)
                sl, tp1, tp2 = levels(price, atr_v, final)
                results.append({'pair': item['symbol'], 'final': final, 'confidence': conf, 'price': price, 'rsi': rsi_v, 'adx': adx_v, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'reasons': reasons, 'df': df, 'price_source': price_source})
            except Exception as e:
                results.append({'pair': item['symbol'], 'final': 'NO TRADE', 'confidence': 0, 'price': np.nan, 'rsi': np.nan, 'adx': np.nan, 'sl': None, 'tp1': None, 'tp2': None, 'reasons': [str(e)], 'df': pd.DataFrame(), 'price_source': 'error'})
            progress.progress(i / len(scan_list))
        st.session_state.scanner_results = results
        st.session_state.scanner_group = group
        st.session_state.scanner_interval = interval
        st.session_state.scanner_vol_filter = vol_filter

    results = st.session_state.scanner_results
    results = [r for r in results if r['final'] in show_states]
    results.sort(key=lambda x: (x['final'] not in ('LONG', 'SHORT'), -x['confidence']))

    counts = {k: sum(1 for r in results if r['final'] == k) for k in STATUS_ORDER}
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"LONG {counts['LONG']}", unsafe_allow_html=True)
    m2.markdown(f"SHORT {counts['SHORT']}", unsafe_allow_html=True)
    m3.markdown(f"WATCH {counts['WATCH']}", unsafe_allow_html=True)
    m4.markdown(f"NO TRADE {counts['NO TRADE']}", unsafe_allow_html=True)

    st.markdown('### Analisi selezionata')
    if not results:
        st.warning('Nessun risultato disponibile con i filtri attuali.')
        return
    left, right = st.columns([2.15, 1])
    with left:
        selected = st.selectbox('Seleziona coppia', [r['pair'] for r in results], index=0)
        if selected:
            row = next(r for r in results if r['pair'] == selected)
            if not row['df'].empty:
                st.plotly_chart(render_chart(row['df'], row, interval), use_container_width=True)
                st.markdown('Il grafico evidenzia prezzo, trend, momentum e livelli operativi del motore.', unsafe_allow_html=True)
                st.markdown(f"{badge(row['final'])} {row['confidence']}/100", unsafe_allow_html=True)
                sl_txt = f"{row['sl']:.4f}" if row.get('sl') is not None else 'N/A'
                tp1_txt = f"{row['tp1']:.4f}" if row.get('tp1') is not None else 'N/A'
                tp2_txt = f"{row['tp2']:.4f}" if row.get('tp2') is not None else 'N/A'
                st.write(f"**SL:** {sl_txt} | **TP1:** {tp1_txt} | **TP2:** {tp2_txt}")
                st.caption(' · '.join(row['reasons'][:4]))
                if row.get('price_source') and row['price_source'] != 'binance':
                    st.caption(f"Storico prezzi: {row['price_source']}")
    with right:
        st.markdown('### Lettura motore')
        row = next((r for r in results if r['pair'] == selected), results[0])
        st.markdown('Cosa sta dicendo il motore', unsafe_allow_html=True)
        for x in chart_explain(row):
            st.write(f'• {x}')
        bias = 'rialzista' if row['final'] == 'LONG' else 'ribassista' if row['final'] == 'SHORT' else 'laterale'
        st.markdown('### Lettura veloce del setup')
        st.write(f"Stato: {row['final']}")
        st.write(f"Bias: {bias}")
        st.write(f"RSI: {row['rsi']:.1f} · ADX: {row['adx']:.1f}")
        st.write(f"Confidenza: {row['confidence']}/100")
        if row['final'] in ('LONG', 'SHORT'):
            st.markdown('### Livelli chiave')
            st.write(f"Entry: {row['price']:.4f}")
            st.write(f"SL: {row['sl']:.4f}")
            st.write(f"TP1: {row['tp1']:.4f}")
            st.write(f"TP2: {row['tp2']:.4f}")
        st.markdown('### Snapshot')
        for r in results[:8]:
            st.write(f"{r['pair']} — RSI {r['rsi']:.1f} · ADX {r['adx']:.1f} · {r['final']} {r['confidence']}/100")

if __name__ == '__main__':
    main()
