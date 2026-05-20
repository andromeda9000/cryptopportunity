import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Cripto Opportunity', page_icon='🧠', layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:1.2rem;max-width:1500px}
[data-testid='stSidebar']{background:linear-gradient(180deg,#0d1320 0%,#0b1120 100%);border-right:1px solid rgba(148,163,184,.10)}
.jarvis-hero{background:linear-gradient(135deg,rgba(59,130,246,.14),rgba(168,85,247,.10));border:1px solid rgba(148,163,184,.14);border-radius:24px;padding:18px 20px;margin-bottom:14px;box-shadow:0 12px 30px rgba(0,0,0,.20)}
.jarvis-title{font-size:38px;font-weight:900;letter-spacing:-.01em;line-height:1.15;display:flex;align-items:center;justify-content:center;gap:0px;flex-wrap:nowrap;white-space:nowrap;overflow:hidden}
.jarvis-sub{color:#94a3b8;font-size:13px;margin-top:6px}
.soft-card{background:rgba(15,23,42,.88);border:1px solid rgba(148,163,184,.12);border-radius:18px;padding:14px 16px;box-shadow:0 8px 18px rgba(0,0,0,.15)}
.section-title{font-size:14px;font-weight:800;color:#e2e8f0;margin:8px 0 10px 2px;letter-spacing:.01em}
div[data-testid='metric-container']{background:rgba(15,23,42,.88);border:1px solid rgba(148,163,184,.12);padding:12px 14px;border-radius:16px;box-shadow:0 6px 18px rgba(0,0,0,.16)}
.status-pill{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;font-weight:800;font-size:12px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08)}
.status-dot{width:10px;height:10px;border-radius:999px;display:inline-block}
.result-row{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:12px 14px;border-radius:16px;border:1px solid rgba(148,163,184,.12);background:rgba(15,23,42,.88);margin-bottom:10px}
.small-muted{color:#94a3b8;font-size:12px}
.chart-panel{background:linear-gradient(180deg, rgba(15,23,42,.95), rgba(10,15,26,.95));border:1px solid rgba(148,163,184,.10);border-radius:22px;padding:14px;box-shadow:0 12px 30px rgba(0,0,0,.22)}
.legend-chip{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);margin-right:6px;font-size:11px;color:#cbd5e1}
.legend-dot{width:8px;height:8px;border-radius:50%}
.explain-box{background:rgba(15,23,42,.88);border:1px solid rgba(148,163,184,.12);border-radius:18px;padding:14px 16px;line-height:1.55;color:#e2e8f0}
</style>
""", unsafe_allow_html=True)

BINANCE='https://api.binance.com'
FG_URL='https://api.alternative.me/fng/'
VIX_YF='https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX'
STATUS_COLORS={'LONG':'#22c55e','SHORT':'#ef4444','WATCH':'#eab308','NO TRADE':'#3b82f6'}
STATUS_ORDER=['LONG','SHORT','WATCH','NO TRADE']

@st.cache_data(ttl=300)
def get_top_crypto_pairs(limit=100):
    r=requests.get(f'{BINANCE}/api/v3/ticker/24hr',timeout=15)
    r.raise_for_status(); data=r.json()
    rows=[{'symbol':x['symbol'],'coin':x['symbol'].replace('USDT',''),'volume':float(x['quoteVolume']),'price':float(x['lastPrice']),'change':float(x['priceChangePercent'])} for x in data if x['symbol'].endswith('USDT')]
    rows.sort(key=lambda x:x['volume'], reverse=True)
    return rows[:limit]

@st.cache_data(ttl=60)
def get_crypto_data(symbol='BTCUSDT', interval='1h', limit=250):
    r=requests.get(f'{BINANCE}/api/v3/klines', params={'symbol':symbol,'interval':interval,'limit':limit}, timeout=15)
    r.raise_for_status(); data=r.json()
    if not data or isinstance(data, dict): return pd.DataFrame()
    df=pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume','close_time','qa_vol','trades','taker_buy_base','taker_buy_quote','ignore'])
    for c in ['open','high','low','close','volume']: df[c]=df[c].astype(float)
    df['timestamp']=pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
    return df

@st.cache_data(ttl=600)
def get_fear_greed():
    try:
        r=requests.get(FG_URL, params={'limit':1,'format':'json'}, timeout=10, headers={'User-Agent':'Mozilla/5.0'})
        d=r.json()['data'][0]
        return int(d['value']), d['value_classification']
    except:
        return None,'N/A'

@st.cache_data(ttl=600)
def get_vix():
    try:
        r=requests.get(VIX_YF, params={'range':'1d','interval':'1d'}, timeout=10, headers={'User-Agent':'Mozilla/5.0'})
        meta=r.json()['chart']['result'][0]['meta']
        return meta.get('regularMarketPrice')
    except:
        return None

def ema(series, period): return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta=series.diff(); gain=delta.where(delta>0,0).rolling(period).mean(); loss=(-delta.where(delta<0,0)).rolling(period).mean(); rs=gain/loss.replace(0,np.nan); return 100-(100/(1+rs))

def macd(series, fast=12, slow=26, signal=9):
    f=series.ewm(span=fast, adjust=False).mean(); s=series.ewm(span=slow, adjust=False).mean(); m=f-s; sig=m.ewm(span=signal, adjust=False).mean(); h=m-sig; return m,sig,h

def atr(df, period=14):
    h,l,c=df['high'],df['low'],df['close']; tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1); return tr.rolling(period).mean()

def adx(df, period=14):
    h,l,c=df['high'],df['low'],df['close']; up=h.diff(); dn=-l.diff(); pdm=up.where((up>dn)&(up>0),0.0); mdm=dn.where((dn>up)&(dn>0),0.0); tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1); atr_=tr.rolling(period).mean(); pdi=100*(pdm.rolling(period).mean()/atr_.replace(0,np.nan)); mdi=100*(mdm.rolling(period).mean()/atr_.replace(0,np.nan)); dx=100*((pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)); return dx.rolling(period).mean(), pdi, mdi

def bollinger(series, period=20, mult=2):
    mid=series.rolling(period).mean(); std=series.rolling(period).std(); return mid+mult*std, mid-mult*std, mid

def supertrend(df, period=10, mult=3.0):
    a=atr(df,period); hl2=(df['high']+df['low'])/2; upper=hl2+mult*a; lower=hl2-mult*a; fup=upper.copy(); flo=lower.copy(); st=pd.Series(np.nan,index=df.index); d=pd.Series(1,index=df.index)
    for i in range(1,len(df)):
        fup.iloc[i]=min(upper.iloc[i], fup.iloc[i-1]) if df['close'].iloc[i-1]<=fup.iloc[i-1] else upper.iloc[i]
        flo.iloc[i]=max(lower.iloc[i], flo.iloc[i-1]) if df['close'].iloc[i-1]>=flo.iloc[i-1] else lower.iloc[i]
        d.iloc[i]=1 if df['close'].iloc[i]>fup.iloc[i-1] else -1 if df['close'].iloc[i]<flo.iloc[i-1] else d.iloc[i-1]
        st.iloc[i]=flo.iloc[i] if d.iloc[i]==1 else fup.iloc[i]
    return st,d

def compute(df):
    df=df.copy(); c=df['close']
    df['EMA20']=ema(c,20); df['EMA50']=ema(c,50); df['EMA200']=ema(c,200)
    df['RSI']=rsi(c); df['MACD'],df['MACD_SIGNAL'],df['MACD_HIST']=macd(c)
    df['ATR']=atr(df); df['ADX'],df['DI+'],df['DI-']=adx(df)
    df['BB_UP'],df['BB_DN'],df['BB_MID']=bollinger(c); df['ST'],df['ST_DIR']=supertrend(df)
    return df

def score(df):
    last=df.iloc[-1]; close=float(last['close']); rsi_v=float(last['RSI']) if pd.notna(last['RSI']) else 50; adx_v=float(last['ADX']) if pd.notna(last['ADX']) else 0; macd_v=float(last['MACD']) if pd.notna(last['MACD']) else 0; sig_v=float(last['MACD_SIGNAL']) if pd.notna(last['MACD_SIGNAL']) else 0; hist_v=float(last['MACD_HIST']) if pd.notna(last['MACD_HIST']) else 0; st_dir=int(last['ST_DIR']) if pd.notna(last['ST_DIR']) else 0; atr_v=float(last['ATR']) if pd.notna(last['ATR']) else None
    ls=ss=0; lr=[]; sr=[]
    if close>last['EMA20']>last['EMA50']>last['EMA200']: ls+=25; lr.append('EMA rialziste')
    if close<last['EMA20']<last['EMA50']<last['EMA200']: ss+=25; sr.append('EMA ribassiste')
    if rsi_v>=55: ls+=10; lr.append('RSI > 55')
    elif rsi_v<=45: ss+=10; sr.append('RSI < 45')
    if macd_v>sig_v and hist_v>0: ls+=15; lr.append('MACD bullish')
    if macd_v<sig_v and hist_v<0: ss+=15; sr.append('MACD bearish')
    if adx_v>=18:
        if last['DI+']>last['DI-']: ls+=10; lr.append('ADX trend up')
        elif last['DI-']>last['DI+']: ss+=10; sr.append('ADX trend down')
    if st_dir==1: ls+=15; lr.append('SuperTrend bullish')
    elif st_dir==-1: ss+=15; sr.append('SuperTrend bearish')
    if pd.notna(last['volume']) and last['volume']>df['volume'].rolling(20).mean().iloc[-1]: ls+=5; ss+=5
    regime_ok = adx_v>=18 and pd.notna(df['volume'].rolling(20).mean().iloc[-1]) and last['volume']>0.8*df['volume'].rolling(20).mean().iloc[-1]
    if not regime_ok: return 'NO TRADE', 0, ['Regime debole'], close, adx_v, rsi_v, atr_v
    if ls>=ss+15 and ls>=50: return 'LONG', min(100,ls), lr, close, adx_v, rsi_v, atr_v
    if ss>=ls+15 and ss>=50: return 'SHORT', min(100,ss), sr, close, adx_v, rsi_v, atr_v
    return 'WATCH', max(ls,ss), ['Bias non allineato'], close, adx_v, rsi_v, atr_v

def levels(price, atr_v, direction):
    atr_v = atr_v if atr_v else price*0.01
    if direction=='LONG': return price-1.5*atr_v, price+1.5*atr_v, price+2.5*atr_v
    if direction=='SHORT': return price+1.5*atr_v, price-1.5*atr_v, price-2.5*atr_v
    return None,None,None

def badge(final):
    c=STATUS_COLORS.get(final,'#94a3b8')
    return f"<span class='status-pill'><span class='status-dot' style='background:{c}'></span><span style='color:{c}'>{final}</span></span>"

def chart_explain(row):
    final=row['final']
    if final=='LONG':
        return ['Prezzo sopra EMA20/50/200', 'Momentum positivo sopra zero', 'RSI e MACD favorevoli', 'SuperTrend rialzista e ADX presente']
    if final=='SHORT':
        return ['Prezzo sotto EMA20/50/200', 'Momentum negativo sotto zero', 'RSI e MACD deboli', 'SuperTrend ribassista e ADX presente']
    if final=='WATCH':
        return ['Alcune conferme ma non allineate', 'Trend presente ma non abbastanza forte', 'Momentum misto', 'Aspetta breakout o pullback migliore']
    return ['Regime troppo debole', 'Volume scarso o ADX basso', 'Direzionalità non chiara', 'Meglio attendere']

def render_chart(df, row, interval='1h'):
    fig=make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=[0.54,0.12,0.11,0.11,0.12], subplot_titles=['Prezzo','Volume','Momentum','RSI','MACD'])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], increasing_line_color='#2dd4bf', decreasing_line_color='#fb7185', increasing_fillcolor='#2dd4bf', decreasing_fillcolor='#fb7185', whiskerwidth=0.4, name='Price'), row=1, col=1)
    for col, color, width in [('EMA20','#f59e0b',2.4),('EMA50','#60a5fa',1.8),('EMA200','#fda4af',1.8)]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=color, width=width)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_UP'], name='BB Upper', line=dict(color='rgba(148,163,184,0.42)', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_DN'], name='BB Lower', line=dict(color='rgba(148,163,184,0.42)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(148,163,184,0.05)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ST'], name='SuperTrend', line=dict(color='#22c55e', width=2.6)), row=1, col=1)
    last = df.iloc[-1]
    x_last = df.index[-1]
    fig.add_trace(go.Scatter(x=[x_last], y=[last['close']], mode='markers+text', text=[row['final']], textposition='top center', marker=dict(size=14, color=STATUS_COLORS[row['final']], line=dict(color='white', width=1)), name='Signal'), row=1, col=1)
    price = row['price']
    atrv = row['price']*0.01 if row['sl'] is None else abs(row['price']-row['sl'])/1.5
    sl, tp1, tp2 = levels(price, atrv, row['final'])
    for lv, name, color, dash in [(price,'Entry','#f59e0b','solid'),(sl,'SL','#ef4444','dash'),(tp1,'TP1','#22c55e','dot'),(tp2,'TP2','#60a5fa','dot')]:
        if lv is not None and np.isfinite(lv):
            fig.add_hline(y=lv, line_color=color, line_width=1.5, line_dash=dash, row=1, col=1)
            fig.add_annotation(x=x_last, y=lv, text=name, showarrow=False, xshift=36, font=dict(size=10, color=color), row=1, col=1)
    vol_colors = ['#2dd4bf' if c>=o else '#fb7185' for c,o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=vol_colors, opacity=0.68), row=2, col=1)
    momentum = (df['close'] - df['EMA20']) / df['EMA20'] * 100
    fig.add_trace(go.Scatter(x=df.index, y=momentum, name='Momentum %', line=dict(color='#a78bfa', width=1.8), fill='tozeroy', fillcolor='rgba(167,139,250,0.08)'), row=3, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.20)', row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#d8b4fe', width=2)), row=4, col=1)
    for level, color in [(70,'rgba(251,113,133,.45)'),(30,'rgba(45,212,191,.45)'),(50,'rgba(255,255,255,.20)')]:
        fig.add_hline(y=level, line_dash='dash', line_color=color, row=4, col=1)
    hist_colors = ['#2dd4bf' if v>=0 else '#fb7185' for v in df['MACD_HIST']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], name='MACD Hist', marker_color=hist_colors, opacity=0.5), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#60a5fa', width=1.6)), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], name='Signal', line=dict(color='#f59e0b', width=1.6)), row=5, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.20)', row=5, col=1)
    fig.update_layout(template='plotly_dark', height=1180, margin=dict(l=18,r=18,t=72,b=18), legend=dict(orientation='h', y=1.03, x=0.01, font=dict(size=11), bgcolor='rgba(0,0,0,0)'), title=dict(text=f'{row["pair"]} · {interval}', font=dict(size=18)), hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(rangeslider_visible=False, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.10)', zeroline=False)
    return fig

def main():
    st.markdown('<div class="jarvis-hero"><div class="jarvis-title"><span style="font-size:1.15em;line-height:1">₿</span><span style="padding:0 14px;white-space:nowrap">Cripto Opportunity</span><span style="font-size:1.15em;line-height:1">Ξ</span></div><div class="jarvis-sub">Grafico chiaro e professionale · entry/SL/TP visualizzati · lettura del motore spiegata</div></div>', unsafe_allow_html=True)
    fg_val, fg_lbl = get_fear_greed(); vix_val = get_vix(); top = get_top_crypto_pairs(100)
    if not top: st.stop()
    big25, alt75 = top[:25], top[25:100]
    stablecoins = {'USDT','USDC','DAI','FDUSD','TUSD','USDE','PYUSD','FRAX','LUSD','GUSD','USDD','USDP','USTC','EURS','EURC','RLUSD','BUSD','USD1'}
    allowed_paired = {'EURUSDT'}
    def is_excluded(sym):
        base = sym.replace('USDT','')
        if sym in allowed_paired:
            return False
        if base in stablecoins:
            return True
        return False

    def is_low_vol(sym):
        base = sym.replace('USDT','')
        low_vol = {'USDC','FDUSD','TUSD','DAI','USDE','PYUSD','GUSD','EURC','EURS','USDP','BUSD','USDD','LUSD'}
        return base in low_vol


    with st.sidebar:
        st.markdown('### Setup')
        group = st.radio('Gruppo', ['Top 25','Altre 75'], index=0)
        vol_filter = st.radio('Filtro vol', ['Solo volatili', 'Anche low vol'], index=0)
        interval = st.selectbox('Timeframe', ['5m','15m','1h','4h','12h','1d'], index=2)
        show_states = st.multiselect('Mostra stati', STATUS_ORDER, default=STATUS_ORDER)
        scan = st.button('Scansiona', type='primary', use_container_width=True)
        st.caption('Stable escluse sempre, tranne EUR/USDT.')

    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Fear & Greed', f'{fg_val if fg_val is not None else "N/A"}', fg_lbl)
    c2.metric('VIX', f'{vix_val:.1f}' if vix_val else 'N/A')
    c3.metric('Top 25', len(big25))
    c4.metric('Altre 75', len(alt75))

    if not scan and 'scanner_results' not in st.session_state:
        st.info('Premi Scansiona per vedere i risultati.')
        return

    if scan or 'scanner_results' not in st.session_state:
        raw_list = big25 if group=='Top 25' else alt75
        scan_list = [x for x in raw_list if not is_excluded(x['symbol'])]
        if vol_filter == 'Solo volatili':
            scan_list = [x for x in scan_list if not is_low_vol(x['symbol'])]
        results=[]; progress = st.progress(0)
        for i, item in enumerate(scan_list, 1):
            try:
                df = get_crypto_data(item['symbol'], interval, 250)
                if df.empty or len(df) < 80: continue
                df = compute(df)
                final, conf, reasons, price, adx_v, rsi_v, atr_v = score(df)
                sl, tp1, tp2 = levels(price, atr_v, final)
                results.append({'pair': item['symbol'], 'final': final, 'confidence': conf, 'price': price, 'rsi': rsi_v, 'adx': adx_v, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'reasons': reasons, 'df': df})
            except Exception as e:
                results.append({'pair': item['symbol'], 'final': 'NO TRADE', 'confidence': 0, 'price': np.nan, 'rsi': np.nan, 'adx': np.nan, 'sl': None, 'tp1': None, 'tp2': None, 'reasons': [str(e)], 'df': pd.DataFrame()})
            progress.progress(i/len(scan_list))
        st.session_state.scanner_results = results
        st.session_state.scanner_group = group
    results = st.session_state.scanner_results
    results = [r for r in results if r['final'] in show_states]
    results.sort(key=lambda x: (x['final'] not in ('LONG','SHORT'), -x['confidence']))

    counts = {k: sum(1 for r in results if r['final']==k) for k in STATUS_ORDER}
    m1,m2,m3,m4 = st.columns(4)
    m1.markdown(f"<div class='soft-card'><div class='small-muted' style='color:{STATUS_COLORS['LONG']}'>LONG</div><div style='font-size:24px;font-weight:800;color:{STATUS_COLORS['LONG']}'>{counts['LONG']}</div></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='soft-card'><div class='small-muted' style='color:{STATUS_COLORS['SHORT']}'>SHORT</div><div style='font-size:24px;font-weight:800;color:{STATUS_COLORS['SHORT']}'>{counts['SHORT']}</div></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='soft-card'><div class='small-muted' style='color:{STATUS_COLORS['WATCH']}'>WATCH</div><div style='font-size:24px;font-weight:800;color:{STATUS_COLORS['WATCH']}'>{counts['WATCH']}</div></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='soft-card'><div class='small-muted' style='color:{STATUS_COLORS['NO TRADE']}'>NO TRADE</div><div style='font-size:24px;font-weight:800;color:{STATUS_COLORS['NO TRADE']}'>{counts['NO TRADE']}</div></div>", unsafe_allow_html=True)

    st.markdown('### Analisi selezionata')
    left, right = st.columns([2.15, 1])
    with left:
        selected = st.selectbox('Seleziona coppia', [r['pair'] for r in results], index=0 if results else None)
        if selected:
            row = next(r for r in results if r['pair'] == selected)
            if not row['df'].empty:
                st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
                st.markdown("""<div class='section-title'>Leggenda grafico</div><div class='legend-chip'><span class='legend-dot' style='background:#2dd4bf'></span>Candele rialziste</div><div class='legend-chip'><span class='legend-dot' style='background:#fb7185'></span>Candele ribassiste</div><div class='legend-chip'><span class='legend-dot' style='background:#f59e0b'></span>EMA20</div><div class='legend-chip'><span class='legend-dot' style='background:#60a5fa'></span>EMA50</div><div class='legend-chip'><span class='legend-dot' style='background:#fda4af'></span>EMA200</div><div class='legend-chip'><span class='legend-dot' style='background:#22c55e'></span>SuperTrend</div>""", unsafe_allow_html=True)
                st.plotly_chart(render_chart(row['df'], row, interval), use_container_width=True)
                st.markdown(f"<div style='margin-top:8px' class='small-muted'>Il grafico evidenzia prezzo, trend, momentum e livelli operativi del motore.</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f"<div class='result-row' style='border-color:{STATUS_COLORS[row['final']]}'><div>{badge(row['final'])}</div><div style='font-size:22px;font-weight:800;color:#e2e8f0'>{row['confidence']}/100</div></div>", unsafe_allow_html=True)
            sl_txt = f"{row['sl']:.4f}" if row.get('sl') is not None else 'N/A'
            tp1_txt = f"{row['tp1']:.4f}" if row.get('tp1') is not None else 'N/A'
            tp2_txt = f"{row['tp2']:.4f}" if row.get('tp2') is not None else 'N/A'
            st.write(f"**SL:** {sl_txt} | **TP1:** {tp1_txt} | **TP2:** {tp2_txt}")
            st.caption(' · '.join(row['reasons'][:4]))
    with right:
        st.markdown('### Lettura motore')
        row = next((r for r in results if r['pair'] == selected), results[0] if results else None)
        if row:
            st.markdown(f"<div class='explain-box'><div style='font-weight:800;margin-bottom:6px'>Cosa sta dicendo il motore</div>{''.join([f'• {x}<br>' for x in chart_explain(row)])}</div>", unsafe_allow_html=True)
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            bias = 'rialzista' if row['final']=='LONG' else 'ribassista' if row['final']=='SHORT' else 'laterale'
            st.markdown(f"<div class='explain-box'><div style='font-weight:800;margin-bottom:6px'>Lettura veloce del setup</div><b>Stato:</b> <span style='color:{STATUS_COLORS[row['final']]}'>{row['final']}</span><br><b>Bias:</b> {bias}<br><b>RSI:</b> {row['rsi']:.1f} · <b>ADX:</b> {row['adx']:.1f}<br><b>Confidenza:</b> {row['confidence']}/100</div>", unsafe_allow_html=True)
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            if row['final'] in ('LONG','SHORT'):
                st.markdown(f"<div class='explain-box'><div style='font-weight:800;margin-bottom:6px'>Livelli chiave</div><b>Entry:</b> {row['price']:.4f}<br><b>SL:</b> {row['sl']:.4f}<br><b>TP1:</b> {row['tp1']:.4f}<br><b>TP2:</b> {row['tp2']:.4f}</div>", unsafe_allow_html=True)
        st.markdown('### Snapshot')
        for r in results[:8]:
            c = STATUS_COLORS[r['final']]
            st.markdown(f"<div class='result-row' style='border-color:{c}'><div><div style='font-weight:800;color:{c}'>{r['pair']}</div><div class='small-muted'>RSI {r['rsi']:.1f} · ADX {r['adx']:.1f}</div></div><div style='font-weight:800;color:{c}'>{r['final']} {r['confidence']}/100</div></div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
