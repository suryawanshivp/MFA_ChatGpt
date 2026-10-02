import json
import math
import re
import time
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import requests, time, math, io
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="MF Research Tool", page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")

MFAPI = "https://api.mfapi.in"
RF = 0.065
MIN_TRACK_YEARS = 2.5
MIN_AUM_CR = 5.0
MIN_PEERS = 5
MAX_FETCH = 30

# ---------- styling ----------
st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1500px}
.small{font-size:.82rem;color:#777}
.metric-card{padding:.7rem 1rem;border:1px solid #ddd;border-radius:10px}
.badge{display:inline-block;padding:.18rem .5rem;border-radius:999px;font-size:.75rem;font-weight:600}
.emerging{background:#fff3cd;color:#856404}
.good{background:#d1e7dd;color:#0f5132}
.warn{background:#fff3cd;color:#664d03}
.bad{background:#f8d7da;color:#842029}
</style>
""", unsafe_allow_html=True)

# ---------- data/cache ----------
@st.cache_data(ttl=86400, show_spinner=False)
def mf_list():
    r=requests.get(f"{MFAPI}/mf", timeout=20)
    r.raise_for_status()
    x=r.json()
    if not isinstance(x,list):
        raise ValueError("MFapi /mf returned an unexpected response.")
    return pd.DataFrame(x)

@st.cache_data(ttl=86400, show_spinner=False)
def mf_search(q):
    r=requests.get(f"{MFAPI}/mf/search", params={"q":q}, timeout=20)
    r.raise_for_status()
    x=r.json()
    return pd.DataFrame(x if isinstance(x,list) else [])

@st.cache_data(ttl=86400, show_spinner=False)
def mf_history(code):
    r=requests.get(f"{MFAPI}/mf/{int(code)}", timeout=30)
    r.raise_for_status()
    x=r.json()
    if x.get("status") not in (None,"SUCCESS"):
        raise ValueError(f"MFapi returned status={x.get('status')}")
    meta=x.get("meta",{})
    data=x.get("data",[])
    if not data:
        raise ValueError("No NAV history returned.")
    d=pd.DataFrame(data)
    d["date"]=pd.to_datetime(d["date"],dayfirst=True,errors="coerce")
    d["nav"]=pd.to_numeric(d["nav"],errors="coerce")
    d=d.dropna(subset=["date","nav"]).query("nav > 0").drop_duplicates("date").sort_values("date")
    return meta,d

def classify(name, cat):
    s=(str(name)+" "+str(cat)).lower()
    if "hybrid" in s:
        family="Hybrid"
    elif "equity" in s or "elss" in s:
        family="Equity"
    else:
        return None
    # Use MFapi's category wording where available; normalize into a stable subcategory.
    raw=str(cat)
    raw=re.sub(r"^(Equity Scheme|Hybrid Scheme)\s*-\s*","",raw,flags=re.I).strip()
    return family, raw if raw and raw.lower()!="nan" else "Other"

def active_growth(meta):
    n=str(meta.get("scheme_name","")).lower()
    bad=["index","etf","fund of fund","fof","idcw","dividend","bonus","interval"]
    return ("direct" in n and "growth" in n and not any(x in n for x in bad))

def annualized_cagr(nav, years):
    if len(nav)<2:return np.nan
    end=nav.iloc[-1]; target=nav.index[-1]-pd.DateOffset(years=years)
    before=nav.loc[nav.index<=target]
    if before.empty:return np.nan
    start=before.iloc[-1]
    actual=(nav.index[-1]-before.index[-1]).days/365.25
    if actual < years*0.98:return np.nan
    return (end/start)**(1/actual)-1

def monthly_returns(nav):
    x=nav.resample("ME").last().dropna()
    return x.pct_change().dropna()

def rolling_cagr_series(nav, years):
    m=nav.resample("ME").last().dropna()
    if len(m)<years*12+1:return pd.Series(dtype=float)
    vals=[]
    for i in range(years*12,len(m)):
        a=m.iloc[i-years*12]; b=m.iloc[i]
        vals.append((b/a)**(1/years)-1)
    return pd.Series(vals,index=m.index[years*12:])

def max_dd(nav):
    peak=nav.cummax()
    return (nav/peak-1).min()

def metrics_for(nav, bench_nav):
    daily=nav.pct_change().dropna()
    bm_daily=bench_nav.pct_change().dropna()
    common=pd.concat([daily,bm_daily],axis=1,join="inner").dropna()
    common.columns=["fund","bench"]
    mu=common["fund"].mean()*252
    vol=common["fund"].std(ddof=1)*np.sqrt(252)
    sharpe=(mu-RF)/vol if vol>0 else np.nan
    downside=common["fund"].copy()
    down_dev=np.sqrt(np.mean(np.minimum(downside-RF/252,0)**2))*np.sqrt(252)
    sortino=(mu-RF)/down_dev if down_dev>0 else np.nan
    beta=common["fund"].cov(common["bench"])/common["bench"].var() if common["bench"].var()>0 else np.nan
    alpha=(mu-RF-beta*((common["bench"].mean()*252)-RF)) if pd.notna(beta) else np.nan
    fm=monthly_returns(nav); bm=monthly_returns(bench_nav)
    cm=pd.concat([fm,bm],axis=1,join="inner").dropna(); cm.columns=["fund","bench"]
    neg=cm[cm["bench"]<0]
    capture=(neg["fund"].sum()/neg["bench"].sum()*100) if len(neg) and neg["bench"].sum()!=0 else np.nan
    win=(cm.tail(36)["fund"]>cm.tail(36)["bench"]).mean()*100 if len(cm)>=12 else np.nan
    return {
        "1Y Return": annualized_cagr(nav,1), "3Y Return":annualized_cagr(nav,3),
        "5Y Return":annualized_cagr(nav,5), "Volatility":vol,
        "Sharpe":sharpe,"Sortino":sortino,"Max Drawdown":max_dd(nav),
        "Alpha":alpha,"Beta":beta,"Downside Capture":capture,"Win Rate":win,
        "Months":len(cm.tail(36))
    }

def normalize(s, higher=True):
    s=pd.to_numeric(s,errors="coerce")
    lo,hi=s.quantile(.05),s.quantile(.95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:
        return pd.Series(50,index=s.index)
    z=((s.clip(lo,hi)-lo)/(hi-lo))*100
    return z if higher else 100-z

def score(df):
    # 100 points. Returns 40, risk 25, risk-adjusted 20, relative/consistency 15.
    out=pd.DataFrame(index=df.index)
    out["r1"]=normalize(df["1Y Return"])
    out["r3"]=normalize(df["3Y Return"])
    out["r5"]=normalize(df["5Y Return"])
    out["vol"]=normalize(df["Volatility"],False)
    out["dd"]=normalize(df["Max Drawdown"])
    out["sh"]=normalize(df["Sharpe"])
    out["so"]=normalize(df["Sortino"])
    out["alpha"]=normalize(df["Alpha"])
    # Beta: modest distance from 1 is neutral; do not reward leverage.
    out["beta"]=100-(df["Beta"]-1).abs().clip(0,1)*100
    out["cap"]=normalize(df["Downside Capture"],False)
    out["win"]=normalize(df["Win Rate"])
    df["Score"]=(
        .10*out.r1+.15*out.r3+.15*out.r5+
        .08*out.vol+.17*out.dd+
        .10*out.sh+.10*out.so+
        .05*out.alpha+.03*out.beta+.04*out.cap+.03*out.win
    )
    return df

def percentile_tier(rank_pct):
    if rank_pct <= 25:return "In Form"
    if rank_pct <= 50:return "On Track"
    if rank_pct <= 75:return "Off Track"
    return "Out of Form"

# ---------- session ----------
if "last_refresh" not in st.session_state: st.session_state.last_refresh=None
if "watchlist" not in st.session_state: st.session_state.watchlist=[]
if "universe" not in st.session_state: st.session_state.universe=None

# ---------- header ----------
c1,c2,c3=st.columns([5,1.2,1.2])
with c1:
    st.title("📊 Mutual Fund Research")
    st.caption("Category-relative research • Equity & Hybrid • NAV-driven • manual refresh")
with c2:
    if st.button("🔄 Refresh data",use_container_width=True):
        mf_list.clear(); mf_search.clear(); mf_history.clear()
        st.session_state.last_refresh=datetime.now().astimezone()
        st.rerun()
with c3:
    dark=st.toggle("Dark",value=False)
    if dark: st.markdown("<style>body{background:#111;color:#eee}</style>",unsafe_allow_html=True)

try:
    schemes=mf_list()
except Exception as e:
    st.error(f"DATA FETCH ERROR — MFapi scheme list failed: {type(e).__name__}: {e}. Retry with Refresh data. If it persists, inspect MFapi availability/network access.")
    st.stop()

schemes["schemeCode"]=pd.to_numeric(schemes["schemeCode"],errors="coerce")
schemes=schemes.dropna(subset=["schemeCode"])
schemes["schemeCode"]=schemes["schemeCode"].astype(int)
schemes["family_sub"]=schemes.apply(lambda r: classify(r.get("schemeName"),r.get("schemeCategory")),axis=1)
schemes=schemes[schemes["family_sub"].notna()].copy()
schemes["Family"]=schemes["family_sub"].str[0]
schemes["Subcategory"]=schemes["family_sub"].str[1]
schemes["ActiveGrowth"]=schemes.apply(lambda r: active_growth({"scheme_name":r["schemeName"]}),axis=1)

# ---------- controls ----------
st.subheader("Research universe")
a,b,c,d=st.columns(4)
with a: family=st.selectbox("Asset class",["Equity","Hybrid"])
with b:
    subs=sorted(schemes.loc[schemes.Family==family,"Subcategory"].dropna().unique())
    sub=st.selectbox("Subcategory",subs)
with c: maxfunds=st.slider("Peer sample size",5,MAX_FETCH,15)
with d:
    q=st.text_input("Find a fund",placeholder="e.g. HDFC Flexi Cap")

cand=schemes[(schemes.Family==family)&(schemes.Subcategory==sub)].copy()
if q:
    cand=cand[cand.schemeName.str.contains(q,case=False,na=False)]
cand=cand[cand.ActiveGrowth].drop_duplicates("schemeCode").sort_values("schemeName")

selected=st.multiselect("Funds to analyze (direct-growth active funds)",cand.schemeCode.tolist(),
                        format_func=lambda x: cand.loc[cand.schemeCode==x,"schemeName"].iloc[0] if len(cand.loc[cand.schemeCode==x]) else str(x),
                        max_selections=maxfunds)

# Optional AUM input
with st.expander("Eligibility data — 6-month average AUM (AMFI CSV)",expanded=False):
    st.write("MFapi does not expose scheme-level 6-month AUM. Upload a CSV with columns: scheme_code, month (YYYY-MM), aum_cr. The app averages the latest six available months.")
    up=st.file_uploader("Upload AUM CSV",type=["csv"])
    aum=None
    if up:
        try:
            aum=pd.read_csv(up)
            req={"scheme_code","month","aum_cr"}
            if not req.issubset(aum.columns): raise ValueError(f"Missing columns: {sorted(req-set(aum.columns))}")
            aum["scheme_code"]=pd.to_numeric(aum.scheme_code,errors="coerce").astype("Int64")
            aum["month"]=pd.to_datetime(aum.month,errors="coerce")
            aum["aum_cr"]=pd.to_numeric(aum.aum_cr,errors="coerce")
            aum=aum.dropna(subset=["scheme_code","month","aum_cr"]).sort_values("month")
            st.success(f"Loaded {len(aum):,} AUM observations.")
        except Exception as e:
            st.error(f"AUM INPUT ERROR — {type(e).__name__}: {e}. Correct the CSV and upload again.")

if len(selected)<MIN_PEERS:
    st.info(f"Select at least {MIN_PEERS} active-fund peers to calculate category-relative ratings/rankings. Current selection: {len(selected)}.")
    st.stop()

# ---------- fetch + calculate ----------
rows=[]; errors=[]
progress=st.progress(0)
for i,code in enumerate(selected):
    try:
        meta,nav=mf_history(code)
        rows.append({"code":code,"name":meta.get("scheme_name",str(code)),"nav":nav,"meta":meta})
    except Exception as e:
        errors.append(f"{code}: {type(e).__name__}: {e}")
    progress.progress((i+1)/len(selected))
progress.empty()

if errors:
    st.warning("Some funds could not be fetched. They were excluded from calculations. " + " | ".join(errors))

if len(rows)<MIN_PEERS:
    st.error(f"CALCULATION BLOCKED — only {len(rows)} valid histories remain; at least {MIN_PEERS} are required. Retry failed fetches or select more peers.")
    st.stop()

# Build equal-weight monthly category benchmark from each fund's monthly returns.
mrets=[]
for x in rows:
    m=monthly_returns(x["nav"]).rename(x["code"])
    mrets.append(m)
monthly=pd.concat(mrets,axis=1).sort_index()
bench_m=monthly.mean(axis=1,skipna=True).dropna()
bench_nav=(1+bench_m).cumprod()
# Align benchmark to daily NAV via month-end values for daily risk stats.
bench_daily=bench_nav.reindex(pd.date_range(bench_nav.index.min(),bench_nav.index.max(),freq="D")).ffill()

result=[]
latest_dates=[]
for x in rows:
    nav=x["nav"].set_index("date")["nav"]
    met=metrics_for(nav,bench_daily)
    age=(nav.index.max()-nav.index.min()).days/365.25
    r={"Code":x["code"],"Fund":x["name"],"Track Record":age,"Emerging":age<3}
    r.update(met)
    result.append(r); latest_dates.append(nav.index.max())
df=pd.DataFrame(result)

# AUM eligibility
df["6M Avg AUM"]=np.nan
if aum is not None:
    latest_month=aum.month.max()
    months=sorted(aum.loc[aum.month<=latest_month,"month"].unique())[-6:]
    am=aum[aum.month.isin(months)].groupby("scheme_code").aum_cr.mean()
    df["6M Avg AUM"]=df.Code.map(am)

df["Eligible"]=(df["Track Record"]>=MIN_TRACK_YEARS)&(df["6M Avg AUM"]>=MIN_AUM_CR)
df["Eligibility"]="Eligible" 
if aum is None: df["Eligibility"]="AUM data required"
df.loc[df["Track Record"]<MIN_TRACK_YEARS,"Eligibility"]="Track record <2.5Y"
if aum is not None: df.loc[df["6M Avg AUM"]<MIN_AUM_CR,"Eligibility"]="AUM <₹5cr"

eligible=df[df.Eligible].copy() if aum is not None else df.iloc[0:0].copy()
if len(eligible)>=MIN_PEERS:
    eligible=score(eligible)
    eligible["Rank"]=eligible["Score"].rank(method="min",ascending=False).astype(int)
    eligible["Percentile"]=(eligible["Score"].rank(pct=True,ascending=False)*100).round(1)
    eligible["Tier"]=eligible["Percentile"].apply(lambda x: percentile_tier(100-x+0.0001))
else:
    eligible["Score"]=np.nan; eligible["Rank"]=np.nan; eligible["Percentile"]=np.nan; eligible["Tier"]="Not rated"

latest=max(latest_dates).date()
fetched=datetime.now().astimezone()
st.caption(f"Latest NAV in loaded universe: **{latest}** • Calculated/fetched: **{fetched.strftime('%Y-%m-%d %H:%M %Z')}** • Category benchmark: equal-weighted monthly returns")

# ---------- dashboard ----------
m1,m2,m3,m4=st.columns(4)
m1.metric("Funds loaded",len(df))
m2.metric("Eligible for rating",len(eligible) if aum is not None else "AUM needed")
m3.metric("Category",f"{family} / {sub}")
m4.metric("Risk-free rate","6.5%")

st.subheader("Leaderboard")
display_cols=["Fund","Score","Rank","1Y Return","3Y Return","5Y Return","Volatility","Sharpe","Sortino","Max Drawdown","Win Rate","Eligibility"]
show=(eligible if len(eligible) else df).copy()
for col in ["1Y Return","3Y Return","5Y Return","Volatility","Max Drawdown"]:
    show[col]=show[col].map(lambda x:f"{x:.1%}" if pd.notna(x) else "—")
for col in ["Sharpe","Sortino","Win Rate","Score"]:
    if col in show: show[col]=show[col].map(lambda x:f"{x:.1f}" if pd.notna(x) else "—")
st.dataframe(show[[c for c in display_cols if c in show.columns]],use_container_width=True,hide_index=True)

l,r=st.columns(2)
with l:
    st.subheader("Return vs volatility")
    chart=eligible if len(eligible) else df
    if len(chart):
        fig=px.scatter(chart,x="Volatility",y="3Y Return",text="Fund",size="Score" if "Score" in chart else None,
                       hover_data=["1Y Return","Max Drawdown","Sharpe","Win Rate"])
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig,use_container_width=True)
with r:
    st.subheader("Category dispersion — 3Y return")
    chart=eligible if len(eligible) else df
    if len(chart):
        fig=px.histogram(chart,x="3Y Return",nbins=12)
        st.plotly_chart(fig,use_container_width=True)

# ---------- compare ----------
st.subheader("Compare up to 4 funds")
compare_names=st.multiselect("Select funds",df.Fund.tolist(),max_selections=4)
if compare_names:
    cm=df[df.Fund.isin(compare_names)].copy()
    radar_metrics=["1Y Return","3Y Return","5Y Return","Sharpe","Sortino","Max Drawdown","Win Rate"]
    vals=pd.DataFrame(index=cm.Fund)
    for met in radar_metrics:
        vals[met]=normalize(cm[met],higher=met not in ["Max Drawdown"])
    fig=go.Figure()
    for name in vals.index:
        v=vals.loc[name].tolist(); fig.add_trace(go.Scatterpolar(r=v+[v[0]],theta=radar_metrics+[radar_metrics[0]],fill="toself",name=name))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,100])),showlegend=True)
    st.plotly_chart(fig,use_container_width=True)

# ---------- watchlist ----------
st.subheader("Watchlist")
watch_candidates=df.Fund.tolist()
wadd=st.multiselect("Add funds to watchlist",watch_candidates,default=[x for x in st.session_state.watchlist if x in watch_candidates])
st.session_state.watchlist=wadd
if wadd:
    w=df[df.Fund.isin(wadd)].copy()
    if len(eligible):
        w=w.merge(eligible[["Fund","Score","Rank","Percentile","Tier"]],on="Fund",how="left")
    st.dataframe(w[["Fund","Score","Rank","Tier","3Y Return","Volatility","Max Drawdown","Win Rate","Eligibility"]],use_container_width=True,hide_index=True)
    st.caption("Tier percentiles are computed within the eligible peer set: top 25% In Form; 25–50% On Track; 50–75% Off Track; bottom 25% Out of Form.")

# ---------- methodology ----------
with st.expander("Methodology / validation rules"):
    st.markdown("""
**Eligibility**
- Track record ≥ 2.5 years.
- 6-month average AUM ≥ ₹5 crore.
- At least 5 eligible active-fund peers in the same MFapi category/subcategory.
- Funds under 3 years receive **Emerging** status but are not automatically rated.

**Returns**
- 1Y/3Y/5Y = annualized trailing CAGR using the nearest available NAV on/before the target date.
- Rolling return distributions are available from the monthly NAV series; the headline values above are trailing values.

**Risk**
- Volatility = standard deviation of daily NAV returns × √252.
- Sharpe = (annualized return − 6.5%) / annualized volatility.
- Sortino = (annualized return − 6.5%) / annualized downside deviation, with daily MAR = 6.5%/252.
- Maximum drawdown = minimum NAV/previous peak − 1.

**Relative metrics**
- Benchmark = equal-weighted monthly return index of the loaded peer set in the selected category/subcategory.
- Beta = covariance(fund, benchmark)/variance(benchmark) on overlapping daily returns.
- Alpha = fund annualized excess return − beta × benchmark annualized excess return.
- Downside capture = sum fund returns in benchmark-negative months / sum benchmark returns in those months × 100.
- Win rate = percentage of the last 36 overlapping months in which fund return > category benchmark return.

**Score (0–100)**
- 1Y return 10%; 3Y 15%; 5Y 15%.
- Volatility 8%; maximum drawdown 17%.
- Sharpe 10%; Sortino 10%.
- Alpha 5%; beta stability 3%; downside capture 4%; 36-month win rate 3%.
- Each component is cross-sectional percentile-normalized using 5th–95th percentile winsorization. Higher is better except volatility, downside capture and drawdown (less negative drawdown is better).
- The score is a research ranking, not a prediction or investment recommendation.

**Important data limitation**
MFapi provides NAV history and scheme metadata, but not scheme-level 6-month AUM. AUM eligibility therefore requires the AMFI-derived CSV input. No fund is silently treated as AUM-eligible when that data is missing.
""")

if st.session_state.last_refresh:
    st.caption(f"Manual refresh completed at {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M %Z')}. Cached responses are reused for 24 hours unless Refresh data is pressed.")
