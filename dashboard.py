"""Live dashboard over the request log. Run with: streamlit run dashboard.py"""

import os
import sqlite3
import time

import pandas as pd
import streamlit as st

DB = os.getenv("SENTINEL_DB", "data/requests.db")
st.set_page_config(page_title="Sentinel", page_icon="🛡️", layout="wide")
st.title("Sentinel gateway")

window = st.sidebar.selectbox("Window", {"Last 15 min": 900, "Last hour": 3600, "Last 24 h": 86400, "Last 7 days": 604800}.items(), index=1, format_func=lambda x: x[0])[1]
if st.sidebar.button("Refresh"):
    st.rerun()

if not os.path.exists(DB):
    st.info(f"No request log at {DB} yet. Send a request through the gateway first.")
    st.stop()

conn = sqlite3.connect(DB)
df = pd.read_sql_query("SELECT * FROM requests WHERE ts >= ?", conn, params=(time.time() - window,))
if df.empty:
    st.info("No requests in this window.")
    st.stop()
df["time"] = pd.to_datetime(df["ts"], unit="s")
ok = df[df.outcome == "ok"]

c = st.columns(6)
c[0].metric("Requests", len(df))
c[1].metric("p95 latency", f"{ok.latency_ms.quantile(.95):.0f} ms" if len(ok) else "-")
c[2].metric("Cache hit rate", f"{df.cache.isin(['exact', 'semantic']).mean():.0%}")
c[3].metric("Blocked", int(df.outcome.str.startswith("blocked").sum()))
c[4].metric("Fallbacks", int(df.fallback_used.sum()))
c[5].metric("Est. cost", f"${df.est_cost_usd.sum():.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Requests per minute by outcome")
    per_min = df.set_index("time").groupby("outcome").resample("1min").size().unstack(0).fillna(0)
    st.bar_chart(per_min)
with right:
    st.subheader("Latency (ok requests)")
    if len(ok):
        st.line_chart(ok.set_index("time")[["latency_ms", "upstream_ms", "guard_ms"]].resample("1min").median())

left, right = st.columns(2)
with left:
    st.subheader("By app")
    st.dataframe(df.groupby("app").agg(requests=("request_id", "count"), p95_ms=("latency_ms", lambda s: s.quantile(.95)),
                                       tokens=("completion_tokens", "sum"), pii=("pii_redacted", "sum"),
                                       blocked=("outcome", lambda s: s.str.startswith("blocked").sum())).round(1))
with right:
    st.subheader("Guard activity")
    st.dataframe(df[df.injection_rules != ""].groupby("injection_rules").size().rename("count").sort_values(ascending=False))

st.subheader("Recent requests")
st.caption("Prompts and completions are never stored; only metadata.")
st.dataframe(df.sort_values("ts", ascending=False).head(100)[
    ["time", "app", "outcome", "status", "cache", "served_model", "fallback_used", "latency_ms", "prompt_tokens",
     "completion_tokens", "pii_redacted", "injection_score"]])
