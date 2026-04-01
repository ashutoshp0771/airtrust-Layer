"""
TrustLayer AI — Real Application
Enterprise AI Reliability Platform powered by Anthropic Claude
Canada Hackathon 2026
"""

from __future__ import annotations
import io
import json
import time
from datetime import datetime
from typing import Optional

from fpdf import FPDF

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trustlayer.detector import TrustLayerDetector
from trustlayer.industries import INDUSTRIES, get_industry
from trustlayer.models import AnalysisRequest
from trustlayer.cross_validator import CrossValidator, CrossValidationResult

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrustLayer AI | Enterprise AI Reliability",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #0066FF;
    --teal:    #00B894;
    --red:     #E74C3C;
    --orange:  #F39C12;
    --green:   #27AE60;
    --dark:    #1E293B;
    --gray:    #64748B;
    --border:  #E2E8F0;
    --bg:      #F8FAFC;
}

.stApp { background: var(--bg); font-family: 'Plus Jakarta Sans', sans-serif; }
#MainMenu, footer, [data-testid="stHeader"] { visibility: hidden; }

/* ── Tab styling — ensure tabs are always visible ── */
button[data-baseweb="tab"] {
    visibility: visible !important;
    opacity: 1 !important;
    color: var(--gray) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 10px 20px !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
}
button[data-baseweb="tab"]:hover {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
    font-weight: 700 !important;
}
[data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1px solid var(--border) !important;
    background: transparent !important;
}
[data-baseweb="tab-panel"] {
    visibility: visible !important;
    opacity: 1 !important;
    padding-top: 16px !important;
}
[data-baseweb="tab-highlight"] {
    background-color: var(--primary) !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
    border-right: 1px solid var(--border);
}

/* Action badges */
.badge-pass  { background:#DCFCE7; color:#166534; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1rem; display:inline-block; }
.badge-flag  { background:#FEF9C3; color:#854D0E; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1rem; display:inline-block; }
.badge-block { background:#FEE2E2; color:#991B1B; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1rem; display:inline-block; }

/* Score circle */
.score-circle {
    width:130px; height:130px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    flex-direction:column; margin:0 auto;
    font-family:'Plus Jakarta Sans', sans-serif;
}

/* Issue/claim pills */
.issue-high   { background:#FEE2E2; color:#991B1B; border-left:3px solid #E74C3C; padding:6px 12px; border-radius:6px; margin:4px 0; font-size:.9rem; }
.issue-medium { background:#FEF9C3; color:#854D0E; border-left:3px solid #F39C12; padding:6px 12px; border-radius:6px; margin:4px 0; font-size:.9rem; }
.issue-low    { background:#DCFCE7; color:#166534; border-left:3px solid #22C55E; padding:6px 12px; border-radius:6px; margin:4px 0; font-size:.9rem; }
.issue-info   { background:#EFF6FF; color:#1D4ED8; border-left:3px solid #0066FF; padding:6px 12px; border-radius:6px; margin:4px 0; font-size:.9rem; }

/* Response box */
.response-box {
    background:#ffffff; border:1px solid var(--border); border-radius:10px;
    padding:16px; font-size:.95rem; line-height:1.7; color:var(--dark);
    max-height:280px; overflow-y:auto;
}

/* Step indicator */
.step-active   { color:#0066FF; font-weight:600; }
.step-done     { color:#27AE60; font-weight:600; }
.step-pending  { color:#94A3B8; }

/* Metric card */
.metric-card {
    background:#fff; border:1px solid var(--border); border-radius:10px;
    padding:14px 18px; text-align:center;
}
.metric-card h2 { margin:0; font-size:1.8rem; }
.metric-card p  { margin:0; color:var(--gray); font-size:.85rem; }

/* Header */
.app-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 100%);
    padding: 20px 28px; border-radius: 12px; margin-bottom: 24px;
    display:flex; align-items:center; gap:16px;
}

/* ── Login page ── */
.login-wrap {
    min-height: 100vh;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #0F172A 0%, #0D2952 50%, #0F172A 100%);
}
.login-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 48px 40px 40px;
    width: 420px;
    backdrop-filter: blur(16px);
    box-shadow: 0 25px 60px rgba(0,0,0,0.5);
}
.login-title {
    color: #fff; font-size: 1.55rem; font-weight: 800;
    text-align: center; margin-bottom: 4px;
}
.login-sub {
    color: #94A3B8; font-size: .88rem; text-align: center; margin-bottom: 32px;
}
.sso-btn {
    display: flex; align-items: center; justify-content: center;
    gap: 10px; width: 100%; padding: 11px 0;
    border-radius: 10px; border: 1px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.06); color: #E2E8F0;
    font-size: .9rem; font-weight: 600; cursor: pointer;
    margin-bottom: 10px; transition: background .2s;
}
.sso-btn:hover { background: rgba(255,255,255,0.12); }
.divider-line {
    display: flex; align-items: center; gap: 12px;
    color: #475569; font-size: .82rem; margin: 20px 0;
}
.divider-line::before, .divider-line::after {
    content: ""; flex: 1; height: 1px; background: rgba(255,255,255,0.08);
}

/* Animated logo */
@keyframes pulse-ring {
    0%   { transform: scale(0.9); opacity: 0.6; }
    70%  { transform: scale(1.15); opacity: 0; }
    100% { transform: scale(0.9); opacity: 0; }
}
@keyframes scan-line {
    0%   { transform: translateY(-22px); opacity: 0; }
    20%  { opacity: 1; }
    80%  { opacity: 1; }
    100% { transform: translateY(22px); opacity: 0; }
}
@keyframes shield-glow {
    0%, 100% { filter: drop-shadow(0 0 6px rgba(0,102,255,0.5)); }
    50%       { filter: drop-shadow(0 0 18px rgba(0,102,255,0.9)); }
}
.logo-svg { animation: shield-glow 3s ease-in-out infinite; }
.pulse-ring { animation: pulse-ring 2.5s ease-out infinite; transform-origin: center; }
.scan-line  { animation: scan-line 2.8s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_cv_result" not in st.session_state:
    st.session_state.last_cv_result = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "login_error" not in st.session_state:
    st.session_state.login_error = ""
if "review_queue" not in st.session_state:
    st.session_state.review_queue = []        # FLAG items waiting for human review
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []           # completed reviewer decisions
if "blocked_responses" not in st.session_state:
    st.session_state.blocked_responses = []   # BLOCK items with what-user-saw record
if "_queue_counter" not in st.session_state:
    st.session_state._queue_counter = 0
if "_selected_flow_node" not in st.session_state:
    st.session_state._selected_flow_node = None

# Transfer pending preset query into the widget key BEFORE the widget renders
if "_pending_query" in st.session_state:
    st.session_state["query_input"] = st.session_state.pop("_pending_query")


# ── Login credentials (from secrets or demo fallback) ─────────────────────────
def _get_demo_users() -> dict:
    try:
        return dict(st.secrets["demo_users"])
    except Exception:
        return {"admin": "trustlayer2026", "demo": "demo123", "judge": "hackathon2026"}


DYNAMIC_LOGO_SVG = """
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" class="logo-svg">
  <!-- Pulse ring -->
  <circle cx="32" cy="32" r="28" fill="none" stroke="#0066FF" stroke-width="2" class="pulse-ring" opacity="0.5"/>
  <!-- Shield body -->
  <path d="M32 6 L54 15 L54 32 C54 44 43 54 32 58 C21 54 10 44 10 32 L10 15 Z"
        fill="url(#shieldGrad)" stroke="#0066FF" stroke-width="1.5"/>
  <!-- Scan line (clipped inside shield) -->
  <clipPath id="shieldClip">
    <path d="M32 6 L54 15 L54 32 C54 44 43 54 32 58 C21 54 10 44 10 32 L10 15 Z"/>
  </clipPath>
  <rect x="10" y="31" width="44" height="2" fill="#00D4FF" opacity="0.7"
        class="scan-line" clip-path="url(#shieldClip)"/>
  <!-- TL text -->
  <text x="32" y="37" text-anchor="middle" font-family="Arial,sans-serif"
        font-weight="900" font-size="16" fill="#ffffff" letter-spacing="1">TL</text>
  <defs>
    <linearGradient id="shieldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="#1E3A5F"/>
      <stop offset="100%" stop-color="#0F172A"/>
    </linearGradient>
  </defs>
</svg>
"""

LOGO_SMALL_SVG = """
<svg width="36" height="36" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" class="logo-svg">
  <path d="M32 6 L54 15 L54 32 C54 44 43 54 32 58 C21 54 10 44 10 32 L10 15 Z"
        fill="url(#sg2)" stroke="#0066FF" stroke-width="1.5"/>
  <text x="32" y="37" text-anchor="middle" font-family="Arial,sans-serif"
        font-weight="900" font-size="16" fill="#ffffff" letter-spacing="1">TL</text>
  <defs>
    <linearGradient id="sg2" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="#1E3A5F"/>
      <stop offset="100%" stop-color="#0F172A"/>
    </linearGradient>
  </defs>
</svg>
"""


# ── Login page ─────────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    st.markdown("""
    <style>
    #MainMenu, footer, header, [data-testid="stSidebar"] { display: none !important; }
    .stApp { background: linear-gradient(135deg,#0F172A 0%,#0D2952 50%,#0F172A 100%) !important; }
    </style>""", unsafe_allow_html=True)

    # Centered login card (3-col layout trick)
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='text-align:center;margin-bottom:8px'>{DYNAMIC_LOGO_SVG}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='login-title'>TrustLayer AI</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='login-sub'>Enterprise AI Reliability Platform<br>"
            "Canada Hackathon 2026</div>",
            unsafe_allow_html=True,
        )

        # SSO buttons (demo — auto-fill admin credentials)
        ms_col, gg_col = st.columns(2)
        with ms_col:
            if st.button("🪟  Sign in with Microsoft", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.user_name = "Microsoft User"
                st.session_state.login_error = ""
                st.rerun()
        with gg_col:
            if st.button("🔵  Sign in with Google", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.user_name = "Google User"
                st.session_state.login_error = ""
                st.rerun()

        st.markdown("<div class='divider-line'>or sign in with credentials</div>",
                    unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter your username", key="login_user")
        password = st.text_input("Password", placeholder="Enter your password",
                                 type="password", key="login_pass")

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

        if st.button("Sign In", type="primary", use_container_width=True):
            users = _get_demo_users()
            if username in users and users[username] == password:
                st.session_state.authenticated = True
                st.session_state.user_name = username
                st.session_state.login_error = ""
                st.rerun()
            else:
                st.session_state.login_error = "Invalid username or password."
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;color:#475569;font-size:.78rem'>"
            "Demo accounts: <code>admin / trustlayer2026</code> &nbsp;|&nbsp; "
            "<code>demo / demo123</code> &nbsp;|&nbsp; <code>judge / hackathon2026</code>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.stop()  # Halt — don't render the main app until authenticated


# ── Helper: get detector (cached per API key) ─────────────────────────────────
@st.cache_resource
def get_detector(api_key: str) -> TrustLayerDetector:
    return TrustLayerDetector(api_key=api_key)


@st.cache_resource
def get_cross_validator(anthropic_key: str, openai_key: str) -> CrossValidator:
    return CrossValidator(anthropic_key=anthropic_key, openai_key=openai_key)


# ── Helper: confidence gauge ──────────────────────────────────────────────────
def confidence_gauge(confidence: float, risk: float, action: str) -> go.Figure:
    color_map = {"PASS": "#27AE60", "FLAG": "#F39C12", "BLOCK": "#E74C3C"}
    color = color_map.get(action, "#64748B")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
            "bar":   {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],  "color": "#FEE2E2"},
                {"range": [50, 75], "color": "#FEF9C3"},
                {"range": [75, 100],"color": "#DCFCE7"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 4},
                "thickness": 0.8,
                "value": confidence,
            },
        },
        title={"text": "Confidence Score", "font": {"size": 14, "color": "#64748B"}},
    ))
    fig.update_layout(
        height=220, margin=dict(t=40, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Helper: technique bar chart ───────────────────────────────────────────────
def technique_chart(scores_dict: dict) -> go.Figure:
    labels = list(scores_dict.keys())
    values = list(scores_dict.values())
    colors = ["#27AE60" if v >= 75 else "#F39C12" if v >= 50 else "#E74C3C" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=12)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Helper: PDF report generator ─────────────────────────────────────────────
def generate_pdf(result, cv_result=None) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # Header
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 8)
    pdf.cell(0, 10, "TrustLayer AI - Analysis Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(15, 19)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Canada Hackathon 2026", ln=True)
    pdf.set_text_color(30, 41, 59)
    pdf.ln(6)

    # Decision banner
    action_colors = {"PASS": (220, 252, 231), "FLAG": (254, 249, 195), "BLOCK": (254, 226, 226)}
    r, g, b = action_colors.get(result.action, (241, 245, 249))
    pdf.set_fill_color(r, g, b)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 12, f"Decision: {result.action}  |  Confidence: {result.confidence_score:.1f}%  |  Risk: {result.risk_score:.1f}/100", ln=True, fill=True)
    pdf.ln(2)

    # Query & Industry
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Query", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, result.query)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 7, "Industry:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, result.industry, ln=True)
    pdf.ln(2)

    # Scores table
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Detection Technique Scores", ln=True)
    pdf.set_font("Helvetica", "", 9)
    scores = result.scores.as_dict()
    col_w = 90
    for i, (k, v) in enumerate(scores.items()):
        if i % 2 == 0:
            pdf.set_x(15)
        label = k.replace("_", " ").title()
        bar_fill = int(v * 1.8)
        pdf.cell(col_w, 6, f"{label}: {v:.0f}%")
        if i % 2 == 1:
            pdf.ln()
    pdf.ln(4)

    # Explanation
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "TrustLayer Explanation", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, result.explanation or "—")
    pdf.ln(3)

    # Issues
    if result.issues or result.fabrication_indicators:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Issues Detected", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for issue in result.issues:
            pdf.multi_cell(0, 5, f"  - {issue}")
        for fi in result.fabrication_indicators:
            pdf.multi_cell(0, 5, f"  - [Fabrication] {fi}")
        pdf.ln(2)

    # Claims
    if result.claims:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, f"Extracted Claims ({len(result.claims)})", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for c in result.claims[:10]:  # cap at 10 to avoid overflow
            pdf.multi_cell(0, 5, f"  [{c.risk.upper()}] {c.text}")
        pdf.ln(2)

    # Cross-validation section
    if cv_result:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Claude + GPT-4o Cross-Validation", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Agreement Score: {cv_result.agreement_pct}%  ({cv_result.agreement_label})", ln=True)
        pdf.cell(0, 6, f"Consensus Decision: {cv_result.consensus_action}  |  Consensus Confidence: {cv_result.consensus_confidence:.1f}%", ln=True)
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "GPT-4o Result", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, f"Action: {cv_result.openai_result.action}  |  Confidence: {cv_result.openai_result.confidence_score:.1f}%  |  Risk: {cv_result.openai_result.risk_score:.1f}", ln=True)
        pdf.multi_cell(0, 5, f"Explanation: {cv_result.openai_result.explanation or '—'}")
        pdf.ln(2)
        if cv_result.disagreement_signals:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Disagreement Signals", ln=True)
            pdf.set_font("Helvetica", "", 9)
            for sig in cv_result.disagreement_signals:
                pdf.multi_cell(0, 5, f"  - {sig}")

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px'>"
        f"{LOGO_SMALL_SVG}"
        f"<div><div style='font-weight:800;font-size:1.05rem;color:#1E293B'>TrustLayer AI</div>"
        f"<div style='font-size:.75rem;color:#64748B'>Enterprise AI Reliability</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    # User info + logout
    st.markdown(
        f"<div style='background:#EFF6FF;border-radius:8px;padding:7px 10px;"
        f"font-size:.82rem;color:#1D4ED8;display:flex;justify-content:space-between;"
        f"align-items:center;margin-bottom:4px'>"
        f"<span>👤 <strong>{st.session_state.user_name}</strong></span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if st.button("Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_name = ""
        st.session_state.history = []
        st.session_state.last_result = None
        st.session_state.last_cv_result = None
        st.rerun()
    st.divider()

    # Anthropic API Key
    api_key = ""
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        st.success("✅ Anthropic key loaded")
    except Exception:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Get your key at console.anthropic.com",
        )

    # OpenAI API Key (optional — for cross-validation)
    openai_key = ""
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
        st.success("✅ OpenAI key loaded")
    except Exception:
        openai_key = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            placeholder="sk-...",
            help="Required for Claude + GPT-4o cross-validation",
        )

    st.divider()

    # Industry selector
    industry_names = list(INDUSTRIES.keys())
    industry = st.selectbox(
        "Industry Module",
        industry_names,
        index=0,
        format_func=lambda x: f"{INDUSTRIES[x]['icon']}  {x}",
    )
    industry_cfg = get_industry(industry)
    st.caption(industry_cfg["description"])

    st.divider()

    # Settings
    with st.expander("⚙️ Detection Settings"):
        self_consistency = st.toggle(
            "Self-consistency check",
            value=False,
            help="Runs 2 extra Claude calls to test response stability. More accurate, slower.",
        )
        use_grounding = st.toggle(
            "Enable industry grounding",
            value=True,
            help="Compares response against authoritative industry data.",
        )
        use_cross_validation = st.toggle(
            "Cross-validate with GPT-4o",
            value=bool(openai_key),
            disabled=not bool(openai_key),
            help="Run detection independently through Claude AND GPT-4o. Agreement score boosts reliability.",
        )
        if use_cross_validation and not openai_key:
            st.caption("Enter your OpenAI key above to enable cross-validation.")

    st.divider()
    st.caption("**Canada Hackathon 2026**")
    st.caption("Detect · Prevent · Govern")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""<div class="app-header">
  {DYNAMIC_LOGO_SVG}
  <div>
    <div style="color:#fff;font-size:1.5rem;font-weight:800;line-height:1.2">TrustLayer AI</div>
    <div style="color:#94A3B8;font-size:.92rem">Real-time hallucination detection · Claude + GPT-4o · Canada Hackathon 2026</div>
    <div style="color:#64748B;font-size:.78rem;margin-top:3px">Signed in as <strong style="color:#7DD3FC">{st.session_state.user_name}</strong></div>
  </div>
</div>""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_detect, tab_history, tab_batch, tab_enterprise, tab_review, tab_howto = st.tabs([
    "🔍  Live Detection",
    "📊  Session History",
    "🚀  Batch Test",
    "🏦  Enterprise Flow",
    "👥  Review Queue",
    "📖  How It Works",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: LIVE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
with tab_detect:

    # ── Query input section ───────────────────────────────────────────────────
    col_input, col_scenario = st.columns([2, 1])

    with col_input:
        st.markdown("#### Your Query")
        query = st.text_area(
            label="query_input",
            label_visibility="collapsed",
            placeholder="Type a question to send to the AI, or load a preset scenario →",
            height=120,
            key="query_input",
        )

    with col_scenario:
        st.markdown("#### Load Preset Scenario")
        scenarios = industry_cfg.get("scenarios", [])
        scenario_labels = [s["label"] for s in scenarios]
        selected_label = st.selectbox(
            "scenario_select",
            label_visibility="collapsed",
            options=["— select —"] + scenario_labels,
        )
        if selected_label != "— select —":
            chosen = next(s for s in scenarios if s["label"] == selected_label)
            # Enhancement 1: show expected outcome badge
            if selected_label.startswith("✅"):
                st.markdown(
                    "<div style='background:#DCFCE7;color:#166534;border-radius:8px;"
                    "padding:6px 10px;font-size:.82rem;font-weight:600;margin-bottom:6px'>"
                    "Expected outcome: <strong>PASS</strong> — high confidence, low risk</div>",
                    unsafe_allow_html=True,
                )
            elif selected_label.startswith("⚠️"):
                st.markdown(
                    "<div style='background:#FEF9C3;color:#854D0E;border-radius:8px;"
                    "padding:6px 10px;font-size:.82rem;font-weight:600;margin-bottom:6px'>"
                    "Expected outcome: <strong>FLAG / BLOCK</strong> — hallucination risk</div>",
                    unsafe_allow_html=True,
                )
            if chosen["query"] and st.button("Load →", use_container_width=True):
                st.session_state["_pending_query"] = chosen["query"]
                st.rerun()

        # Show what grounding data will be used
        if use_grounding and industry_cfg.get("grounding_context"):
            with st.expander("📋 Grounding data active"):
                st.code(industry_cfg["grounding_context"], language=None)

    # Custom grounding context
    with st.expander("📎 Paste your own grounding context (optional)"):
        custom_context = st.text_area(
            "custom_ctx",
            label_visibility="collapsed",
            placeholder="Paste authoritative data here — product specs, policy docs, clinical guidelines…",
            height=100,
        )

    # ── Analyze button ────────────────────────────────────────────────────────
    st.markdown("")
    run_col, info_col = st.columns([1, 3])
    with run_col:
        analyze_btn = st.button(
            "🔍  Analyze with TrustLayer",
            type="primary",
            use_container_width=True,
            disabled=not api_key or not query,
        )
    with info_col:
        if not api_key:
            st.warning("⚠️ Enter your Anthropic API key in the sidebar to begin.")
        elif not query:
            st.info("💬 Type a query or load a preset scenario above.")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if analyze_btn and api_key and query:
        # Determine grounding context
        context = None
        if custom_context.strip():
            context = custom_context.strip()
        elif use_grounding:
            context = industry_cfg.get("grounding_context") or None

        request = AnalysisRequest(
            query=query,
            industry=industry,
            context=context,
            self_consistency_check=self_consistency,
        )

        # Choose pipeline: cross-validation or single Claude
        run_cv = use_cross_validation and bool(openai_key)

        # Progress steps
        progress_placeholder = st.empty()
        if run_cv:
            steps = [
                ("Claude: generating response...", "⏳"),
                ("Claude: detection analysis...",  "⏳"),
                ("GPT-4o: cross-validation...",    "⏳"),
                ("Computing consensus decision...", "⏳"),
            ]
        else:
            steps = [
                ("Sending query to Claude...",           "⏳"),
                ("Extracting claims and citations...",   "⏳"),
                ("Running 8 detection algorithms...",    "⏳"),
                ("Computing confidence + risk scores...", "⏳"),
            ]

        with progress_placeholder.container():
            step_cols = st.columns(len(steps))
            for i, (label, icon) in enumerate(steps):
                step_cols[i].markdown(
                    f"<div class='step-pending'>{icon} {label}</div>",
                    unsafe_allow_html=True,
                )

        def update_step(i: int, done: bool = False):
            icon = "✅" if done else "🔄"
            cls  = "step-done" if done else "step-active"
            step_cols[i].markdown(
                f"<div class='{cls}'>{icon} {steps[i][0]}</div>",
                unsafe_allow_html=True,
            )

        try:
            if run_cv:
                # Cross-validation pipeline
                cv = get_cross_validator(api_key, openai_key)
                update_step(0)
                # run() handles generate + both analyses internally
                # We update steps incrementally via timing approximation
                update_step(1)
                update_step(2)
                cv_result = cv.run(request)
                update_step(0, done=True)
                update_step(1, done=True)
                update_step(2, done=True)
                update_step(3, done=True)

                result = cv_result.claude_result  # primary result for display
                st.session_state.last_cv_result = cv_result
                st.session_state.last_result = result

                # Use consensus for history
                history_action = cv_result.consensus_action
                history_conf   = cv_result.consensus_confidence
                history_risk   = cv_result.consensus_risk
            else:
                # Single Claude pipeline
                detector = get_detector(api_key)
                update_step(0)
                llm_response = detector.generate_response(request)
                update_step(0, done=True)
                update_step(1)
                update_step(2)
                result = detector.analyze(request, llm_response)
                update_step(1, done=True)
                update_step(2, done=True)
                update_step(3, done=True)

                st.session_state.last_result = result
                st.session_state.last_cv_result = None
                history_action = result.action
                history_conf   = result.confidence_score
                history_risk   = result.risk_score

            st.session_state.history.append({
                "time":       datetime.now().strftime("%H:%M:%S"),
                "industry":   industry,
                "query":      query[:80] + ("…" if len(query) > 80 else ""),
                "action":     history_action,
                "confidence": history_conf,
                "risk":       history_risk,
                "issues":     len(result.issues),
                "validated":  "Yes" if run_cv else "No",
            })

            # ── Route to review queue or blocked log ──────────────────────────
            # Use Claude's raw individual action (result.action) for review-queue
            # routing — not the consensus. This catches the common case where
            # Claude says FLAG but cross-validation disagreement escalates it to
            # BLOCK: the human reviewer must still see the item.
            SAFE_FALLBACK = (
                "I'm sorry, I'm not able to provide specific details on that right now. "
                "Please speak with one of our advisors who can give you accurate, "
                "personalised guidance based on your situation."
            )
            raw_claude_action = result.action  # Claude's own verdict before consensus
            escalated_to_block = (raw_claude_action == "FLAG" and history_action == "BLOCK")

            if raw_claude_action == "FLAG":
                # Always queue FLAG items — including ones escalated to BLOCK by
                # cross-validation disagreement (escalated flag shown in the card)
                st.session_state._queue_counter += 1
                st.session_state.review_queue.append({
                    "id":                   st.session_state._queue_counter,
                    "time":                 datetime.now().strftime("%H:%M:%S"),
                    "industry":             industry,
                    "query":                query,
                    "ai_response":          result.llm_response,
                    "confidence":           round(result.confidence_score, 1),
                    "risk":                 round(result.risk_score, 1),
                    "consensus_action":     history_action,
                    "consensus_confidence": round(history_conf, 1),
                    "escalated":            escalated_to_block,
                    "issues":               result.issues,
                    "fabrication":          result.fabrication_indicators,
                    "explanation":          result.explanation,
                    "reviewer":             st.session_state.user_name,
                    "status":               "pending",
                    "reviewer_note":        "",
                })

            if history_action == "BLOCK":
                # Log all BLOCKs (direct blocks + disagreement-escalated flags)
                st.session_state.blocked_responses.append({
                    "time":         datetime.now().strftime("%H:%M:%S"),
                    "industry":     industry,
                    "query":        query,
                    "ai_response":  result.llm_response,
                    "user_saw":     SAFE_FALLBACK,
                    "confidence":   round(history_conf, 1),
                    "risk":         round(history_risk, 1),
                    "issues":       result.issues,
                    "fabrication":  result.fabrication_indicators,
                    "explanation":  result.explanation,
                    "escalated":    escalated_to_block,
                })

        except Exception as e:
            progress_placeholder.empty()
            st.error(f"❌ Analysis failed: {e}")
            st.stop()

        progress_placeholder.empty()

    # ── Cross-validation panel ────────────────────────────────────────────────
    cv_result: Optional[CrossValidationResult] = st.session_state.get("last_cv_result")
    if cv_result:
        st.markdown("---")
        st.markdown("### Claude + GPT-4o Cross-Validation")

        # Agreement badge + consensus
        agree_col, claude_col, gpt_col = st.columns([1, 1.5, 1.5])

        with agree_col:
            pct = cv_result.agreement_pct
            color = cv_result.agreement_color
            st.markdown(
                f"<div style='background:#fff;border:2px solid {color};border-radius:12px;"
                f"padding:16px;text-align:center'>"
                f"<div style='font-size:2rem;font-weight:800;color:{color}'>{pct}%</div>"
                f"<div style='font-size:.85rem;color:{color};font-weight:600'>{cv_result.agreement_label}</div>"
                f"<hr style='margin:8px 0'>"
                f"<div style='font-size:.8rem;color:#64748B'>Agreement Score</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            # Consensus badge
            badge_cls = f"badge-{cv_result.consensus_action.lower()}"
            st.markdown(
                f"<div style='text-align:center'>"
                f"<div style='font-size:.8rem;color:#64748B;margin-bottom:4px'>Consensus Decision</div>"
                f"<div class='{badge_cls}'>{cv_result.consensus_action}</div>"
                f"<div style='font-size:.8rem;color:#64748B;margin-top:4px'>"
                f"Confidence: {cv_result.consensus_confidence:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with claude_col:
            st.markdown("**Score Comparison: Claude vs GPT-4o**")
            claude_scores = cv_result.claude_result.scores.as_dict()
            gpt_scores    = cv_result.openai_result.scores.as_dict()
            labels = list(claude_scores.keys())

            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(
                name="Claude",
                x=labels,
                y=list(claude_scores.values()),
                marker_color="#0066FF",
                opacity=0.85,
            ))
            fig_cv.add_trace(go.Bar(
                name="GPT-4o",
                x=labels,
                y=list(gpt_scores.values()),
                marker_color="#10B981",
                opacity=0.85,
            ))
            fig_cv.update_layout(
                barmode="group",
                height=260,
                margin=dict(t=10, b=60, l=10, r=10),
                yaxis=dict(range=[0, 115], showgrid=False),
                xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_cv, use_container_width=True)

            # Enhancement 2: per-technique delta table
            with st.expander("Delta Table — technique-by-technique divergence"):
                delta_rows = []
                for k in labels:
                    c_val = claude_scores[k]
                    g_val = gpt_scores.get(k, 0)
                    diff  = c_val - g_val
                    delta_rows.append({
                        "Technique":  k.replace("_", " ").title(),
                        "Claude %":   f"{c_val:.0f}%",
                        "GPT-4o %":   f"{g_val:.0f}%",
                        "Delta":      f"{diff:+.0f}%",
                        "Agreement":  "✅ OK" if abs(diff) <= 15 else "⚠️ Diverge",
                    })
                df_delta = pd.DataFrame(delta_rows)

                def color_delta(val):
                    if "Diverge" in str(val): return "background-color:#FEF9C3;color:#854D0E;font-weight:600"
                    if "OK" in str(val):      return "background-color:#DCFCE7;color:#166534"
                    return ""

                st.dataframe(
                    df_delta.style.applymap(color_delta, subset=["Agreement"]),
                    use_container_width=True,
                    hide_index=True,
                )

        with gpt_col:
            st.markdown("**GPT-4o (OpenAI)**")
            st.metric("GPT-4o Action",      cv_result.openai_result.action)
            st.metric("GPT-4o Confidence",  f"{cv_result.openai_result.confidence_score:.1f}%")
            st.metric("GPT-4o Risk",        f"{cv_result.openai_result.risk_score:.1f}")
            st.metric("Processing",         f"{cv_result.processing_ms} ms total")

        # Disagreement signals
        if cv_result.disagreement_signals:
            with st.expander(f"⚠️ Disagreement Signals ({len(cv_result.disagreement_signals)})"):
                for sig in cv_result.disagreement_signals:
                    st.markdown(f"<div class='issue-medium'>⚡ {sig}</div>",
                                unsafe_allow_html=True)
        else:
            st.success("Both models are in strong agreement — high reliability result.")

        # GPT-4o explanation
        if cv_result.openai_result.explanation:
            with st.expander("GPT-4o Explanation"):
                st.markdown(
                    f"<div style='background:#f0fdf4;border-left:4px solid #10B981;"
                    f"padding:12px 16px;border-radius:8px;font-size:.95rem'>"
                    f"{cv_result.openai_result.explanation}</div>",
                    unsafe_allow_html=True,
                )

    # ── Display results ───────────────────────────────────────────────────────
    result = st.session_state.get("last_result")

    if result:
        st.markdown("---")

        # ── Row 1: Scores + Action ────────────────────────────────────────────
        col_gauge, col_risk, col_action, col_meta = st.columns([2, 1, 1.5, 1.5])

        with col_gauge:
            st.plotly_chart(
                confidence_gauge(result.confidence_score, result.risk_score, result.action),
                use_container_width=True,
            )

        with col_risk:
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Risk Score", f"{result.risk_score:.0f} / 100")
            st.metric("Issues Found", len(result.issues))
            st.metric("Claims Extracted", len(result.claims))

        with col_action:
            st.markdown("<br><br>", unsafe_allow_html=True)
            badge_class = f"badge-{result.action.lower()}"
            st.markdown(
                f"<div style='text-align:center'>"
                f"<div class='{badge_class}'>{result.action_emoji}  {result.action}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            action_desc = {
                "PASS":  "Response delivered to user",
                "FLAG":  "Routed to human review",
                "BLOCK": "Blocked — fallback triggered",
            }
            st.caption(action_desc.get(result.action, ""))

        with col_meta:
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Industry", result.industry.split("—")[0].strip())
            st.metric("Latency", f"{result.processing_ms or 0} ms")
            if result.citations_found:
                cite_status = "Valid" if result.citations_valid else "⚠️ Suspect"
                st.metric("Citations", f"{len(result.citations_found)} ({cite_status})")

        # ── Explanation ───────────────────────────────────────────────────────
        color_map = {"PASS": "#DCFCE7", "FLAG": "#FEF9C3", "BLOCK": "#FEE2E2"}
        border_map = {"PASS": "#22C55E", "FLAG": "#F59E0B", "BLOCK": "#EF4444"}
        bg    = color_map.get(result.action, "#F1F5F9")
        bdr   = border_map.get(result.action, "#94A3B8")

        st.markdown(
            f"<div style='background:{bg};border-left:4px solid {bdr};"
            f"padding:12px 16px;border-radius:8px;margin:8px 0;font-size:.95rem'>"
            f"<strong>TrustLayer Decision:</strong> {result.explanation}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Row 2: Issues + Technique breakdown ───────────────────────────────
        col_issues, col_chart = st.columns([1, 1])

        with col_issues:
            st.markdown("#### 🔎 Issues Detected")

            if not result.issues and not result.fabrication_indicators:
                st.success("No significant issues detected.")
            else:
                for issue in result.issues:
                    st.markdown(f"<div class='issue-high'>⚠️ {issue}</div>",
                                unsafe_allow_html=True)
                for fi in result.fabrication_indicators:
                    st.markdown(f"<div class='issue-medium'>🔴 {fi}</div>",
                                unsafe_allow_html=True)
                for ni in result.numerical_issues:
                    st.markdown(f"<div class='issue-medium'>🔢 {ni}</div>",
                                unsafe_allow_html=True)
                for ti in result.temporal_issues:
                    st.markdown(f"<div class='issue-info'>📅 {ti}</div>",
                                unsafe_allow_html=True)

        with col_chart:
            st.markdown("#### 📊 Detection Technique Scores")
            st.plotly_chart(
                technique_chart(result.scores.as_dict()),
                use_container_width=True,
            )

        # ── Row 3: Expandable details ─────────────────────────────────────────
        with st.expander("📄 Full AI Response"):
            st.markdown(
                f"<div class='response-box'>{result.llm_response}</div>",
                unsafe_allow_html=True,
            )

        if result.claims:
            with st.expander(f"📋 Extracted Claims ({len(result.claims)})"):
                claims_data = [
                    {
                        "Claim": c.text,
                        "Risk":  c.risk.upper(),
                        "Issue": c.issue or "—",
                    }
                    for c in result.claims
                ]
                df = pd.DataFrame(claims_data)

                def color_risk(val):
                    if val == "HIGH":   return "background-color:#FEE2E2;color:#991B1B"
                    if val == "MEDIUM": return "background-color:#FEF9C3;color:#854D0E"
                    return "background-color:#DCFCE7;color:#166534"

                st.dataframe(
                    df.style.applymap(color_risk, subset=["Risk"]),
                    use_container_width=True,
                    hide_index=True,
                )

        if result.citations_found:
            with st.expander(f"🔗 Citations Found ({len(result.citations_found)})"):
                for cite in result.citations_found:
                    status = "✅" if result.citations_valid else "❌ Suspect"
                    st.markdown(f"- {status} `{cite}`")

        with st.expander("🔧 Raw JSON Output"):
            raw_output = {
                "query":            result.query,
                "industry":         result.industry,
                "action":           result.action,
                "confidence_score": result.confidence_score,
                "risk_score":       result.risk_score,
                "explanation":      result.explanation,
                "scores":           result.scores.as_dict(),
                "issues":           result.issues,
                "fabrication_indicators": result.fabrication_indicators,
                "claims":           [c.model_dump() for c in result.claims],
                "citations_found":  result.citations_found,
                "citations_valid":  result.citations_valid,
                "numerical_issues": result.numerical_issues,
                "temporal_issues":  result.temporal_issues,
                "processing_ms":    result.processing_ms,
                "timestamp":        result.timestamp.isoformat(),
            }
            st.json(raw_output)

        # Enhancement 5: PDF + JSON download buttons side by side
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "⬇️  Download Result JSON",
                data=json.dumps(raw_output, indent=2),
                file_name=f"trustlayer_{result.industry.replace(' ','_')}_{result.timestamp.strftime('%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl_col2:
            cv_for_pdf = st.session_state.get("last_cv_result")
            try:
                pdf_bytes = generate_pdf(result, cv_for_pdf)
                st.download_button(
                    "📄  Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"trustlayer_{result.industry.replace(' ','_')}_{result.timestamp.strftime('%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception:
                pass  # PDF unavailable silently


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: SESSION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown("#### Session Analysis History")

    if not st.session_state.history:
        st.info("No analyses run yet this session. Go to **Live Detection** to get started.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analyzed",   len(df_hist))
        c2.metric("Blocked",  int((df_hist["action"] == "BLOCK").sum()))
        c3.metric("Flagged",  int((df_hist["action"] == "FLAG").sum()))
        c4.metric("Passed",   int((df_hist["action"] == "PASS").sum()))

        st.markdown("---")

        def color_action(val):
            if val == "BLOCK": return "background-color:#FEE2E2;color:#991B1B;font-weight:700"
            if val == "FLAG":  return "background-color:#FEF9C3;color:#854D0E;font-weight:700"
            return "background-color:#DCFCE7;color:#166534;font-weight:700"

        st.dataframe(
            df_hist.style.applymap(color_action, subset=["action"]),
            use_container_width=True,
            hide_index=True,
        )

        # Confidence trend
        if len(df_hist) > 1:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                y=df_hist["confidence"],
                mode="lines+markers",
                line=dict(color="#0066FF", width=2),
                marker=dict(size=8),
                name="Confidence %",
            ))
            fig_trend.add_hline(y=75, line_dash="dash", line_color="#27AE60",
                                annotation_text="PASS threshold (75%)")
            fig_trend.add_hline(y=50, line_dash="dash", line_color="#F39C12",
                                annotation_text="FLAG threshold (50%)")
            fig_trend.update_layout(
                title="Confidence Score Trend",
                height=280,
                yaxis=dict(range=[0, 105]),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        st.download_button(
            "⬇️  Export History CSV",
            data=df_hist.to_csv(index=False),
            file_name="trustlayer_session_history.csv",
            mime="text/csv",
        )

        if st.button("🗑️  Clear History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: BATCH TEST
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("#### Batch Test — Run All Hallucination-Risk Scenarios")
    st.markdown(
        "Runs every **⚠️ Hallucination Risk** scenario across all industry modules "
        "in a single click. Compare PASS / FLAG / BLOCK outcomes side by side."
    )

    if not api_key:
        st.warning("Enter your Anthropic API key in the sidebar to use batch testing.")
    else:
        # Build scenario list
        batch_scenarios = []
        for ind_name, ind_cfg in INDUSTRIES.items():
            for sc in ind_cfg.get("scenarios", []):
                if sc["label"].startswith("⚠️") and sc.get("query"):
                    batch_scenarios.append({
                        "industry": ind_name,
                        "label":    sc["label"],
                        "query":    sc["query"],
                    })

        st.info(f"{len(batch_scenarios)} hallucination-risk scenarios across {len(INDUSTRIES)} industries")

        run_batch = st.button(
            "🚀  Run All Hallucination-Risk Scenarios",
            type="primary",
            use_container_width=False,
            disabled=not api_key,
        )

        if run_batch:
            detector_b = get_detector(api_key)
            batch_results = []
            progress_bar = st.progress(0, text="Starting batch run...")

            for i, sc in enumerate(batch_scenarios):
                progress_bar.progress(
                    (i) / len(batch_scenarios),
                    text=f"Running: {sc['industry']} — {sc['label'][:50]}...",
                )
                try:
                    ind_cfg_b = get_industry(sc["industry"])
                    req_b = AnalysisRequest(
                        query=sc["query"],
                        industry=sc["industry"],
                        context=ind_cfg_b.get("grounding_context") or None,
                        self_consistency_check=False,
                    )
                    llm_resp_b = detector_b.generate_response(req_b)
                    res_b = detector_b.analyze(req_b, llm_resp_b)
                    batch_results.append({
                        "Industry":   sc["industry"],
                        "Scenario":   sc["label"].replace("⚠️ ", ""),
                        "Action":     res_b.action,
                        "Confidence": round(res_b.confidence_score, 1),
                        "Risk":       round(res_b.risk_score, 1),
                        "Issues":     len(res_b.issues),
                        "Latency ms": res_b.processing_ms,
                    })
                except Exception as e:
                    batch_results.append({
                        "Industry":   sc["industry"],
                        "Scenario":   sc["label"].replace("⚠️ ", ""),
                        "Action":     "ERROR",
                        "Confidence": 0,
                        "Risk":       0,
                        "Issues":     0,
                        "Latency ms": 0,
                    })

            progress_bar.progress(1.0, text="Batch complete!")
            st.session_state["batch_results"] = batch_results

        # Show results
        if "batch_results" in st.session_state and st.session_state.batch_results:
            br = st.session_state.batch_results
            df_batch = pd.DataFrame(br)

            # Summary metrics
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Scenarios Run",  len(df_batch))
            b2.metric("BLOCK",  int((df_batch["Action"] == "BLOCK").sum()))
            b3.metric("FLAG",   int((df_batch["Action"] == "FLAG").sum()))
            b4.metric("PASS",   int((df_batch["Action"] == "PASS").sum()))

            st.markdown("---")

            def color_batch_action(val):
                if val == "BLOCK": return "background-color:#FEE2E2;color:#991B1B;font-weight:700"
                if val == "FLAG":  return "background-color:#FEF9C3;color:#854D0E;font-weight:700"
                if val == "PASS":  return "background-color:#DCFCE7;color:#166534;font-weight:700"
                return "background-color:#F1F5F9;color:#64748B"

            st.dataframe(
                df_batch.style.applymap(color_batch_action, subset=["Action"]),
                use_container_width=True,
                hide_index=True,
            )

            # Action distribution chart
            action_counts = df_batch["Action"].value_counts().reset_index()
            action_counts.columns = ["Action", "Count"]
            color_seq = {"BLOCK": "#E74C3C", "FLAG": "#F39C12", "PASS": "#27AE60", "ERROR": "#94A3B8"}
            fig_dist = go.Figure(go.Bar(
                x=action_counts["Action"],
                y=action_counts["Count"],
                marker_color=[color_seq.get(a, "#94A3B8") for a in action_counts["Action"]],
                text=action_counts["Count"],
                textposition="outside",
            ))
            fig_dist.update_layout(
                title="Action Distribution — Hallucination-Risk Scenarios",
                height=260,
                yaxis=dict(showgrid=False),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            st.download_button(
                "⬇️  Export Batch Results CSV",
                data=df_batch.to_csv(index=False),
                file_name="trustlayer_batch_results.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: ENTERPRISE FLOW SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_enterprise:
    st.markdown("#### 🏦 Enterprise Banking App — Interactive Pipeline")
    st.markdown(
        "Click any stage in the pipeline to explore that persona's role, inputs, and outputs. "
        "Run a query in **Live Detection** first to see the full flow come alive."
    )

    result_ent = st.session_state.get("last_result")
    action_ent = result_ent.action if result_ent else None

    # ── Helper: flow node HTML ─────────────────────────────────────────────
    def _flow_node(label, sublabel, active=False, color="#1E293B", width="155px", selected=False):
        border = f"3px solid {color}" if active else "2px solid #E2E8F0"
        bg     = f"{color}28" if selected else (f"{color}14" if active else "#fff")
        ring   = f"box-shadow:0 0 0 4px {color}40;" if selected else ""
        text_color = color if active else "#94A3B8"
        return (
            f"<div style='border:{border};background:{bg};border-radius:12px;"
            f"padding:14px 8px;text-align:center;width:{width};flex-shrink:0;{ring}'>"
            f"<div style='font-weight:700;font-size:.88rem;color:{text_color}'>{label}</div>"
            f"<div style='font-size:.72rem;color:#64748B;margin-top:3px'>{sublabel}</div>"
            f"</div>"
        )

    def _arrow(label="", color="#94A3B8"):
        return (
            f"<div style='display:flex;flex-direction:column;align-items:center;"
            f"justify-content:center;padding:0 2px;color:{color};font-size:.72rem;flex-shrink:0'>"
            f"<span>{label}</span>"
            f"<span style='font-size:1.3rem'>→</span>"
            f"</div>"
        )

    node_colors = {"PASS": "#27AE60", "FLAG": "#F39C12", "BLOCK": "#E74C3C", None: "#94A3B8"}
    nc          = node_colors.get(action_ent, "#94A3B8")
    sel         = st.session_state._selected_flow_node
    cv_result_ent = st.session_state.get("last_cv_result")
    gpt4o_active  = bool(cv_result_ent and cv_result_ent.openai_result)

    if action_ent == "PASS":
        outcome_lbl, outcome_sub, outcome_col = "✅ Customer", "Receives response", "#27AE60"
    elif action_ent == "FLAG":
        outcome_lbl, outcome_sub, outcome_col = "⚠️ Review Queue", "Human reviewer", "#F39C12"
    elif action_ent == "BLOCK":
        outcome_lbl, outcome_sub, outcome_col = "🚫 Safe Fallback", "User sees fallback", "#E74C3C"
    else:
        outcome_lbl, outcome_sub, outcome_col = "⬜ Outcome", "Run a query first", "#94A3B8"

    flow_html = (
        "<div style='display:flex;align-items:center;gap:2px;overflow-x:auto;"
        "padding:20px 12px;background:#F8FAFC;border-radius:14px;margin:12px 0'>"
        + _flow_node("👤 Customer",    "Submits query",           active=True,              color="#0066FF", selected=(sel=="customer"))
        + _arrow("query")
        + _flow_node("🏦 Banking App", "Enterprise chatbot",      active=bool(result_ent),  color="#0066FF", selected=(sel=="bankapp"))
        + _arrow("sends to")
        + _flow_node("🤖 Claude LLM",  "Generates response",      active=bool(result_ent),  color="#7B2D8B", selected=(sel=="claude"))
        + _arrow("analyze")
        + _flow_node("🟢 GPT-4o",      "Cross-validates",         active=gpt4o_active,      color="#10A37F", selected=(sel=="gpt4o"), width="130px")
        + _arrow("consensus", nc)
        + _flow_node("🛡️ TrustLayer",  "Final decision",          active=bool(result_ent),  color="#0066FF", selected=(sel=="trustlayer"), width="135px")
        + _arrow("decision", nc)
        + _flow_node(outcome_lbl,       outcome_sub,               active=bool(result_ent),  color=outcome_col, selected=(sel=="outcome"))
        + "</div>"
    )
    st.markdown(flow_html, unsafe_allow_html=True)

    # ── Stage selector buttons ─────────────────────────────────────────────
    st.markdown("<div style='font-size:.85rem;color:#64748B;margin-bottom:6px'>👇 Click a stage to explore:</div>", unsafe_allow_html=True)
    pb1, pb2, pb3, pb4, pb5, pb6 = st.columns(6)
    with pb1:
        if st.button("👤 Customer", use_container_width=True, key="btn_ent_customer"):
            st.session_state._selected_flow_node = "customer"
            st.rerun()
    with pb2:
        if st.button("🏦 Banking App", use_container_width=True, key="btn_ent_bankapp"):
            st.session_state._selected_flow_node = "bankapp"
            st.rerun()
    with pb3:
        if st.button("🤖 Claude LLM", use_container_width=True, key="btn_ent_claude"):
            st.session_state._selected_flow_node = "claude"
            st.rerun()
    with pb4:
        if st.button("🟢 GPT-4o", use_container_width=True, key="btn_ent_gpt4o"):
            st.session_state._selected_flow_node = "gpt4o"
            st.rerun()
    with pb5:
        if st.button("🛡️ TrustLayer", use_container_width=True, key="btn_ent_trustlayer"):
            st.session_state._selected_flow_node = "trustlayer"
            st.rerun()
    with pb6:
        if st.button("⬜ Outcome", use_container_width=True, key="btn_ent_outcome"):
            st.session_state._selected_flow_node = "outcome"
            st.rerun()

    # ── Persona detail panel ───────────────────────────────────────────────
    if sel:
        st.markdown("---")

        # ── CUSTOMER ──────────────────────────────────────────────────────
        if sel == "customer":
            p_col, d_col = st.columns([1, 3])
            with p_col:
                st.markdown(
                    "<div style='background:#EFF6FF;border:2px solid #BFDBFE;border-radius:16px;"
                    "padding:24px 16px;text-align:center;height:100%'>"
                    "<div style='font-size:3rem'>👤</div>"
                    "<div style='font-weight:700;color:#1D4ED8;margin-top:8px;font-size:1.1rem'>Customer</div>"
                    "<div style='font-size:.78rem;color:#3B82F6;margin-top:4px'>Banking App User</div>"
                    "<hr style='border-color:#BFDBFE;margin:14px 0'>"
                    "<div style='font-size:.78rem;color:#475569;text-align:left;line-height:1.8'>"
                    "<b>Role:</b> End user of the banking chatbot<br>"
                    "<b>Sees:</b> A clean chat interface<br>"
                    "<b>Unaware of:</b> TrustLayer running silently<br>"
                    "<b>Protected from:</b> Hallucinated financial advice"
                    "</div></div>",
                    unsafe_allow_html=True,
                )
            with d_col:
                if result_ent:
                    st.markdown("##### 💬 What the customer asked:")
                    st.markdown(
                        f"<div style='background:#fff;border:2px solid #BFDBFE;border-radius:10px;"
                        f"padding:16px;font-size:1rem;line-height:1.6;margin-bottom:16px'>"
                        f"💬 &nbsp;<em>\"{result_ent.query}\"</em>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("##### 📨 What the customer received:")
                    if action_ent == "PASS":
                        st.markdown(
                            f"<div style='background:#F0FDF4;border:2px solid #86EFAC;border-radius:10px;"
                            f"padding:16px;font-size:.93rem;line-height:1.7'>"
                            f"✅ <b>Response delivered as-is</b><br><br>{result_ent.llm_response}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption("No intervention — TrustLayer cleared the response.")
                    elif action_ent == "FLAG":
                        st.markdown(
                            "<div style='background:#FFFBEB;border:2px solid #FCD34D;border-radius:10px;"
                            "padding:16px;font-size:.95rem;font-style:italic;color:#92400E'>"
                            "⏳ \"Thank you for your question. One of our specialists is reviewing "
                            "this for you and will respond shortly with accurate information.\""
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption("Customer is waiting. The flagged response is in the 👥 Review Queue tab.")
                    elif action_ent == "BLOCK":
                        st.markdown(
                            "<div style='background:#FFF1F2;border:2px solid #FECDD3;border-radius:10px;"
                            "padding:16px;font-size:.95rem;font-style:italic;color:#991B1B'>"
                            "🛡️ \"I'm sorry, I'm not able to provide specific details on that right now. "
                            "Please speak with one of our advisors who can give you accurate, "
                            "personalised guidance based on your situation.\""
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption("Safe fallback delivered. The harmful AI response was never shown.")
                else:
                    st.info("Run a query in **Live Detection** to see the customer's journey here.")

        # ── BANKING APP ───────────────────────────────────────────────────
        elif sel == "bankapp":
            p_col, d_col = st.columns([1, 3])
            with p_col:
                industry_label = result_ent.industry if result_ent else "BFSI Banking"
                rf_label       = str(get_industry(result_ent.industry)["risk_factor"]) + "×" if result_ent else "1.3×"
                st.markdown(
                    f"<div style='background:#EFF6FF;border:2px solid #BFDBFE;border-radius:16px;"
                    f"padding:24px 16px;text-align:center;height:100%'>"
                    f"<div style='font-size:3rem'>🏦</div>"
                    f"<div style='font-weight:700;color:#1D4ED8;margin-top:8px;font-size:1.1rem'>Banking App</div>"
                    f"<div style='font-size:.78rem;color:#3B82F6;margin-top:4px'>Enterprise Chatbot Layer</div>"
                    f"<hr style='border-color:#BFDBFE;margin:14px 0'>"
                    f"<div style='font-size:.78rem;color:#475569;text-align:left;line-height:1.8'>"
                    f"<b>Role:</b> Enterprise application layer<br>"
                    f"<b>Routes queries to:</b> Claude LLM<br>"
                    f"<b>Integrated with:</b> TrustLayer middleware<br>"
                    f"<b>Industry:</b> {industry_label}<br>"
                    f"<b>Risk multiplier:</b> {rf_label}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            with d_col:
                if result_ent:
                    st.markdown("##### 📤 Query forwarded to Claude LLM:")
                    st.markdown(
                        f"<div style='background:#fff;border:2px solid #BFDBFE;border-radius:10px;"
                        f"padding:16px;font-size:.95rem;line-height:1.6;margin-bottom:16px'>"
                        f"📤 <em>\"{result_ent.query}\"</em>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("##### 📋 Enterprise grounding context sent to TrustLayer:")
                    grounding_cfg = get_industry(result_ent.industry)
                    ctx = grounding_cfg.get("grounding_context", "No grounding context configured.")
                    st.markdown(
                        f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;"
                        f"padding:14px;font-size:.84rem;line-height:1.65;max-height:200px;overflow-y:auto'>"
                        f"📋 {ctx[:700]}{'…' if len(ctx) > 700 else ''}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Industry: {result_ent.industry} · "
                        f"Risk factor: {grounding_cfg['risk_factor']}× · "
                        f"TrustLayer processes EVERY response before delivery"
                    )
                else:
                    st.info("Run a query in **Live Detection** to see the Banking App's role.")

        # ── CLAUDE LLM ────────────────────────────────────────────────────
        elif sel == "claude":
            p_col, d_col = st.columns([1, 3])
            with p_col:
                proc = str(result_ent.processing_ms) + " ms" if result_ent and result_ent.processing_ms else "—"
                st.markdown(
                    f"<div style='background:#F5F3FF;border:2px solid #C4B5FD;border-radius:16px;"
                    f"padding:24px 16px;text-align:center;height:100%'>"
                    f"<div style='font-size:3rem'>🤖</div>"
                    f"<div style='font-weight:700;color:#6D28D9;margin-top:8px;font-size:1.1rem'>Claude LLM</div>"
                    f"<div style='font-size:.78rem;color:#7C3AED;margin-top:4px'>AI Response Generator</div>"
                    f"<hr style='border-color:#C4B5FD;margin:14px 0'>"
                    f"<div style='font-size:.78rem;color:#475569;text-align:left;line-height:1.8'>"
                    f"<b>Model:</b> claude-sonnet-4-5<br>"
                    f"<b>Role:</b> Generates raw AI response<br>"
                    f"<b>Risk:</b> Can hallucinate facts &amp; stats<br>"
                    f"<b>Output:</b> Goes to TrustLayer next<br>"
                    f"<b>Processing:</b> {proc}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            with d_col:
                if result_ent:
                    st.markdown("##### ⚡ Raw LLM output — unfiltered, before TrustLayer:")
                    st.markdown(
                        f"<div style='background:#fff;border:2px solid #C4B5FD;border-radius:10px;"
                        f"padding:16px;font-size:.9rem;line-height:1.7;max-height:270px;overflow-y:auto'>"
                        f"{result_ent.llm_response}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Characters", len(result_ent.llm_response))
                    m2.metric("Words", len(result_ent.llm_response.split()))
                    m3.metric("TrustLayer verdict", result_ent.action)
                    if action_ent == "PASS":
                        st.success("✅ This response passed all TrustLayer checks and was delivered to the customer.")
                    elif action_ent == "FLAG":
                        st.warning("⚠️ This response was flagged — sent for human review. Customer sees a holding message.")
                    elif action_ent == "BLOCK":
                        st.error("🚫 This response was BLOCKED — it was never delivered. Customer received a safe fallback.")
                else:
                    st.info("Run a query in **Live Detection** to see Claude's raw output here.")

        # ── GPT-4o ────────────────────────────────────────────────────────
        elif sel == "gpt4o":
            p_col, d_col = st.columns([1, 3])
            with p_col:
                gpt_action = cv_result_ent.openai_result.action if gpt4o_active else "—"
                gpt_conf   = f"{cv_result_ent.openai_result.confidence_score:.0f}%" if gpt4o_active else "—"
                gpt_risk   = f"{cv_result_ent.openai_result.risk_score:.0f}" if gpt4o_active else "—"
                st.markdown(
                    f"<div style='background:#F0FFF8;border:2px solid #6EE7B7;border-radius:16px;"
                    f"padding:24px 16px;text-align:center;height:100%'>"
                    f"<div style='font-size:3rem'>🟢</div>"
                    f"<div style='font-weight:700;color:#10A37F;margin-top:8px;font-size:1.1rem'>GPT-4o</div>"
                    f"<div style='font-size:.78rem;color:#059669;margin-top:4px'>OpenAI Cross-Validator</div>"
                    f"<hr style='border-color:#6EE7B7;margin:14px 0'>"
                    f"<div style='font-size:.78rem;color:#475569;text-align:left;line-height:1.8'>"
                    f"<b>Model:</b> gpt-4o<br>"
                    f"<b>Role:</b> Independent second opinion<br>"
                    f"<b>Analyzes:</b> Same response as Claude<br>"
                    f"<b>Verdict:</b> {gpt_action}<br>"
                    f"<b>Confidence:</b> {gpt_conf}<br>"
                    f"<b>Risk:</b> {gpt_risk}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            with d_col:
                if gpt4o_active:
                    cv = cv_result_ent
                    gpt = cv.openai_result

                    # Agreement banner
                    agree_pct = cv.agreement_score * 100
                    agree_color = "#27AE60" if agree_pct >= 80 else "#F39C12" if agree_pct >= 65 else "#E74C3C"
                    agree_label = "Strong agreement" if agree_pct >= 80 else "Partial agreement" if agree_pct >= 65 else "Disagreement — escalated"
                    st.markdown(
                        f"<div style='background:{agree_color}18;border:2px solid {agree_color};"
                        f"border-radius:10px;padding:14px;margin-bottom:16px;text-align:center'>"
                        f"<div style='font-size:1.5rem;font-weight:800;color:{agree_color}'>"
                        f"Agreement: {agree_pct:.0f}%</div>"
                        f"<div style='font-size:.85rem;color:{agree_color};margin-top:4px'>{agree_label}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Claude vs GPT-4o verdict comparison
                    c_left, c_right = st.columns(2)
                    with c_left:
                        claude_badge = f"badge-{result_ent.action.lower()}"
                        st.markdown(
                            f"<div style='background:#F5F3FF;border:2px solid #C4B5FD;"
                            f"border-radius:10px;padding:14px;text-align:center'>"
                            f"<div style='font-weight:700;color:#6D28D9;margin-bottom:8px'>🤖 Claude</div>"
                            f"<div class='{claude_badge}'>{result_ent.action_emoji} {result_ent.action}</div>"
                            f"<div style='margin-top:8px;font-size:.82rem;color:#64748B'>"
                            f"Confidence: {result_ent.confidence_score:.0f}%<br>"
                            f"Risk: {result_ent.risk_score:.0f}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with c_right:
                        gpt_badge = f"badge-{gpt.action.lower()}"
                        st.markdown(
                            f"<div style='background:#F0FFF8;border:2px solid #6EE7B7;"
                            f"border-radius:10px;padding:14px;text-align:center'>"
                            f"<div style='font-weight:700;color:#10A37F;margin-bottom:8px'>🟢 GPT-4o</div>"
                            f"<div class='{gpt_badge}'>{gpt.action_emoji if hasattr(gpt,'action_emoji') else gpt.action} {gpt.action}</div>"
                            f"<div style='margin-top:8px;font-size:.82rem;color:#64748B'>"
                            f"Confidence: {gpt.confidence_score:.0f}%<br>"
                            f"Risk: {gpt.risk_score:.0f}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # Score comparison bars
                    if gpt.scores:
                        st.markdown("##### 📊 Detection Score Comparison — Claude vs GPT-4o:")
                        claude_scores = result_ent.scores.as_dict()
                        gpt_scores    = gpt.scores.as_dict()
                        for technique in claude_scores:
                            c_val = claude_scores[technique]
                            g_val = gpt_scores.get(technique, 0)
                            diff  = abs(c_val - g_val)
                            diff_color = "#E74C3C" if diff > 20 else "#F39C12" if diff > 10 else "#27AE60"
                            st.markdown(
                                f"<div style='margin:5px 0'>"
                                f"<div style='display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:2px'>"
                                f"<span style='color:#1E293B'>{technique}</span>"
                                f"<span style='color:{diff_color};font-weight:600'>Δ {diff:.0f}%</span></div>"
                                f"<div style='display:flex;gap:3px;align-items:center'>"
                                f"<div style='font-size:.7rem;color:#7B2D8B;width:28px'>Claude</div>"
                                f"<div style='flex:1;background:#E2E8F0;border-radius:3px;height:6px'>"
                                f"<div style='background:#7B2D8B;width:{c_val:.0f}%;height:6px;border-radius:3px'></div></div>"
                                f"<div style='font-size:.7rem;color:#7B2D8B;width:32px;text-align:right'>{c_val:.0f}%</div></div>"
                                f"<div style='display:flex;gap:3px;align-items:center;margin-top:2px'>"
                                f"<div style='font-size:.7rem;color:#10A37F;width:28px'>GPT-4o</div>"
                                f"<div style='flex:1;background:#E2E8F0;border-radius:3px;height:6px'>"
                                f"<div style='background:#10A37F;width:{g_val:.0f}%;height:6px;border-radius:3px'></div></div>"
                                f"<div style='font-size:.7rem;color:#10A37F;width:32px;text-align:right'>{g_val:.0f}%</div></div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                    # GPT-4o explanation
                    if gpt.explanation:
                        st.markdown("##### 💬 GPT-4o Explanation:")
                        st.markdown(
                            f"<div style='background:#F0FFF8;border:1px solid #6EE7B7;border-radius:10px;"
                            f"padding:14px;font-size:.9rem;line-height:1.6;color:#065F46'>"
                            f"{gpt.explanation}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info(
                        "GPT-4o cross-validation was not run for this query. "
                        "Enable **Cross-validate with GPT-4o** in the sidebar (requires OpenAI API key) "
                        "and run a new query to see GPT-4o's independent analysis here."
                    )

        # ── TRUSTLAYER ────────────────────────────────────────────────────
        elif sel == "trustlayer":
            p_col, d_col = st.columns([1, 3])
            with p_col:
                proc = str(result_ent.processing_ms) + " ms" if result_ent and result_ent.processing_ms else "—"
                st.markdown(
                    f"<div style='background:#EFF6FF;border:2px solid #0066FF;border-radius:16px;"
                    f"padding:24px 16px;text-align:center;height:100%'>"
                    f"<div style='font-size:3rem'>🛡️</div>"
                    f"<div style='font-weight:700;color:#0066FF;margin-top:8px;font-size:1.1rem'>TrustLayer</div>"
                    f"<div style='font-size:.78rem;color:#3B82F6;margin-top:4px'>AI Safety Middleware</div>"
                    f"<hr style='border-color:#BFDBFE;margin:14px 0'>"
                    f"<div style='font-size:.78rem;color:#475569;text-align:left;line-height:1.8'>"
                    f"<b>Techniques:</b> 8 detection algorithms<br>"
                    f"<b>Outputs:</b> PASS / FLAG / BLOCK<br>"
                    f"<b>Cross-validated:</b> Claude + GPT-4o<br>"
                    f"<b>Processing time:</b> {proc}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            with d_col:
                if result_ent:
                    badge_cls = f"badge-{result_ent.action.lower()}"
                    st.markdown(
                        f"<div style='text-align:center;margin-bottom:16px'>"
                        f"<div class='{badge_cls}' style='font-size:1.2rem;padding:10px 28px'>"
                        f"{result_ent.action_emoji} {result_ent.action} &nbsp;·&nbsp; "
                        f"Confidence {result_ent.confidence_score:.0f}% &nbsp;·&nbsp; "
                        f"Risk Score {result_ent.risk_score:.0f}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("##### 📊 8 Detection Technique Scores:")
                    for technique, score_val in result_ent.scores.as_dict().items():
                        bar_color = "#27AE60" if score_val >= 75 else "#F39C12" if score_val >= 50 else "#E74C3C"
                        st.markdown(
                            f"<div style='margin:5px 0'>"
                            f"<div style='display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:2px'>"
                            f"<span style='color:#1E293B'>{technique}</span>"
                            f"<span style='font-weight:700;color:{bar_color}'>{score_val:.0f}%</span></div>"
                            f"<div style='background:#E2E8F0;border-radius:4px;height:7px'>"
                            f"<div style='background:{bar_color};width:{score_val:.0f}%;height:7px;border-radius:4px'></div>"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )
                    if result_ent.issues:
                        st.markdown("##### ⚠️ Issues detected:")
                        for iss in result_ent.issues[:5]:
                            st.markdown(f"<div class='issue-high'>⚠️ {iss}</div>", unsafe_allow_html=True)
                        if len(result_ent.issues) > 5:
                            st.caption(f"+{len(result_ent.issues)-5} more issues")
                    else:
                        st.success("No issues detected — all claims appear accurate.")
                else:
                    st.info("Run a query in **Live Detection** to see TrustLayer's full analysis.")

        # ── OUTCOME ───────────────────────────────────────────────────────
        elif sel == "outcome":
            if action_ent == "PASS":
                icon, title, sub, border_c, bg_c = "✅", "PASS", "Delivered safely", "#27AE60", "#F0FDF4"
            elif action_ent == "FLAG":
                icon, title, sub, border_c, bg_c = "⚠️", "FLAG", "Human review queue", "#F39C12", "#FFFBEB"
            elif action_ent == "BLOCK":
                icon, title, sub, border_c, bg_c = "🚫", "BLOCK", "Safe fallback sent", "#E74C3C", "#FFF1F2"
            else:
                icon, title, sub, border_c, bg_c = "⬜", "Pending", "Run a query first", "#94A3B8", "#F8FAFC"

            p_col, d_col = st.columns([1, 3])
            with p_col:
                st.markdown(
                    f"<div style='background:{bg_c};border:2px solid {border_c};border-radius:16px;"
                    f"padding:24px 16px;text-align:center;height:100%'>"
                    f"<div style='font-size:3rem'>{icon}</div>"
                    f"<div style='font-weight:800;color:{border_c};margin-top:8px;font-size:1.3rem'>{title}</div>"
                    f"<div style='font-size:.78rem;color:#64748B;margin-top:4px'>{sub}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with d_col:
                if action_ent == "PASS":
                    st.success("Response cleared all 8 detection checks. Delivered to the customer without modification.")
                    st.markdown(
                        f"<div style='background:#fff;border:2px solid #86EFAC;border-radius:10px;"
                        f"padding:16px;font-size:.9rem;line-height:1.7;max-height:260px;overflow-y:auto'>"
                        f"{result_ent.llm_response}</div>",
                        unsafe_allow_html=True,
                    )
                elif action_ent == "FLAG":
                    st.warning("Response scored borderline — routed to human review. Customer sees a holding message.")
                    st.markdown(
                        "<div style='background:#FFFBEB;border:2px solid #FCD34D;border-radius:10px;padding:16px'>"
                        "<b>📋 Review Queue entry created. A compliance reviewer will:</b><br><br>"
                        "&nbsp;&nbsp;• <b>Approve</b> → deliver the original AI response<br>"
                        "&nbsp;&nbsp;• <b>Reject</b> → send safe fallback instead<br>"
                        "&nbsp;&nbsp;• <b>Escalate</b> → raise to senior compliance officer<br><br>"
                        "See the <b>👥 Review Queue</b> tab for the queued item."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif action_ent == "BLOCK":
                    st.error("Response scored below threshold — blocked. Customer received a safe fallback.")
                    if st.session_state.blocked_responses:
                        latest = st.session_state.blocked_responses[-1]
                        b_left, b_right = st.columns(2)
                        with b_left:
                            st.markdown(
                                "<div style='background:#FEE2E2;border-left:4px solid #E74C3C;"
                                "padding:10px;border-radius:8px;margin-bottom:8px'>"
                                "<b style='color:#991B1B'>❌ AI Generated (BLOCKED — never shown)</b></div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div class='response-box' style='border-color:#FECDD3'>"
                                f"{latest['ai_response']}</div>",
                                unsafe_allow_html=True,
                            )
                            if latest["issues"]:
                                st.markdown("**Why it was blocked:**")
                                for iss in latest["issues"][:4]:
                                    st.markdown(f"<div class='issue-high'>⛔ {iss}</div>", unsafe_allow_html=True)
                        with b_right:
                            st.markdown(
                                "<div style='background:#DCFCE7;border-left:4px solid #22C55E;"
                                "padding:10px;border-radius:8px;margin-bottom:8px'>"
                                "<b style='color:#166534'>✅ Safe Fallback (what customer received)</b></div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div class='response-box' style='border-color:#DCFCE7;font-style:italic'>"
                                f"{latest['user_saw']}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                "<div style='background:#F0FDF4;border:1px solid #86EFAC;"
                                "border-radius:8px;padding:10px;margin-top:8px;font-size:.85rem;color:#166534'>"
                                "✔ Customer protected<br>✔ Harmful content never delivered<br>"
                                "✔ Interaction logged for compliance audit</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("Run a query in **Live Detection** to see the outcome here.")
    else:
        if result_ent:
            st.info("👆 Click any stage above to explore that persona's view of the pipeline.")
        else:
            st.info("Run a query in **Live Detection** then click a stage above to explore the pipeline.")

    # ── All blocked responses history ─────────────────────────────────────
    if st.session_state.blocked_responses:
        st.markdown("---")
        with st.expander(f"📋 All Blocked Responses This Session ({len(st.session_state.blocked_responses)})"):
            for i, br in enumerate(reversed(st.session_state.blocked_responses)):
                st.markdown(
                    f"<div style='background:#FFF1F2;border:1px solid #FECDD3;"
                    f"border-radius:8px;padding:10px;margin-bottom:8px'>"
                    f"<div style='font-weight:700;color:#991B1B'>🚫 BLOCK #{len(st.session_state.blocked_responses)-i} "
                    f"· {br['industry']} · {br['time']}</div>"
                    f"<div style='font-size:.85rem;color:#64748B;margin:4px 0'>"
                    f"<strong>Query:</strong> {br['query'][:120]}{'…' if len(br['query'])>120 else ''}</div>"
                    f"<div style='font-size:.82rem;color:#991B1B'>"
                    f"Confidence: {br['confidence']}% · Risk: {br['risk']} · "
                    f"Issues: {len(br['issues'])}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: HUMAN REVIEW QUEUE
# ─────────────────────────────────────────────────────────────────────────────
with tab_review:
    pending = [r for r in st.session_state.review_queue if r["status"] == "pending"]
    reviewed = [r for r in st.session_state.review_queue if r["status"] != "pending"]

    # ── Header metrics ─────────────────────────────────────────────────────
    st.markdown("#### 👥 Human Review Queue")
    st.markdown(
        "FLAG responses land here for a compliance reviewer to **Approve**, **Reject**, "
        "or **Escalate to BLOCK**. Every decision is logged to the audit trail."
    )
    rq1, rq2, rq3, rq4 = st.columns(4)
    rq1.metric("Pending Review", len(pending))
    rq2.metric("Approved",  sum(1 for r in st.session_state.review_queue if r["status"] == "approved"))
    rq3.metric("Rejected",  sum(1 for r in st.session_state.review_queue if r["status"] == "rejected"))
    rq4.metric("Escalated", sum(1 for r in st.session_state.review_queue if r["status"] == "escalated"))

    st.markdown("---")

    if not st.session_state.review_queue:
        st.info(
            "No items in the review queue yet. "
            "Run queries in **Live Detection** — FLAG decisions will appear here automatically."
        )
    else:
        # ── Pending items ──────────────────────────────────────────────────
        if pending:
            st.markdown(f"### ⏳ Pending Review  ({len(pending)} items)")
            for item in pending:
                with st.container():
                    escalated = item.get("escalated", False)
                    header_bg    = "#FFF1F2"      if escalated else "#FFFBEB"
                    header_border= "#FECDD3"      if escalated else "#FCD34D"
                    header_color = "#991B1B"      if escalated else "#92400E"
                    status_label = (
                        f"⬆️ ESCALATED → {item.get('consensus_action','BLOCK')}"
                        if escalated else "PENDING REVIEW"
                    )
                    status_bg    = "#FEE2E2"      if escalated else "#FEF9C3"
                    status_border= "#FECDD3"      if escalated else "#FCD34D"
                    flag_icon    = "🚨" if escalated else "⚠️"
                    st.markdown(
                        f"<div style='background:{header_bg};border:2px solid {header_border};"
                        f"border-radius:12px;padding:16px;margin-bottom:8px'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                        f"<div style='font-weight:700;color:{header_color};font-size:1rem'>"
                        f"{flag_icon} FLAG #{item['id']}  ·  {item['industry']}  ·  {item['time']}</div>"
                        f"<div style='background:{status_bg};border:1px solid {status_border};"
                        f"border-radius:8px;padding:4px 12px;font-size:.8rem;"
                        f"color:{header_color};font-weight:600'>{status_label}</div>"
                        f"</div>"
                        + (
                            f"<div style='margin-top:8px;background:#FEE2E2;border-radius:6px;"
                            f"padding:6px 10px;font-size:.82rem;color:#991B1B'>"
                            f"🔴 <strong>Cross-validation disagreement escalated this FLAG → BLOCK.</strong> "
                            f"Review the AI response below before making your decision.</div>"
                            if escalated else ""
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                    # Query + AI response + TrustLayer verdict side-by-side
                    qi_col, ai_col, tl_col = st.columns([1.2, 1.8, 1])

                    with qi_col:
                        st.markdown("**Customer Query**")
                        st.markdown(
                            f"<div style='background:#fff;border:1px solid #FEF3C7;"
                            f"border-radius:8px;padding:10px;font-size:.9rem;"
                            f"min-height:100px'>{item['query']}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div style='font-size:.8rem;color:#64748B;margin-top:4px'>"
                            f"Industry: <strong>{item['industry']}</strong></div>",
                            unsafe_allow_html=True,
                        )

                    with ai_col:
                        st.markdown("**AI-Generated Response (unfiltered)**")
                        st.markdown(
                            f"<div style='background:#fff;border:1px solid #FEF3C7;"
                            f"border-radius:8px;padding:10px;font-size:.88rem;"
                            f"line-height:1.6;max-height:160px;overflow-y:auto'>"
                            f"{item['ai_response']}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        if item.get("issues"):
                            st.markdown(
                                f"<div style='margin-top:4px'>"
                                + "".join(
                                    f"<div class='issue-medium' style='font-size:.78rem'>"
                                    f"⚠️ {iss[:90]}</div>"
                                    for iss in item["issues"][:3]
                                )
                                + ("" if len(item["issues"]) <= 3 else
                                   f"<div style='font-size:.75rem;color:#64748B'>"
                                   f"+{len(item['issues'])-3} more</div>")
                                + "</div>",
                                unsafe_allow_html=True,
                            )

                    with tl_col:
                        st.markdown("**TrustLayer Verdict**")
                        consensus_lbl   = item.get("consensus_action", "FLAG")
                        consensus_badge = f"badge-{consensus_lbl.lower()}"
                        escalated_html  = (
                            f"<div style='margin-top:6px;font-size:.72rem;color:#64748B'>"
                            f"Consensus (cross-val)</div>"
                            f"<div class='{consensus_badge}' style='margin-top:2px'>"
                            f"{consensus_lbl}</div>"
                        ) if item.get("escalated") else ""
                        st.markdown(
                            f"<div style='background:#FFFBEB;border:1px solid #FCD34D;"
                            f"border-radius:8px;padding:10px;text-align:center'>"
                            f"<div style='font-size:.72rem;color:#64748B;margin-bottom:4px'>"
                            f"Claude raw decision</div>"
                            f"<div class='badge-flag'>⚠️ FLAG</div>"
                            f"{escalated_html}"
                            f"<div style='margin-top:8px;font-size:.9rem'>"
                            f"<strong>Conf:</strong> {item['confidence']}%</div>"
                            f"<div style='font-size:.9rem'>"
                            f"<strong>Risk:</strong> {item['risk']}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        if item.get("explanation"):
                            st.caption(item["explanation"][:120])

                    # Reviewer note input
                    note_key = f"note_{item['id']}"
                    reviewer_note = st.text_input(
                        "Reviewer note (optional)",
                        key=note_key,
                        placeholder="Add a note for the audit trail…",
                    )

                    # Action buttons
                    act_a, act_b, act_c, act_d = st.columns([1, 1, 1, 2])
                    with act_a:
                        if st.button(
                            "✅ Approve",
                            key=f"approve_{item['id']}",
                            help="Response is acceptable — mark as reviewed and deliver",
                            type="primary",
                            use_container_width=True,
                        ):
                            item["status"] = "approved"
                            item["reviewer_note"] = reviewer_note
                            st.session_state.audit_log.append({
                                "time":              datetime.now().strftime("%H:%M:%S"),
                                "reviewer":          st.session_state.user_name,
                                "item_id":           item["id"],
                                "industry":          item["industry"],
                                "original_decision": "FLAG",
                                "reviewer_action":   "APPROVED",
                                "final_decision":    "PASS",
                                "query":             item["query"][:100],
                                "note":              reviewer_note or "—",
                            })
                            st.rerun()

                    with act_b:
                        if st.button(
                            "🚫 Reject",
                            key=f"reject_{item['id']}",
                            help="Response is unacceptable — escalate to BLOCK",
                            use_container_width=True,
                        ):
                            item["status"] = "rejected"
                            item["reviewer_note"] = reviewer_note
                            st.session_state.blocked_responses.append({
                                "time":        datetime.now().strftime("%H:%M:%S"),
                                "industry":    item["industry"],
                                "query":       item["query"],
                                "ai_response": item["ai_response"],
                                "user_saw":    (
                                    "I'm sorry, I'm not able to provide specific details on that right now. "
                                    "Please speak with one of our advisors who can give you accurate, "
                                    "personalised guidance based on your situation."
                                ),
                                "confidence":  item["confidence"],
                                "risk":        item["risk"],
                                "issues":      item["issues"],
                                "fabrication": item.get("fabrication", []),
                                "explanation": item["explanation"],
                            })
                            st.session_state.audit_log.append({
                                "time":              datetime.now().strftime("%H:%M:%S"),
                                "reviewer":          st.session_state.user_name,
                                "item_id":           item["id"],
                                "industry":          item["industry"],
                                "original_decision": "FLAG",
                                "reviewer_action":   "REJECTED",
                                "final_decision":    "BLOCK",
                                "query":             item["query"][:100],
                                "note":              reviewer_note or "—",
                            })
                            st.rerun()

                    with act_c:
                        if st.button(
                            "⬆️ Escalate",
                            key=f"escalate_{item['id']}",
                            help="Escalate to senior compliance team",
                            use_container_width=True,
                        ):
                            item["status"] = "escalated"
                            item["reviewer_note"] = reviewer_note
                            st.session_state.audit_log.append({
                                "time":              datetime.now().strftime("%H:%M:%S"),
                                "reviewer":          st.session_state.user_name,
                                "item_id":           item["id"],
                                "industry":          item["industry"],
                                "original_decision": "FLAG",
                                "reviewer_action":   "ESCALATED",
                                "final_decision":    "ESCALATED",
                                "query":             item["query"][:100],
                                "note":              reviewer_note or "—",
                            })
                            st.rerun()

                    st.markdown("<hr style='border-color:#FEF3C7;margin:8px 0'>", unsafe_allow_html=True)

        # ── Completed items ────────────────────────────────────────────────
        if reviewed:
            st.markdown(f"### ✅ Reviewed Items  ({len(reviewed)})")
            for item in reversed(reviewed):
                status_map = {
                    "approved":  ("✅ APPROVED → PASS",  "#DCFCE7", "#166534"),
                    "rejected":  ("🚫 REJECTED → BLOCK", "#FEE2E2", "#991B1B"),
                    "escalated": ("⬆️ ESCALATED",        "#EFF6FF", "#1D4ED8"),
                }
                label, bg, fg = status_map.get(item["status"], ("—", "#F1F5F9", "#64748B"))
                st.markdown(
                    f"<div style='background:{bg};border-radius:8px;padding:10px 14px;"
                    f"margin-bottom:8px;display:flex;justify-content:space-between;"
                    f"align-items:center'>"
                    f"<div>"
                    f"<span style='font-weight:700;color:{fg}'>{label}</span>"
                    f"<span style='color:#64748B;font-size:.82rem;margin-left:12px'>"
                    f"#{item['id']} · {item['industry']} · {item['time']}</span><br>"
                    f"<span style='font-size:.82rem;color:#475569'>"
                    f"{item['query'][:110]}{'…' if len(item['query'])>110 else ''}"
                    f"</span>"
                    f"</div>"
                    f"<div style='font-size:.8rem;color:{fg};font-weight:600;text-align:right'>"
                    f"Conf: {item['confidence']}%<br>Risk: {item['risk']}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                if item.get("reviewer_note"):
                    st.caption(f"Reviewer note: {item['reviewer_note']}")

        st.markdown("---")

        # ── Audit Log ─────────────────────────────────────────────────────
        st.markdown("### 📋 Compliance Audit Log")
        if not st.session_state.audit_log:
            st.caption("No reviewer decisions yet.")
        else:
            audit_action_colors = {
                "APPROVED":  "background-color:#DCFCE7;color:#166534;font-weight:700",
                "REJECTED":  "background-color:#FEE2E2;color:#991B1B;font-weight:700",
                "ESCALATED": "background-color:#EFF6FF;color:#1D4ED8;font-weight:700",
            }

            def _color_audit(val):
                return audit_action_colors.get(val, "")

            df_audit = pd.DataFrame(st.session_state.audit_log)
            st.dataframe(
                df_audit.style.map(_color_audit, subset=["reviewer_action"]),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "⬇️ Export Audit Log CSV",
                data=df_audit.to_csv(index=False),
                file_name=f"trustlayer_audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
with tab_howto:
    st.markdown("#### How TrustLayer AI Works")

    st.markdown("""
    TrustLayer AI intercepts every AI response, analyzes it using **8 parallel detection
    algorithms**, and returns a **Confidence Score** and **Risk Score** that drive an
    automated **PASS / FLAG / BLOCK** decision.
    """)

    st.markdown("---")
    st.markdown("#### Detection Pipeline")

    pipeline_steps = [
        ("1", "Generate Response",    "Claude generates a response to your query acting as an industry-specific AI assistant.", "#0066FF"),
        ("2", "Claim Extraction",     "TrustLayer extracts all factual claims, citations, numerical assertions, and temporal statements.", "#0066FF"),
        ("3", "8-Technique Analysis", "All 8 detection algorithms run in parallel against the response and grounding data.", "#F39C12"),
        ("4", "Ensemble Scoring",     "Individual scores are combined into a weighted Confidence Score and Risk Score.", "#F39C12"),
        ("5", "Decision Policy",      "Scores are compared to thresholds: PASS (≥75%), FLAG (50–74%), BLOCK (<50%).", "#E74C3C"),
        ("6", "Audit Log",            "Every interaction is logged with full scores, claims, issues, and compliance tags.", "#27AE60"),
    ]
    for step, title, desc, color in pipeline_steps:
        st.markdown(
            f"<div style='display:flex;gap:12px;align-items:flex-start;margin:10px 0'>"
            f"<div style='background:{color};color:#fff;border-radius:50%;width:32px;height:32px;"
            f"display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0'>{step}</div>"
            f"<div><strong>{title}</strong><br><span style='color:#64748B;font-size:.9rem'>{desc}</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### The 8 Detection Techniques")

    techniques = [
        ("Semantic Entropy Analysis",       "87%", "45ms",  "Measures model confidence via token probability distributions. High entropy = low confidence."),
        ("Self-Consistency Checking",       "82%", "120ms", "Samples multiple responses and checks factual agreement across them."),
        ("Source Verification",             "95%", "200ms", "Validates all cited cases, URLs, statutes, and references against real databases."),
        ("Enterprise Knowledge Grounding",  "98%", "150ms", "Compares response claims against your authoritative grounding data. Highest accuracy."),
        ("Claim Extraction & Classification","91%","35ms",  "Identifies and risk-classifies all assertive claims by type and criticality."),
        ("Hallucination Pattern Recognition","79%","25ms",  "Classifies response against known hallucination signatures using fine-tuned model."),
        ("Temporal Consistency Checking",   "93%", "40ms",  "Flags date-sensitive claims that may be outdated or inconsistent with reality."),
        ("Numerical Claim Validation",      "96%", "80ms",  "Validates all rates, dosages, thresholds, and statistics against authoritative ranges."),
    ]

    df_tech = pd.DataFrame(techniques, columns=["Technique", "Accuracy", "Latency", "Description"])

    def color_acc(val):
        pct = int(val.replace("%", ""))
        if pct >= 90: return "color:#166534;font-weight:700"
        if pct >= 80: return "color:#854D0E;font-weight:700"
        return "color:#991B1B;font-weight:700"

    st.dataframe(
        df_tech.style.map(color_acc, subset=["Accuracy"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("#### Decision Thresholds")

    col1, col2, col3 = st.columns(3)
    col1.markdown("""
    <div style='background:#DCFCE7;border-radius:10px;padding:16px;text-align:center'>
    <div style='font-size:1.4rem;font-weight:800;color:#166534'>✅ PASS</div>
    <div style='color:#166534;font-size:.9rem'>Confidence ≥ 75%<br>Risk Score &lt; 30</div>
    <div style='color:#64748B;font-size:.85rem;margin-top:8px'>Response delivered to user</div>
    </div>""", unsafe_allow_html=True)

    col2.markdown("""
    <div style='background:#FEF9C3;border-radius:10px;padding:16px;text-align:center'>
    <div style='font-size:1.4rem;font-weight:800;color:#854D0E'>⚠️ FLAG</div>
    <div style='color:#854D0E;font-size:.9rem'>Confidence 50–74%<br>Risk Score 30–59</div>
    <div style='color:#64748B;font-size:.85rem;margin-top:8px'>Routed to human review</div>
    </div>""", unsafe_allow_html=True)

    col3.markdown("""
    <div style='background:#FEE2E2;border-radius:10px;padding:16px;text-align:center'>
    <div style='font-size:1.4rem;font-weight:800;color:#991B1B'>🚫 BLOCK</div>
    <div style='color:#991B1B;font-size:.9rem'>Confidence &lt; 50%<br>Risk Score ≥ 60</div>
    <div style='color:#64748B;font-size:.85rem;margin-top:8px'>Blocked + fallback triggered</div>
    </div>""", unsafe_allow_html=True)
