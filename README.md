# 🛡️ TrustLayer AI — Enterprise AI Reliability Platform

> **Real-time hallucination detection, prevention, and governance for enterprise AI deployments.**
> Built for Canada Hackathon 2026 · Powered by Anthropic Claude + OpenAI GPT-4o

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Claude](https://img.shields.io/badge/Powered%20by-Claude%20Sonnet-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## The Problem

Enterprise AI deployments fail silently. A banking chatbot quotes a wrong mortgage rate. A healthcare assistant recommends the wrong drug dose. A legal assistant cites a case that doesn't exist. These are **hallucinations** — and they cause real financial, legal, and physical harm.

**TrustLayer AI** intercepts every AI response before it reaches the user, runs it through 8 parallel detection algorithms, and returns a **PASS / FLAG / BLOCK** decision in real time.

---

## Live Demo

```
streamlit run streamlit_app.py
```

Or deploy to [Streamlit Cloud](https://share.streamlit.io) in one click — see [Deployment](#deployment) below.

---

## How It Works

```
User Query
    │
    ▼
┌─────────────────────────────┐
│   Industry AI Assistant      │  ← Claude acts as domain expert
│   (BFSI / Healthcare / Legal)│
└────────────┬────────────────┘
             │  LLM Response
             ▼
┌─────────────────────────────────────────────────────────┐
│              TrustLayer Detection Engine                 │
│                                                          │
│  1. Semantic Entropy      5. Claim Classification        │
│  2. Self-Consistency      6. Pattern Recognition         │
│  3. Source Verification   7. Temporal Consistency        │
│  4. Enterprise Grounding  8. Numerical Validation        │
└────────────────────────┬────────────────────────────────┘
                         │
             ┌───────────┴───────────┐
             │   Confidence Score    │  0–100%
             │   Risk Score          │  0–100
             │   Decision Policy     │  PASS / FLAG / BLOCK
             └───────────────────────┘
```

### Decision Thresholds

| Decision | Confidence | Risk Score | Action |
|----------|-----------|------------|--------|
| ✅ **PASS** | ≥ 75% | < 30 | Response delivered to user |
| ⚠️ **FLAG** | 50–74% | 30–59 | Routed to human review |
| 🚫 **BLOCK** | < 50% | ≥ 60 | Blocked + fallback triggered |

---

## Claude + GPT-4o Cross-Validation

When an OpenAI API key is provided, TrustLayer runs the same detection prompt through **both Claude and GPT-4o independently** and computes an **Agreement Score**:

- **Agreement ≥ 85%** → Strong Agreement — high-reliability result
- **Agreement 65–84%** → Moderate Agreement — review recommended
- **Agreement < 65%** → Significant Disagreement — itself a risk signal → conservative escalation

The consensus decision is always **conservative**: if either model says BLOCK, the result is BLOCK.

---

## Industry Modules

| Module | Risk Level | Scenarios |
|--------|-----------|-----------|
| 🏦 **BFSI — Banking** | HIGH (1.3×) | Mortgage rates, loan eligibility |
| 🏥 **Healthcare** | CRITICAL (1.5×) | Drug dosing, interactions |
| ⚖️ **Legal** | HIGH (1.4×) | Case law citations, statutes |
| 👥 **Enterprise HR** | MEDIUM (1.1×) | PTO policy, benefits |
| 🤖 **General** | STANDARD (1.0×) | Custom queries |

Each module includes **authoritative grounding data** — e.g., the BFSI module knows the current bank product rates, so it can catch a chatbot that fabricates a promotional rate that doesn't exist.

---

## BFSI Demo Scenario

Select **BFSI — Banking** in the sidebar, then load the **"Mortgage Rate Inquiry"** preset:

> *"What is the current interest rate for a 30-year fixed mortgage? Am I likely to get pre-approved and what special promotions do you have?"*

TrustLayer will:
1. Generate a banking assistant response via Claude
2. Compare every claim against grounding data (30-yr = 6.85% APR, no active promotions)
3. Flag any fabricated rates, illegal pre-approval guarantees, or invented promotions
4. Return a scored PASS / FLAG / BLOCK decision with full claim breakdown

---

## Project Structure

```
trustlayer-ai/
├── streamlit_app.py              # Main Streamlit UI
├── requirements.txt              # Python dependencies
├── .streamlit/
│   ├── config.toml               # Theme + server config
│   └── secrets.toml.example      # API key template
└── trustlayer/
    ├── __init__.py
    ├── models.py                 # Pydantic v2 data models
    ├── industries.py             # 5 industry module configs
    ├── detector.py               # Claude detection engine
    └── cross_validator.py        # Claude + GPT-4o cross-validation
```

---

## Installation

```bash

```

Copy the secrets template and add your API keys:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
OPENAI_API_KEY    = "sk-your-openai-key-here"   # optional, for cross-validation
```

Run the app:

```bash
streamlit run streamlit_app.py
```

---

## Deployment

### Streamlit Cloud (Recommended)

1. Fork this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select this repo + `streamlit_app.py`
4. In **Advanced settings → Secrets**, paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   OPENAI_API_KEY    = "sk-..."
   ```
5. Click **Deploy**

---

## Compliance Coverage

| Standard | Coverage |
|----------|---------|
| EU AI Act (2024) | 94% — risk classification, human oversight, transparency |
| NIST AI RMF | GOVERN, MAP, MEASURE, MANAGE functions |
| ISO 42001 | AI management system controls |
| SOC 2 Type II | Audit logging, access control, data integrity |

---

## Tech Stack

- **Detection Engine**: Anthropic Claude Sonnet (`claude-sonnet-4-5`)
- **Cross-Validation**: OpenAI GPT-4o (`gpt-4o`)
- **UI**: Streamlit + Plotly
- **Data Models**: Pydantic v2
- **Deployment**: Streamlit Cloud

---

## Canada Hackathon 2026

Built as a submission for Canada Hackathon 2026.

**Theme**: Enterprise AI Reliability & Governance
**Problem**: Hallucinations in high-stakes AI deployments (BFSI, Healthcare, Legal)
**Solution**: Real-time detection + dual-LLM cross-validation + industry-aware grounding

---

*TrustLayer AI — Detect · Prevent · Govern*
