import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SkyCast — Kathmandu Rain Predictor",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DESIGN SYSTEM (same SkyCast brand established earlier)
# ============================================================
SKY_DARK = "#1B2A4A"
SKY = "#2E4374"
SKY_LIGHT = "#EAF0FB"
MIST_BG = "#F5F7FB"
GOLD = "#E3B23C"
GOLD_DARK = "#C4941F"
STORM = "#5B4B8A"
RAIN_BLUE = "#3E7CB1"
INK = "#1A2233"
MUTED = "#5C6478"
CARD_BG = "#FFFFFF"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Manrope', sans-serif; }}
.stApp {{ background: {MIST_BG}; }}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
.sc-hero {{
    background: linear-gradient(135deg, {SKY_DARK} 0%, {SKY} 55%, {STORM} 100%);
    border-radius: 20px;
    padding: 2.4rem 2.6rem;
    margin-bottom: 1.6rem;
    color: white;
    position: relative;
    overflow: hidden;
}}
.sc-cloud {{
    position: absolute;
    background: rgba(255,255,255,0.10);
    border-radius: 50%;
    filter: blur(1px);
}}
@keyframes drift {{ from {{ transform: translateX(-10%); }} to {{ transform: translateX(10%); }} }}
.sc-cloud-1 {{ width: 200px; height: 200px; top: -80px; right: 10%; animation: drift 14s ease-in-out infinite alternate; }}
.sc-cloud-2 {{ width: 130px; height: 130px; top: 30px; right: -15px; animation: drift 10s ease-in-out infinite alternate-reverse; }}
.sc-eyebrow {{ font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; font-size: 0.7rem; color: {GOLD}; margin-bottom: 0.5rem; position: relative; z-index: 1; }}
.sc-title {{ font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.1rem; line-height: 1.12; margin: 0 0 0.5rem 0; color: white; position: relative; z-index: 1; }}
.sc-subtitle {{ font-size: 0.96rem; color: rgba(255,255,255,0.82); max-width: 680px; line-height: 1.5; margin: 0; position: relative; z-index: 1; }}
.sc-card {{ background: {CARD_BG}; border-radius: 16px; padding: 1.4rem 1.6rem; border: 1px solid rgba(27,42,74,0.08); box-shadow: 0 1px 3px rgba(27,42,74,0.05); margin-bottom: 1.1rem; }}
.sc-card-title {{ font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.08rem; color: {SKY_DARK}; margin-bottom: 0.2rem; }}
.sc-card-sub {{ color: {MUTED}; font-size: 0.83rem; margin-bottom: 0.9rem; }}
.sc-result {{ border-radius: 18px; padding: 1.6rem 1.9rem; color: white; }}
.sc-result-label {{ font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; font-size: 0.7rem; opacity: 0.85; margin-bottom: 0.3rem; }}
.sc-result-headline {{ font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.7rem; margin: 0 0 0.4rem 0; }}
.sc-result-body {{ font-size: 0.9rem; line-height: 1.5; opacity: 0.95; }}
.sc-pill {{ display: inline-block; padding: 0.26rem 0.68rem; border-radius: 999px; font-size: 0.74rem; font-weight: 700; background: {SKY_LIGHT}; color: {SKY_DARK}; }}
.sc-badge-note {{ background: rgba(217,99,75,0.08); border-left: 3px solid {GOLD_DARK}; border-radius: 8px; padding: 0.7rem 0.95rem; font-size: 0.78rem; color: {INK}; line-height: 1.5; }}
.sc-dice-note {{ background: rgba(91,75,138,0.08); border-left: 3px solid {STORM}; border-radius: 8px; padding: 0.7rem 0.95rem; font-size: 0.8rem; color: {INK}; line-height: 1.5; margin-bottom: 1rem; }}
div[data-testid="stSlider"] label, .stSelectbox label {{ color: {SKY_DARK} !important; font-weight: 600 !important; font-size: 0.82rem !important; }}
.stButton>button {{
    background: linear-gradient(135deg, {GOLD_DARK}, {GOLD});
    color: {INK}; border: none; border-radius: 10px; padding: 0.65rem 1.3rem;
    font-weight: 700; font-size: 0.93rem; width: 100%;
    box-shadow: 0 2px 8px rgba(196,148,31,0.35); transition: transform 0.15s ease;
}}
.stButton>button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(196,148,31,0.45); }}
.sc-dice-btn button {{
    background: linear-gradient(135deg, {STORM}, {SKY}) !important;
    box-shadow: 0 2px 8px rgba(91,75,138,0.35) !important;
}}
.sc-dice-btn button:hover {{ box-shadow: 0 4px 12px rgba(91,75,138,0.45) !important; }}
section[data-testid="stSidebar"] {{ background: {SKY_DARK}; }}
section[data-testid="stSidebar"] * {{ color: rgba(255,255,255,0.92) !important; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15); }}
div[data-baseweb="tab-list"] {{ gap: 4px; }}
button[data-baseweb="tab"] {{ font-weight: 700; font-size: 0.95rem; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_logo(size=36):
    st.markdown(f"""
    <svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="32" cy="32" r="30" fill="{SKY_DARK}"/>
        <ellipse cx="26" cy="27" rx="14" ry="9" fill="{SKY_LIGHT}"/>
        <ellipse cx="36" cy="24" rx="9" ry="7" fill="{SKY_LIGHT}"/>
        <line x1="24" y1="42" x2="21" y2="50" stroke="{GOLD}" stroke-width="3" stroke-linecap="round"/>
        <line x1="32" y1="42" x2="29" y2="50" stroke="{GOLD}" stroke-width="3" stroke-linecap="round"/>
        <line x1="40" y1="42" x2="37" y2="50" stroke="{GOLD}" stroke-width="3" stroke-linecap="round"/>
    </svg>
    """, unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_pipeline():
    return joblib.load("rain_prediction.pkl")


try:
    model = load_pipeline()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

FEATURE_ORDER = ['year', 'month', 'day', 'max_temp_C', 'min_temp_C', 'avg_temp_c',
                 'humidity', 'windgust', 'windspeed', 'sealevelpressure',
                 'cloudcover', 'visibility', 'solarradiation', 'uvindex']

MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

# Slider bounds, reused both for rendering and for randomizing
BOUNDS = {
    "year": (2015, 2030),
    "day": (1, 31),
    "max_temp": (5.0, 40.0),
    "min_temp": (0.0, 25.0),
    "avg_temp": (5.0, 30.0),
    "humidity": (20, 100),
    "pressure": (995, 1030),
    "cloudcover": (0, 100),
    "visibility": (0.0, 15.0),
    "windspeed": (0.0, 40.0),
    "solarradiation": (0.0, 350.0),
    "windgust": (0.0, 60.0),
    "uvindex": (0, 10),
}


def randomize_conditions():
    """Pick a physically-plausible random set of conditions and stash them in
    session_state so the sliders below pick them up on the next render."""
    st.session_state["rc_year"] = random.randint(*BOUNDS["year"])
    st.session_state["rc_month_name"] = random.choice(MONTH_NAMES)
    st.session_state["rc_day"] = random.randint(*BOUNDS["day"])

    min_t = round(random.uniform(*BOUNDS["min_temp"]), 1)
    max_t = round(random.uniform(min_t, BOUNDS["max_temp"][1]), 1)
    avg_t = round((min_t + max_t) / 2 + random.uniform(-1.5, 1.5), 1)
    st.session_state["rc_min_temp"] = min_t
    st.session_state["rc_max_temp"] = max_t
    st.session_state["rc_avg_temp"] = min(max(avg_t, BOUNDS["avg_temp"][0]), BOUNDS["avg_temp"][1])

    st.session_state["rc_humidity"] = random.randint(*BOUNDS["humidity"])
    st.session_state["rc_pressure"] = random.randint(*BOUNDS["pressure"])
    st.session_state["rc_cloudcover"] = random.randint(*BOUNDS["cloudcover"])
    st.session_state["rc_visibility"] = round(random.uniform(*BOUNDS["visibility"]), 1)

    windspeed = round(random.uniform(*BOUNDS["windspeed"]), 1)
    st.session_state["rc_windspeed"] = windspeed
    st.session_state["rc_windgust"] = round(min(windspeed + random.uniform(2, 15), BOUNDS["windgust"][1]), 1)

    st.session_state["rc_solarradiation"] = round(random.uniform(*BOUNDS["solarradiation"]), 1)
    st.session_state["rc_uvindex"] = random.randint(*BOUNDS["uvindex"])

    st.session_state["rc_auto_predict"] = True


# ============================================================
# LOAD/COMPUTE SARIMA SEASONAL FORECAST (cached, computed once)
# ============================================================
@st.cache_data
def get_seasonal_forecast():
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        df = pd.read_csv("cleaned_dataset.csv")
        monthly = df.groupby(['year', 'month']).agg(rainy_days_pct=('rain', 'mean')).reset_index()
        monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
        monthly = monthly.set_index('date').sort_index()

        full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq='MS')
        monthly_full = monthly.reindex(full_index)
        ts = monthly_full['rainy_days_pct'] * 100
        ts = ts.interpolate(method='time')
        ts = ts.fillna(ts.groupby(ts.index.month).transform('mean'))
        ts = ts.asfreq('MS')

        p = (ts / 100).clip(0.01, 0.99)
        ts_logit = np.log(p / (1 - p))

        sarima = SARIMAX(ts_logit, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                          enforce_stationarity=False, enforce_invertibility=False)
        fit = sarima.fit(disp=False)

        months_ahead = (pd.Timestamp('2026-12-01').to_period('M') - ts.index[-1].to_period('M')).n
        forecast = fit.get_forecast(steps=max(months_ahead, 1))
        forecast_logit = forecast.predicted_mean
        ci_logit = forecast.conf_int(alpha=0.20)

        def logit_to_pct(x):
            return 100 / (1 + np.exp(-x))

        forecast_pct = forecast_logit.apply(logit_to_pct)
        lower_pct = ci_logit.iloc[:, 0].apply(logit_to_pct)
        upper_pct = ci_logit.iloc[:, 1].apply(logit_to_pct)

        rows = []
        for date, pred, lower, upper in zip(forecast_pct.index, forecast_pct.values,
                                              lower_pct.values, upper_pct.values):
            if date.year == 2026:
                rows.append({
                    "Month": MONTH_NAMES[date.month - 1],
                    "month_num": date.month,
                    "Predicted": round(pred, 1),
                    "Lower": round(lower, 1),
                    "Upper": round(upper, 1)
                })
        return pd.DataFrame(rows)
    except Exception:
        return None


def risk_bucket(p):
    if p >= 0.66:
        return "high", RAIN_BLUE, "Rain likely"
    elif p >= 0.33:
        return "moderate", GOLD_DARK, "Could go either way"
    else:
        return "low", "#3E8E5A", "Rain unlikely"


def render_gauge(prob, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 40, "family": "Fraunces", "color": INK}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "showticklabels": False},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "#E7F4EC"},
                {"range": [33, 66], "color": "#FBEFD6"},
                {"range": [66, 100], "color": "#E3EEF8"},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", font={"family": "Manrope"})
    return fig


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    c1, c2 = st.columns([1, 4])
    with c1:
        render_logo(34)
    with c2:
        st.markdown("<div style='font-family:Fraunces,serif; font-weight:700; font-size:1.15rem; padding-top:4px;'>SkyCast</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("##### What this app does")
    st.markdown(
        "**2026 Seasonal Outlook** — a calendar-based view showing how rainy a "
        "typical month tends to be, using a time-series model (SARIMA) fit on "
        "10 years of historical seasonal patterns.\n\n"
        "**Rain Check** — set your own conditions (or roll the dice for a random "
        "set) and the model classifies whether *those conditions* look like Rain "
        "or No Rain. It's a what-if tool, not a live forecast — it doesn't know "
        "tomorrow's actual weather on its own."
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("##### Model card")
    st.markdown(
        "- **Classifier:** Random Forest\n"
        "- **Test accuracy:** ~0.84\n"
        "- **ROC-AUC:** ~0.90\n"
        "- **Top driver:** Humidity\n"
        "- **Excludes:** direct precipitation probability, to avoid the model "
        "relying on a forecast value instead of real weather signals"
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Built as a portfolio project. Not a real weather forecasting service.")

# ============================================================
# HERO
# ============================================================
st.markdown(f"""
<div class="sc-hero">
    <div class="sc-cloud sc-cloud-1"></div>
    <div class="sc-cloud sc-cloud-2"></div>
    <div class="sc-eyebrow">Kathmandu · 2015–2025 weather data</div>
    <div class="sc-title">What does 2026 look like — and what does the model think today?</div>
    <p class="sc-subtitle">
        Start with the 2026 seasonal outlook, built from 10 years of Kathmandu's
        historical rainfall pattern. Then switch to Rain Check to set your own
        weather conditions — or let the model roll the dice on a random set — and
        see whether that combination looks like Rain or No Rain.
    </p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Could not load `rain_prediction.pkl`. Make sure it's in the same folder "
              f"as this app.\n\nDetails: {load_error}")
    st.stop()

tab1, tab2 = st.tabs(["📅 2026 Seasonal Outlook", "🎲 Rain Check (your conditions)"])

# ============================================================
# TAB 1 — SARIMA SEASONAL FORECAST (shown first)
# ============================================================
with tab1:
    st.markdown("""
    <div class="sc-card">
        <div class="sc-card-title">📅 2026 Seasonal Rain Outlook</div>
        <div class="sc-card-sub">
            Based on 10 years of Kathmandu's historical monthly rainfall pattern (SARIMA
            time-series model). This shows each month's typical rain likelihood — not a
            specific day's forecast.
        </div>
    """, unsafe_allow_html=True)

    forecast_df = get_seasonal_forecast()

    if forecast_df is None or forecast_df.empty:
        st.warning("Seasonal forecast unavailable — make sure `cleaned_dataset.csv` and "
                   "the `statsmodels` package are available alongside this app.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df["Month"], y=forecast_df["Upper"],
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["Month"], y=forecast_df["Lower"],
            fill="tonexty", fillcolor="rgba(62,124,177,0.15)",
            line=dict(width=0), name="80% confidence range", hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["Month"], y=forecast_df["Predicted"],
            mode="lines+markers", line=dict(color=RAIN_BLUE, width=3),
            marker=dict(size=8), name="Predicted rainy days (%)"
        ))
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Rainy days (%)", range=[0, 100]),
            font={"family": "Manrope"}, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.dataframe(
            forecast_df[["Month", "Predicted", "Lower", "Upper"]].rename(
                columns={"Predicted": "Predicted Rainy Days (%)",
                         "Lower": "Lower Bound (80% CI)", "Upper": "Upper Bound (80% CI)"}
            ),
            use_container_width=True, hide_index=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sc-badge-note">
        <b>What this is, and isn't.</b> This shows the seasonal pattern — how rainy a typical
        month tends to be, based on 10 years of history. It does not know 2026's specific
        weather anomalies (like an El Niño year) and cannot predict any individual day.
        Treat it as a climate-pattern summary, not a precise forecast. Want to check a
        specific set of conditions instead? Head to the Rain Check tab.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 2 — RAIN CHECK CLASSIFIER (manual or randomized conditions)
# ============================================================
with tab2:
    st.markdown(f"""
    <div class="sc-dice-note">
        🎲 <b>Not sure what to enter?</b> Click <b>"Guess random conditions"</b> below and
        the app will pick a plausible random set of weather values for you, then show
        what the model predicts — Rain or No Rain — for that assumption.
    </div>
    """, unsafe_allow_html=True)

    dice_col, _ = st.columns([1, 2])
    with dice_col:
        st.markdown('<div class="sc-dice-btn">', unsafe_allow_html=True)
        if st.button("🎲 Guess random conditions", use_container_width=True):
            randomize_conditions()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("""
        <div class="sc-card">
            <div class="sc-card-title">📅 Date</div>
            <div class="sc-card-sub">Captures seasonal rainfall patterns.</div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            year = st.slider("Year", *BOUNDS["year"], key="rc_year",
                              value=st.session_state.get("rc_year", datetime.now().year))
        with c2:
            month_name = st.selectbox(
                "Month", MONTH_NAMES, key="rc_month_name",
                index=MONTH_NAMES.index(st.session_state.get("rc_month_name", MONTH_NAMES[min(datetime.now().month - 1, 11)]))
            )
            month = MONTH_NAMES.index(month_name) + 1
        with c3:
            day = st.slider("Day", *BOUNDS["day"], key="rc_day",
                             value=st.session_state.get("rc_day", min(datetime.now().day, 31)))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="sc-card">
            <div class="sc-card-title">🌡️ Temperature</div>
            <div class="sc-card-sub">Assumed daily high, low, and average, in Celsius.</div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            max_temp = st.slider("Max temp (°C)", *BOUNDS["max_temp"], key="rc_max_temp",
                                  value=st.session_state.get("rc_max_temp", 25.0))
        with c2:
            min_temp = st.slider("Min temp (°C)", *BOUNDS["min_temp"], key="rc_min_temp",
                                  value=st.session_state.get("rc_min_temp", 14.0))
        with c3:
            avg_temp = st.slider("Avg temp (°C)", *BOUNDS["avg_temp"], key="rc_avg_temp",
                                  value=st.session_state.get("rc_avg_temp", 19.0))
        if min_temp > max_temp:
            st.warning("Min temp is higher than max temp — please adjust so min ≤ max.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="sc-card">
            <div class="sc-card-title">💧 Atmosphere</div>
            <div class="sc-card-sub">Assumed humidity, pressure, and cloud cover.</div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            humidity = st.slider("Humidity (%)", *BOUNDS["humidity"], key="rc_humidity",
                                  value=st.session_state.get("rc_humidity", 70))
            pressure = st.slider("Sea level pressure (hPa)", *BOUNDS["pressure"], key="rc_pressure",
                                  value=st.session_state.get("rc_pressure", 1014))
        with c2:
            cloudcover = st.slider("Cloud cover (%)", *BOUNDS["cloudcover"], key="rc_cloudcover",
                                    value=st.session_state.get("rc_cloudcover", 40))
            visibility = st.slider("Visibility (km)", *BOUNDS["visibility"], key="rc_visibility",
                                    value=st.session_state.get("rc_visibility", 4.0))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="sc-card">
            <div class="sc-card-title">💨 Wind & Sun</div>
            <div class="sc-card-sub">Assumed wind speed, gust, solar radiation, and UV index.</div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            windspeed = st.slider("Wind speed (km/h)", *BOUNDS["windspeed"], key="rc_windspeed",
                                   value=st.session_state.get("rc_windspeed", 10.0))
            solarradiation = st.slider("Solar radiation (W/m²)", *BOUNDS["solarradiation"], key="rc_solarradiation",
                                        value=st.session_state.get("rc_solarradiation", 180.0))
        with c2:
            windgust = st.slider("Wind gust (km/h)", *BOUNDS["windgust"], key="rc_windgust",
                                  value=st.session_state.get("rc_windgust", 18.0))
            uvindex = st.slider("UV index", *BOUNDS["uvindex"], key="rc_uvindex",
                                 value=st.session_state.get("rc_uvindex", 6))
        st.markdown("</div>", unsafe_allow_html=True)

        manual_clicked = st.button("Check these conditions →", use_container_width=True)

    auto_predict = st.session_state.pop("rc_auto_predict", False)
    predict_clicked = manual_clicked or auto_predict

    with right:
        st.markdown("""
        <div class="sc-card" style="min-height: 100%;">
            <div class="sc-card-title">☁️ Result</div>
            <div class="sc-card-sub">Appears here after you click Check — or after a random guess.</div>
        """, unsafe_allow_html=True)

        if predict_clicked:
            row = {
                'year': year, 'month': month, 'day': day,
                'max_temp_C': max_temp, 'min_temp_C': min_temp, 'avg_temp_c': avg_temp,
                'humidity': humidity, 'windgust': windgust, 'windspeed': windspeed,
                'sealevelpressure': pressure, 'cloudcover': cloudcover,
                'visibility': visibility, 'solarradiation': solarradiation, 'uvindex': uvindex
            }
            input_df = pd.DataFrame([row])[FEATURE_ORDER]

            proba = model.predict_proba(input_df)[0]
            prob_rain = float(proba[1])
            level, color, tag = risk_bucket(prob_rain)

            if auto_predict:
                st.caption("🎲 Based on a randomly guessed set of conditions")

            st.markdown(f'<span class="sc-pill">{tag}</span>', unsafe_allow_html=True)
            st.plotly_chart(render_gauge(prob_rain, color), use_container_width=True, config={"displayModeBar": False})

            if level == "high":
                headline, body = "These conditions look like Rain", "This combination closely matches rainy days in the historical data."
            elif level == "moderate":
                headline, body = "Could go either way", "Mixed signals — this combination doesn't clearly match either pattern."
            else:
                headline, body = "These conditions look like No Rain", "This combination is more typical of dry days."

            st.markdown(f"""
            <div class="sc-result" style="background: linear-gradient(135deg, {color}, {color}CC); margin-top:1rem;">
                <div class="sc-result-label">Model classification</div>
                <div class="sc-result-headline">{headline}</div>
                <div class="sc-result-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("P(No Rain)", f"{proba[0]*100:.1f}%")
            m2.metric("P(Rain)", f"{proba[1]*100:.1f}%")

            try:
                rf = model.named_steps["classifier"]
                fi = pd.DataFrame({"Feature": FEATURE_ORDER, "Importance": rf.feature_importances_}) \
                    .sort_values("Importance", ascending=False).head(6)
                fig_fi = go.Figure(go.Bar(x=fi["Importance"], y=fi["Feature"], orientation="h", marker_color=SKY))
                fig_fi.update_layout(title="What mattered most (model-wide)", height=250,
                                      margin=dict(l=10, r=10, t=40, b=10),
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      yaxis=dict(autorange="reversed"), font={"family": "Manrope", "size": 12})
                st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})
            except Exception:
                pass
        else:
            st.markdown(
                f"<div style='padding: 2.2rem 0; text-align:center; color:{MUTED};'>"
                f"Set your conditions and click <b>Check these conditions</b> — or click "
                f"<b>🎲 Guess random conditions</b> above — to see the result here.</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sc-badge-note">
        <b>This is a what-if classifier, not a forecast.</b> Whether you set the sliders
        yourself or let the app guess random values, the model answers the same question:
        "if the weather looked like this, would it be a rainy day?" — based on patterns in
        past data. It does not know any specific future date's actual weather, because
        those readings don't exist yet. To check a real upcoming day, plug in numbers from
        a live weather forecast. For a purely calendar-based outlook, see the Seasonal
        Outlook tab.
    </div>
    """, unsafe_allow_html=True)