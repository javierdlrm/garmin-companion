"""Garmin Companion — training-readiness dashboard.

Calls the combined ``garminrecommendation`` deployment and renders a Status /
Recommendation / Why card. Styled in the Hopsworks/Garmin-Companion palette (see
assets/banner.svg). Run after notebook 3 has started the deployment::

    streamlit run streamlit_garmin_app.py
"""

import datetime as dt

import streamlit as st

import hopsworks

# --- palette (from assets/banner.svg) -------------------------------------------
TEAL = "#1EB182"
MINT = "#9bf3d4"
LIGHT = "#d8fff1"
CARD = "#0a3f2e"
STATUS = {
    "green":  {"c": "#2ee6a6", "icon": "🟢", "title": "READY", "tag": "Train hard"},
    "yellow": {"c": "#ffd166", "icon": "🟡", "title": "CAUTION", "tag": "Easy / mobility"},
    "red":    {"c": "#ff6b6b", "icon": "🔴", "title": "REST", "tag": "Recover"},
}
RECO = {
    "green": "You're recovered — a hard or high-intensity session is fine today.",
    "yellow": "Do mobility, easy zone-2, or technical practice. Avoid max-effort intervals "
              "or hard lower-body work.",
    "red": "Prioritise recovery: rest, light mobility, sleep. Skip hard training today.",
}

st.set_page_config(page_title="Garmin Companion", page_icon="🏃", layout="wide")

st.markdown(
    f"""
    <style>
      .stApp {{ background: radial-gradient(1200px 500px at 80% -10%, #0f6f50 0%, #06281d 60%); }}
      .hw-band {{
        background: linear-gradient(90deg,#06281d 0%,#0f6f50 55%,{TEAL} 100%);
        border-radius: 16px; padding: 1.3rem 1.6rem; margin-bottom: 1.2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
      }}
      .hw-band h1 {{ color:#fff; margin:0; font-size:2rem; font-weight:800; letter-spacing:-1px; }}
      .hw-band p {{ color:{LIGHT}; margin:.25rem 0 0; font-size:1rem; }}
      .hw-card {{
        background:{CARD}; border-radius:14px; padding:1.1rem 1.2rem; height:100%;
        border-top:4px solid var(--accent,{TEAL});
        box-shadow:0 4px 18px rgba(0,0,0,0.25);
      }}
      .hw-card .lbl {{ color:{MINT}; font-size:.8rem; text-transform:uppercase; letter-spacing:1px; font-weight:700; }}
      .hw-card .val {{ color:#fff; font-size:1.9rem; font-weight:800; margin-top:.2rem; }}
      .hw-card .sub {{ color:{LIGHT}; font-size:.85rem; opacity:.85; }}
      .hw-status {{
        border-radius:18px; padding:1.4rem 1.6rem; margin:.4rem 0 1rem;
        background:linear-gradient(135deg, rgba(0,0,0,.25), var(--accent));
        box-shadow:0 10px 36px rgba(0,0,0,.4);
      }}
      .hw-status .big {{ font-size:2.6rem; font-weight:900; color:#06281d; margin:0; }}
      .hw-status .tag {{ font-size:1rem; font-weight:700; color:#06281d; opacity:.8; }}
      .hw-status .reco {{ font-size:1.05rem; color:#06281d; margin-top:.4rem; font-weight:600; }}
      .bar-bg {{ background:rgba(255,255,255,.12); border-radius:8px; height:10px; width:100%; }}
      .bar-fg {{ height:10px; border-radius:8px; }}
      div[data-testid="stMetricValue"] {{ color:{TEAL}; }}
    </style>
    <div class="hw-band">
      <h1>🏃 Garmin Companion</h1>
      <p>Training Readiness · Stress Anomaly · Recovery — <b>not medical advice</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_clients():
    project = hopsworks.login()
    ms = project.get_model_serving()
    dep = ms.get_deployment("garminrecommendation")
    if not dep.is_running():
        dep.start(await_running=300)
    return project, dep


@st.cache_data(ttl=600)
def load_activities(user_id):
    """Recent activities -> [(label, activity_id)] for the suggestion dropdown."""
    project, _ = get_clients()
    fs = project.get_feature_store()
    df = fs.get_feature_group("fg_garmin_activity_raw", 1).read()
    df = df[df["user_id"] == user_id].sort_values("activity_start_time", ascending=False)
    opts = []
    for _, r in df.head(25).iterrows():
        d = str(r["activity_start_time"])[:10]
        typ = str(r.get("activity_type") or "activity").replace("_", " ")
        dur = r.get("duration_min")
        dur_s = f" · {dur:.0f}min" if dur == dur else ""  # NaN-safe
        opts.append((f"{d} · {typ}{dur_s}", str(r["activity_id"])))
    return opts


def card(col, label, value, sub, accent):
    col.markdown(
        f"""<div class="hw-card" style="--accent:{accent}">
              <div class="lbl">{label}</div>
              <div class="val">{value}</div>
              <div class="sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def bar(col, label, frac, accent):
    pct = max(0.0, min(1.0, frac)) * 100
    col.markdown(
        f"""<div style="margin-top:.4rem"><div class="sub" style="color:{LIGHT};font-size:.8rem">{label}</div>
              <div class="bar-bg"><div class="bar-fg" style="width:{pct:.0f}%;background:{accent}"></div></div>
            </div>""",
        unsafe_allow_html=True,
    )


# --- inputs ----------------------------------------------------------------------
c1, c2, c3 = st.columns([1, 1, 2])
user_id = c1.text_input("Athlete", "javier")
date = c2.date_input("Date", dt.date(2026, 6, 3))

try:
    acts = load_activities(user_id)
except Exception as exc:
    acts = []
    st.warning(f"Could not load activities: {exc}")

act_labels = ["— Daily check (no workout) —"] + [a[0] for a in acts]
choice = c3.selectbox("Workout for recovery estimate", act_labels,
                      help="Pick a recent activity to also estimate post-workout recovery time.")
activity_id = None
if choice != act_labels[0]:
    activity_id = dict((a[0], a[1]) for a in acts)[choice]

go = st.button("Get recommendation", type="primary", use_container_width=True)

# --- prediction ------------------------------------------------------------------
if go:
    req = {"user_id": user_id, "date": date.isoformat()}
    if activity_id:
        req["activity_id"] = activity_id
    try:
        _, dep = get_clients()
        with st.spinner("Scoring readiness, stress & recovery…"):
            out = dep.predict(inputs=[req])
        pred = out.get("predictions", out)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    status = pred.get("overall_status", "yellow")
    s = STATUS.get(status, STATUS["yellow"])

    st.markdown(
        f"""<div class="hw-status" style="--accent:{s['c']}">
              <p class="big">{s['icon']} {s['title']}</p>
              <div class="tag">{s['tag']}</div>
              <div class="reco">{RECO.get(status,'')}</div>
            </div>""",
        unsafe_allow_html=True,
    )

    readiness = pred.get("readiness", {}) or {}
    stress = pred.get("stress", {}) or {}
    recovery = pred.get("recovery", {}) or {}

    m1, m2, m3 = st.columns(3)
    r_cls = str(readiness.get("class", "—")).capitalize()
    r_conf = readiness.get("confidence")
    card(m1, "Readiness", f"{STATUS.get(readiness.get('class'),{}).get('icon','')} {r_cls}",
         "model: weak-label classifier", STATUS.get(readiness.get("class"), {}).get("c", TEAL))
    if isinstance(r_conf, (int, float)):
        bar(m1, f"confidence {r_conf:.0%}", r_conf, STATUS.get(readiness.get("class"), {}).get("c", TEAL))

    s_lvl = str(stress.get("level", "—")).capitalize()
    s_score = stress.get("score")
    s_accent = {"high": "#ff6b6b", "moderate": "#ffd166", "normal": "#2ee6a6"}.get(stress.get("level"), TEAL)
    card(m2, "Stress anomaly", s_lvl, "intra-day vs. your time-of-day baseline", s_accent)
    if isinstance(s_score, (int, float)):
        bar(m2, f"anomaly score {s_score:.2f}", s_score, s_accent)

    rem = recovery.get("predicted_recovery_hours_remaining")
    if isinstance(rem, (int, float)):
        rec_accent = "#ff6b6b" if rem > 24 else "#ffd166" if rem > 8 else "#2ee6a6"
        card(m3, "Recovery", f"~{rem:.0f} h", "until post-workout markers normalise", rec_accent)
    else:
        card(m3, "Recovery", "—", "pick a workout above to estimate", "#7ad8ff")

    with st.expander("Why — raw model output"):
        st.json(pred)
