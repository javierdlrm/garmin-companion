"""Garmin Companion — minimal dashboard.

Optional capstone (mirrors draft §1 / §15.4). Calls the combined deployment and renders
the Status / Recommendation / Why card. Run after notebook 3 has started the deployment::

    streamlit run streamlit_garmin_app.py
"""

import streamlit as st

import hopsworks

RECO = {
    "green": "You're recovered — a hard or high-intensity session is fine today.",
    "yellow": "Do mobility, easy zone 2, or technical practice. Avoid max-effort intervals "
    "or hard lower-body work.",
    "red": "Prioritise recovery: rest, light mobility, sleep. Skip hard training today.",
}
COLOR = {"green": "🟢", "yellow": "🟡", "red": "🔴"}


@st.cache_resource
def get_deployment():
    project = hopsworks.login()
    ms = project.get_model_serving()
    dep = ms.get_deployment("garminrecommendation")
    if not dep.is_running():
        dep.start(await_running=300)
    return dep


def main():
    st.set_page_config(page_title="Garmin Companion", page_icon="🏃")
    st.title("🏃 Garmin Companion — Training Readiness")
    st.caption("Personal training-readiness & recovery monitoring. **Not medical advice.**")

    user_id = st.text_input("User", "javier")
    date = st.text_input("Date (YYYY-MM-DD)", "2024-06-30")
    activity_id = st.text_input("Activity id (optional, for recovery estimate)", "")

    if st.button("Get recommendation"):
        req = {"user_id": user_id, "date": date}
        if activity_id.strip():
            req["activity_id"] = activity_id.strip()
        dep = get_deployment()
        out = dep.predict(inputs=[req])
        pred = out.get("predictions", out)

        status = pred.get("overall_status", "yellow")
        st.header(f"{COLOR.get(status, '🟡')} Status: {status.capitalize()}")
        st.subheader("Recommendation")
        st.write(RECO.get(status, ""))

        st.subheader("Why")
        readiness = pred.get("readiness", {})
        stress = pred.get("stress", {})
        recovery = pred.get("recovery", {})
        st.write(f"- Readiness: **{readiness.get('class')}** "
                 f"(confidence {readiness.get('confidence')})")
        st.write(f"- Stress anomaly: **{stress.get('level')}** (score {stress.get('score')})")
        rem = recovery.get("predicted_recovery_hours_remaining")
        if rem is not None:
            st.write(f"- Post-workout recovery: ~**{rem:.0f}h** remaining")
        st.json(pred)


if __name__ == "__main__":
    main()
