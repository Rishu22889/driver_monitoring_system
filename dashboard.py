import streamlit as st
import requests
import time
import pandas as pd
import base64

st.set_page_config(layout="wide")
st.title("🚗 Driver Monitoring Dashboard")

BACKEND = "http://127.0.0.1:8000"

if "history" not in st.session_state:
    st.session_state.history = []

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

if "last_risk_level" not in st.session_state:
    st.session_state.last_risk_level = "Low"

if "alert_sound_base64" not in st.session_state:
    with open("data/assets/alert.mp3", "rb") as f:
        audio_bytes = f.read()
        st.session_state.alert_sound_base64 = base64.b64encode(audio_bytes).decode()

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("▶ Start Monitoring", use_container_width=True):
        try:
            requests.post(f"{BACKEND}/start")
            st.session_state.monitoring = True
            st.session_state.history = []
        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    if st.button("⏹ Stop Monitoring", use_container_width=True):
        try:
            requests.post(f"{BACKEND}/stop")
        except:
            pass
        st.session_state.monitoring = False

if st.session_state.monitoring:
    try:
        response = requests.get(f"{BACKEND}/latest", timeout=1)
        data = response.json()

        if data:
            row = data[0]

            fatigue = float(row[0])
            distraction = float(row[1])
            emotion = float(row[2])
            risk_score = float(row[3])
            risk_level = row[4]

            st.session_state.history.append(risk_score)
            if len(st.session_state.history) > 50:
                st.session_state.history.pop(0)

            st.markdown("## 🚦 Driver Safety Status")

            if risk_level == "High":
                st.error(f"🚨 HIGH RISK  |  Score: {risk_score:.2f}")

                if st.session_state.last_risk_level != "High":
                    audio_html = f"""
                    <audio autoplay>
                        <source src="data:audio/mp3;base64,{st.session_state.alert_sound_base64}" type="audio/mp3">
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)

            elif risk_level == "Medium":
                st.warning(f"⚠️ MEDIUM RISK  |  Score: {risk_score:.2f}")
            else:
                st.success(f"✅ LOW RISK  |  Score: {risk_score:.2f}")

            st.session_state.last_risk_level = risk_level

            st.progress(min(risk_score, 1.0))

            st.divider()
            st.markdown("### Component Breakdown")

            c1, c2, c3 = st.columns(3)
            c1.metric("Fatigue", f"{fatigue:.2f}")
            c2.metric("Distraction", f"{distraction:.2f}")
            c3.metric("Emotion Risk", f"{emotion:.2f}")

            st.divider()
            st.subheader("📈 Risk Trend (Recent 50 Frames)")

            if len(st.session_state.history) > 5:
                df = pd.DataFrame({"Risk Score": st.session_state.history})
                df["Smoothed Risk"] = df["Risk Score"].rolling(window=5).mean()
                st.line_chart(df)
            else:
                st.info("Collecting data...")

    except Exception as e:
        st.error(f"⚠ Error: {e}")

    time.sleep(1)
    st.rerun()

else:
    st.info("Monitoring Stopped")