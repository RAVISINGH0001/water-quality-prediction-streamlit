import streamlit as st
import numpy as np
import pickle
import os

# ======================================================
# 1) PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Water Quality Prediction App",
    page_icon="💧",
    layout="wide"
)

# ======================================================
# 2) LOAD MODEL + SCALER (SAFE PATH)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "water_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# ======================================================
# 3) PRESET EXAMPLES (LOW / MEDIUM / HIGH)
# ======================================================
LOW_EXAMPLE = {
    "ph": 7.00, "hardness": 150.0, "solids": 20000.0, "chloramines": 7.0,
    "sulfate": 250.0, "conductivity": 400.0, "organic_carbon": 10.0,
    "trihalomethanes": 60.0, "turbidity": 4.0
}

MEDIUM_EXAMPLE = {
    "ph": 9.20, "hardness": 150.0, "solids": 20000.0, "chloramines": 7.0,
    "sulfate": 250.0, "conductivity": 400.0, "organic_carbon": 10.0,
    "trihalomethanes": 60.0, "turbidity": 4.0
}

HIGH_EXAMPLE = {
    "ph": 9.80, "hardness": 320.0, "solids": 59000.0, "chloramines": 12.5,
    "sulfate": 410.0, "conductivity": 720.0, "organic_carbon": 20.0,
    "trihalomethanes": 107.0, "turbidity": 8.0
}

# ======================================================
# 4) SESSION STATE DEFAULT
# ======================================================
if "ph" not in st.session_state:
    st.session_state.update(LOW_EXAMPLE)

# ======================================================
# 5) CSS FOR PREMIUM UI
# ======================================================
st.markdown("""
<style>
.stApp{
    background: radial-gradient(circle at top left, #1f2937 0%, #0b1220 60%, #050814 100%);
    color: #e5e7eb;
}
.title{font-size: 44px; font-weight: 900; margin-bottom: 0;}
.subtitle{font-size: 18px; color: #cbd5e1; margin-top: 6px;}
.card{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.25);
}
.label{font-size: 14px; color: #94a3b8;}
.big-value{font-size: 30px; font-weight: 900; color: #e5e7eb;}
section[data-testid="stSidebar"]{
    background: rgba(2, 6, 23, 0.85);
    border-right: 1px solid rgba(255,255,255,0.08);
}
.stButton > button{
    border-radius: 14px;
    padding: 12px 15px;
    font-size: 16px;
    font-weight: 800;
    border: 1px solid rgba(255,255,255,0.12);
}
hr{
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 14px 0;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 6) FUNCTIONS (RULE RISK + ML LABEL)
# ======================================================
def rule_based_risk(ph, turbidity, trihalomethanes, solids):
    danger_signals = 0
    reasons = []

    if ph < 6.5 or ph > 8.5:
        danger_signals += 1
        reasons.append("⚠️ pH is outside safe range (6.5–8.5)")

    if turbidity > 5:
        danger_signals += 1
        reasons.append("⚠️ Turbidity is high (>5)")

    if trihalomethanes > 80:
        danger_signals += 1
        reasons.append("⚠️ Trihalomethanes is high (>80)")

    if solids > 50000:
        danger_signals += 1
        reasons.append("⚠️ Solids are very high (>50000)")

    if danger_signals >= 2:
        rules_risk = "HIGH"
    elif danger_signals == 1:
        rules_risk = "MEDIUM"
    else:
        rules_risk = "LOW"

    return rules_risk, danger_signals, reasons


def ml_risk_label(score):
    if score >= 70:
        return "LOW", "🟢", "ML indicates LOW risk (high safe probability)."
    elif score >= 40:
        return "MEDIUM", "🟡", "ML indicates MEDIUM risk (moderate safe probability)."
    else:
        return "HIGH", "🔴", "ML indicates HIGH risk (low safe probability)."


# ======================================================
# 7) HEADER
# ======================================================
st.markdown('<div class="title">💧 Water Quality Prediction App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict whether water is <b>Safe (Potable)</b> or <b>Not Safe</b> using Machine Learning + Health Rules.</div>',
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ======================================================
# 8) SIDEBAR: TRY EXAMPLE VALUES
# ======================================================
st.sidebar.markdown("## 🧪 Try Example Values")
st.sidebar.caption("Click a preset to auto-fill sliders (for quick testing).")

btn1, btn2, btn3 = st.sidebar.columns([1.5, 1.5, 1.5])

low_click = btn1.button("🟢 Low", use_container_width=True)
med_click = btn2.button("🟡 Med", use_container_width=True)
high_click = btn3.button("🔴 High", use_container_width=True)

if low_click:
    st.session_state.update(LOW_EXAMPLE)
    st.sidebar.success("✅ Loaded LOW example")

if med_click:
    st.session_state.update(MEDIUM_EXAMPLE)
    st.sidebar.warning("✅ Loaded MEDIUM example")

if high_click:
    st.session_state.update(HIGH_EXAMPLE)
    st.sidebar.error("✅ Loaded HIGH example")

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Input Water Parameters")

# ======================================================
# 9) SIDEBAR SLIDERS (SESSION KEYS)
# ======================================================
ph = st.sidebar.slider("pH", 0.0, 14.0, float(st.session_state["ph"]), key="ph")
hardness = st.sidebar.slider("Hardness", 0.0, 400.0, float(st.session_state["hardness"]), key="hardness")
solids = st.sidebar.slider("Solids (ppm)", 0.0, 70000.0, float(st.session_state["solids"]), key="solids")
chloramines = st.sidebar.slider("Chloramines", 0.0, 15.0, float(st.session_state["chloramines"]), key="chloramines")
sulfate = st.sidebar.slider("Sulfate", 0.0, 500.0, float(st.session_state["sulfate"]), key="sulfate")
conductivity = st.sidebar.slider("Conductivity", 0.0, 800.0, float(st.session_state["conductivity"]), key="conductivity")
organic_carbon = st.sidebar.slider("Organic Carbon", 0.0, 30.0, float(st.session_state["organic_carbon"]), key="organic_carbon")
trihalomethanes = st.sidebar.slider("Trihalomethanes", 0.0, 120.0, float(st.session_state["trihalomethanes"]), key="trihalomethanes")
turbidity = st.sidebar.slider("Turbidity", 0.0, 10.0, float(st.session_state["turbidity"]), key="turbidity")

# ======================================================
# 10) PREPARE INPUT ARRAY
# ======================================================
input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                        conductivity, organic_carbon, trihalomethanes, turbidity]])

# ======================================================
# 11) MAIN LAYOUT
# ======================================================
left_col, right_col = st.columns([1.3, 1])

# ---------------- LEFT PANEL ----------------
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🧪 Selected Water Test Values")
    st.write("These are the values you entered (from sidebar):")

    r1 = st.columns(3)
    r1[0].markdown(f"<div class='label'>pH</div><div class='big-value'>{ph:.2f}</div>", unsafe_allow_html=True)
    r1[1].markdown(f"<div class='label'>Hardness</div><div class='big-value'>{hardness:.2f}</div>", unsafe_allow_html=True)
    r1[2].markdown(f"<div class='label'>Solids</div><div class='big-value'>{solids:.2f}</div>", unsafe_allow_html=True)

    r2 = st.columns(3)
    r2[0].markdown(f"<div class='label'>Chloramines</div><div class='big-value'>{chloramines:.2f}</div>", unsafe_allow_html=True)
    r2[1].markdown(f"<div class='label'>Sulfate</div><div class='big-value'>{sulfate:.2f}</div>", unsafe_allow_html=True)
    r2[2].markdown(f"<div class='label'>Conductivity</div><div class='big-value'>{conductivity:.2f}</div>", unsafe_allow_html=True)

    r3 = st.columns(3)
    r3[0].markdown(f"<div class='label'>Organic Carbon</div><div class='big-value'>{organic_carbon:.2f}</div>", unsafe_allow_html=True)
    r3[1].markdown(f"<div class='label'>Trihalomethanes</div><div class='big-value'>{trihalomethanes:.2f}</div>", unsafe_allow_html=True)
    r3[2].markdown(f"<div class='label'>Turbidity</div><div class='big-value'>{turbidity:.2f}</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RIGHT PANEL ----------------
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🔍 Prediction Panel")

    if st.button("✅ Predict Water Quality", use_container_width=True):

        # ✅ Scale input
        input_scaled = scaler.transform(input_data)

        # ✅ ML prediction + probability score
        prediction = model.predict(input_scaled)[0]
        probability_safe = model.predict_proba(input_scaled)[0][1]
        score = int(probability_safe * 100)

        # ✅ ML risk label
        ml_level, emoji, ml_msg = ml_risk_label(score)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ✅ ML output
        st.markdown("### 🤖 ML Prediction Result")
        if prediction == 1:
            st.success("✅ ML Prediction: Water is SAFE (Potable)")
        else:
            st.error("❌ ML Prediction: Water is NOT SAFE (Not Potable)")

        if ml_level == "LOW":
            st.success(f"{emoji} ML Risk Level: LOW")
        elif ml_level == "MEDIUM":
            st.warning(f"{emoji} ML Risk Level: MEDIUM")
        else:
            st.error(f"{emoji} ML Risk Level: HIGH")

        st.caption(ml_msg)

        st.markdown("### 💡 Water Quality Score (ML)")
        st.progress(score)
        st.write(f"**Score:** {score}/100")

        # ✅ Rule Based Risk
        rules_risk, danger_count, reasons = rule_based_risk(ph, turbidity, trihalomethanes, solids)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🧪 Risk Level (Health Rules)")
        st.write(f"✅ **Danger Signals Detected:** {danger_count}")

        if rules_risk == "LOW":
            st.success("🟢 Rules Risk: LOW (Good water quality)")
        elif rules_risk == "MEDIUM":
            st.warning("🟡 Rules Risk: MEDIUM (Be careful)")
        else:
            st.error("🔴 Rules Risk: HIGH (Unsafe water)")

        st.markdown("### 🧠 Why this risk?")
        if not reasons:
            st.write("✅ All values are within safe ranges.")
        else:
            for r in reasons:
                st.write(r)

        # ✅ FINAL decision override
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### ✅ Final Decision")

        final_prediction = prediction
        if rules_risk == "HIGH":
            final_prediction = 0  # override for safety

        if final_prediction == 1:
            st.success("✅ FINAL: Water is SAFE to drink")
        else:
            st.error("❌ FINAL: Water is NOT SAFE to drink")

        # ✅ Decision Summary Panel
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🧾 Decision Summary (ML vs Rules vs FINAL)")

        s1, s2, s3 = st.columns(3)

        with s1:
            if prediction == 1:
                st.success("✅ ML says: SAFE")
            else:
                st.error("❌ ML says: NOT SAFE")
            st.caption(f"ML score: {score}/100")

        with s2:
            if rules_risk == "LOW":
                st.success("🟢 Rules: LOW risk")
            elif rules_risk == "MEDIUM":
                st.warning("🟡 Rules: MEDIUM risk")
            else:
                st.error("🔴 Rules: HIGH risk")
            st.caption(f"Danger signals: {danger_count}")

        with s3:
            if final_prediction == 1:
                st.success("✅ FINAL: SAFE")
            else:
                st.error("❌ FINAL: NOT SAFE")

            if (prediction == 1) and (rules_risk == "HIGH"):
                st.caption("⚠️ Overridden by safety rules")
            else:
                st.caption("Final based on ML + rules")

        # ✅ Last info line (your requested line)
        st.info("✅ ML gives prediction based on learned dataset patterns. 🔒 Safety rules ensure real-world reliability.")

        # ✅ DOWNLOAD REPORT (FULL DETAILS)
        st.markdown("<hr>", unsafe_allow_html=True)
        report = f"""
WATER QUALITY REPORT

ML PREDICTION:
- Result: {"SAFE (Potable)" if prediction == 1 else "NOT SAFE (Not Potable)"}
- ML Score (Safe probability): {score}/100
- ML Risk Level: {ml_level}

RULE-BASED CHECK (HEALTH RULES):
- Risk Level: {rules_risk}
- Danger Signals Detected: {danger_count}

FINAL DECISION:
- {"SAFE" if final_prediction == 1 else "NOT SAFE"}
- Note: {"ML overridden due to HIGH risk values" if (prediction == 1 and rules_risk == "HIGH") else "Final based on ML + rules"}

INPUT VALUES:
pH: {ph}
Hardness: {hardness}
Solids: {solids}
Chloramines: {chloramines}
Sulfate: {sulfate}
Conductivity: {conductivity}
Organic Carbon: {organic_carbon}
Trihalomethanes: {trihalomethanes}
Turbidity: {turbidity}

REASONS:
{chr(10).join(reasons) if reasons else "No issues detected"}
"""
        st.download_button(
            "📄 Download Report",
            data=report,
            file_name="water_quality_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    else:
        st.info("👈 Use Example Buttons OR set values from sidebar, then click **Predict Water Quality**.")

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# 12) FOOTER
# ======================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built using Machine Learning & Streamlit")



