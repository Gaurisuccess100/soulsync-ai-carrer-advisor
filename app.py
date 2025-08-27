import streamlit as st
from ml_engine.src.inference import predict_career, recommend_courses, predict_risk, demand_for_career, career_twin_template
import pandas as pd

st.set_page_config(page_title="SolSync AI — ML Demo", layout="wide")
st.title("SolSync AI — Career Twin (ML Demo)")

skills = st.multiselect("Select your skills",
    ["python","ml","statistics","sql","tensorflow","cloud","html","css","javascript","react","uiux","figma","security","networking"])

col1, col2, col3 = st.columns(3)
with col1:
    analytic = st.slider("Analytical", 0.0, 1.0, 0.6)
with col2:
    creative = st.slider("Creative", 0.0, 1.0, 0.4)
with col3:
    exp = st.number_input("Experience (years)", 0.0, 10.0, 0.5)

if st.button("Generate Results"):
    career, probs = predict_career(skills, analytic, creative, exp)
    st.success(f"Recommended career: **{career}**")
    st.write("Probabilities:"); st.json(probs)

    st.subheader("Career Twin (template)")
    for t in career_twin_template(career):
        st.markdown(f"**Year {t['year']}**")
        for m in t["milestones"]:
            st.write("- " + m)

    st.subheader("Courses")
    recs = recommend_courses(",".join(skills) if skills else career)
    for r in recs:
        st.write(f"- [{r['title']}]({r['url']}) — {r['provider']} (skills: {r['skills']})")

    st.subheader("Risk")
    # estimate matched skills for risk calc
    req = []
    try:
        cp = pd.read_csv("ml_engine/data/career_profiles.csv")
        req = cp[cp['career']==career]['skills'].iloc[0].split(',')
    except Exception:
        pass
    matched = sum(1 for s in req if s in skills)
    demand = demand_for_career(career)
    risk = predict_risk(matched, 2.0, exp, demand)
    st.write(f"Risk probability: **{risk*100:.1f}%**")
