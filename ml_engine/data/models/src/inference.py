from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

# Load artifacts (call once at import)
career_clf = joblib.load(MODELS/"career_clf.joblib")
skills_mlb = joblib.load(MODELS/"skills_mlb.joblib")
risk_model = joblib.load(MODELS/"risk_model.joblib")
tfidf = joblib.load(MODELS/"tfidf.joblib")
course_matrix = joblib.load(MODELS/"course_matrix.joblib")
courses_df = pd.read_csv(MODELS/"courses_saved.csv")
career_profiles = pd.read_csv(DATA/"career_profiles.csv")
trends_df = pd.read_csv(DATA/"trends.csv")

def predict_career(user_skills, analytic_score, creative_score, experience_years):
    skills_clean = [s.strip().lower() for s in user_skills if s.strip()]
    X_sk = skills_mlb.transform([skills_clean])
    X_num = np.array([[analytic_score, creative_score, experience_years]])
    X = np.hstack([X_sk, X_num])
    pred = career_clf.predict(X)[0]
    probs = dict(zip(career_clf.classes_, career_clf.predict_proba(X)[0].tolist()))
    return pred, probs

def recommend_courses(user_skills_str, top_n=5):
    v = tfidf.transform([user_skills_str.lower()])
    sims = cosine_similarity(v, course_matrix).flatten()
    idx = sims.argsort()[-top_n:][::-1]
    return courses_df.iloc[idx][['title','provider','url','skills']].to_dict(orient='records')

def demand_for_career(career_name):
    row = career_profiles[career_profiles['career']==career_name]
    if row.empty: return 0.7
    req_skills = row['skills'].iloc[0].split(',')
    vals = []
    for s in req_skills:
        d = trends_df[trends_df['skill']==s]['demand_score']
        vals.append(float(d.mean()) if not d.empty else 0.6)
    return float(sum(vals)/len(vals)) if vals else 0.6

def predict_risk(num_skills_matched, learning_hours_per_week, experience_years, demand_score):
    x = [[num_skills_matched, learning_hours_per_week, experience_years, demand_score]]
    return float(risk_model.predict_proba(x)[0][1])

def career_twin_template(career_name):
    row = career_profiles[career_profiles['career']==career_name]
    base = row['skills'].iloc[0].split(',') if not row.empty else []
    out=[]
    for y in range(1,6):
        if y==1:
            ms=["Learn core: "+", ".join(base[:2]), "Build 1 small project", "Finish 1 course"]
        elif y==2:
            ms=["Portfolio project", "Internship applications", "Intermediate tools"]
        elif y==3:
            ms=["Join real project", "Junior role applications", "Start specialization"]
        elif y==4:
            ms=["Lead mini-projects", "Mentor juniors", "Advanced skills"]
        else:
            ms=["Senior contributions", "Choose management or deep-tech", "Network & speak"]
        out.append({"year":y, "milestones":ms, "skills_to_learn":base[:2]})
    return out
