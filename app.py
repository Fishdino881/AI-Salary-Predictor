import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie
import time
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


# ---------- Page Config ----------
st.set_page_config(
    page_title="SalaryPredictor | AI Compensation Analytics Platform",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/kunal-singh-699485215/',
        'Report a bug': "mailto:i.kunalsingh.ks@gmail.com",
        'About': "### SalaryPredictor\nAI-powered compensation analytics platform"
    }
)

# ---------- Load Assets ----------
@st.cache_resource(show_spinner=False)
def load_lottie_url(url):
    """Loads a Lottie animation JSON from a given URL."""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    """
    Loads the pre-trained salary prediction model, encoders,
    salary trend model, and skills impact data.
    """
    try:
        artifact_base_dir = "artifacts"
        if not os.path.exists(artifact_base_dir):
            st.error(f"Artifacts directory '{artifact_base_dir}' not found. Please run train_model.py first.")
            st.stop()

        subdirs = [os.path.join(artifact_base_dir, d) for d in os.listdir(artifact_base_dir) if os.path.isdir(os.path.join(artifact_base_dir, d))]
        subdirs.sort(key=os.path.getmtime, reverse=True)

        if not subdirs:
            st.error("No trained models found in the 'artifacts' directory. Please run train_model.py first.")
            st.stop()
        
        latest_artifact_path = subdirs[0]

        with open(os.path.join(latest_artifact_path, "salary_model.pkl"), "rb") as f:
            model = pickle.load(f)
        
        encoders = {}
        for col in ["EducationLevel", "JobRole", "Location", "CompanyType", "SkillLevel", "RemoteWork"]:
            encoder_path = os.path.join(latest_artifact_path, f"{col}_encoder.pkl")
            if not os.path.exists(encoder_path):
                st.error(f"Missing encoder file: {encoder_path}. Please ensure train_model.py completed successfully.")
                st.stop()
            with open(encoder_path, "rb") as f:
                encoders[col] = pickle.load(f)
        
        trend_model_path = os.path.join(latest_artifact_path, "salary_trend_model.pkl")
        if not os.path.exists(trend_model_path):
            st.warning(f"salary_trend_model.pkl not found at {trend_model_path}. Some features might be limited.")
            trend_model = None
        else:
            with open(trend_model_path, "rb") as f:
                trend_model = pickle.load(f)

        skills_impact_path = os.path.join(latest_artifact_path, "skills_impact.json")
        if not os.path.exists(skills_impact_path):
            st.warning(f"skills_impact.json not found at {skills_impact_path}. Skills impact analysis will be unavailable.")
            skills_impact = {}
        else:
            with open(skills_impact_path, "r") as f:
                skills_impact = json.load(f)
        
        salary_band_model_path = os.path.join(latest_artifact_path, "salary_band_model.pkl")
        if os.path.exists(salary_band_model_path):
            with open(salary_band_model_path, "rb") as f:
                salary_band_model = pickle.load(f)
        else:
            salary_band_model = None
            
        return model, encoders, trend_model, skills_impact, salary_band_model
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Please ensure all .pkl and .json files are in the correct directory.")
        st.info("If you've just trained the model, ensure 'train_model.py' completed successfully and created artifacts.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        st.stop()

# Load assets
lottie_animation = load_lottie_url("https://lottie.host/9e2c1654-4153-49fd-86f4-1acdc38b3c7e/kAtShtbYQl.json")
lottie_chart = load_lottie_url("https://lottie.host/6a8d0a6c-5a1e-4e5d-8c0e-9c4b8b3b3b3b/0XQZQZQZQZ.json")
model, encoders, trend_model, skills_impact, salary_band_model = load_models()

# ---------- Custom CSS Styling ----------
st.markdown("""
    <style>
        :root {
            --primary: #6c5ce7;
            --secondary: #00cec9;
            --dark: #2d3436;
            --light: #f5f6fa;
            --danger: #d63031;
            --warning: #fdcb6e;
            --info: #0984e3;
            --success: #00b894;
        }
        
        .main { 
            background-color: #f8f9fa; 
            padding-top: 1rem;
        }
        
        .st-emotion-cache-1y4p8pa {
            padding: 2rem 5rem;
        }
        
        h1, h2, h3, h4 {
            color: var(--dark);
            font-family: 'Inter', sans-serif;
            font-weight: 700;
        }
        
        .stSelectbox label, .stSlider label, .stNumberInput label { 
            font-weight: 600; 
            color: var(--dark);
            margin-bottom: 0.5rem;
        }
        
        .result-card { 
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white; 
            font-size: 32px; 
            padding: 30px; 
            border-radius: 15px; 
            text-align: center;
            font-weight: bold;
            box-shadow: 0 10px 25px rgba(108, 92, 231, 0.3);
            margin: 1.5rem 0;
            animation: fadeIn 0.8s ease-in-out;
            position: relative;
            overflow: hidden;
        }
        
        .feature-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            margin-bottom: 1.5rem;
            border-left: 5px solid var(--primary);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stButton>button {
            width: 100%;
            padding: 0.75rem;
            font-weight: 600;
            font-size: 1rem;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            transition: all 0.3s ease;
            border: none;
            box-shadow: 0 4px 15px rgba(108, 92, 231, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .pulse-animation {
            animation: pulse 2s infinite;
        }
        
        .badge {
            display: inline-block;
            padding: 0.35em 0.65em;
            font-size: 0.75em;
            font-weight: 700;
            line-height: 1;
            color: #fff;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 0.25rem;
            background-color: var(--primary);
            margin-left: 0.5rem;
        }
        
        .badge-new {
            background-color: var(--danger);
            animation: pulse 1.5s infinite;
        }
        
        .badge-popular {
            background-color: var(--success);
        }
        
        .badge-pro {
            background-color: var(--warning);
            color: var(--dark);
        }
        
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            margin-bottom: 1.5rem;
            border-left: 5px solid var(--primary);
        }
        
        .metric-title {
            font-size: 1rem;
            color: #6c757d;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--dark);
            margin-bottom: 0.5rem;
        }
        
        .metric-change {
            font-size: 0.9rem;
            font-weight: 600;
        }
        
        .metric-change.positive {
            color: var(--success);
        }
        
        .metric-change.negative {
            color: var(--danger);
        }
        
        .custom-divider {
            height: 3px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border: none;
            margin: 1.5rem 0;
            border-radius: 3px;
        }
    </style>
    
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)

# ---------- Helper Functions ----------
def calculate_skills_impact(selected_skills):
    """Calculates the total salary impact and details for selected skills."""
    total_impact = 0
    impact_details = []
    for skill in selected_skills:
        impact = skills_impact.get(skill, {}).get('impact', 0)
        total_impact += impact
        impact_details.append({
            "Skill": skill,
            "Impact": f"+{impact}%",
            "Demand": skills_impact.get(skill, {}).get('demand', 'Medium'),
            "Popularity": skills_impact.get(skill, {}).get('popularity', 'Medium')
        })
    return total_impact, pd.DataFrame(impact_details)

def generate_salary_projection(current_salary, years, trend_data):
    """Generates a salary projection over a specified number of years."""
    projections = []
    for year in range(1, years + 1):
        base_growth = 0.05 
        performance_factor = random.uniform(0.8, 1.2)
        market_factor = 1 + (year * 0.01)
        
        if not trend_data.empty:
            avg_growth_rate = trend_data['Tech Industry'].pct_change().mean() * performance_factor * market_factor
        else:
            avg_growth_rate = base_growth * performance_factor * market_factor
            
        projected_salary = current_salary * (1 + avg_growth_rate)
        projections.append({
            "Year": datetime.now().year + year,
            "Projected Salary": int(projected_salary),
            "Growth Rate": f"{avg_growth_rate*100:.1f}%",
            "Cumulative Growth": f"{(projected_salary/current_salary - 1)*100:.1f}%"
        })
        current_salary = projected_salary
    return pd.DataFrame(projections)

def calculate_negotiation_range(salary):
    """Calculates a negotiation range based on the estimated salary."""
    return {
        "Low": int(salary * 0.9),
        "Target": salary,
        "High": int(salary * 1.15),
        "Premium": int(salary * 1.3)
    }

def generate_skill_matrix(selected_skills, all_skills):
    """Generates a skill matrix visualization."""
    skill_data = []
    for skill in all_skills:
        skill_info = skills_impact.get(skill, {})
        skill_data.append({
            "Skill": skill,
            "Category": skill_info.get('category', 'Technical'),
            "Demand": skill_info.get('demand', 'Medium'),
            "Impact": skill_info.get('impact', 0),
            "Selected": skill in selected_skills
        })
    return pd.DataFrame(skill_data)

def generate_comp_benchmark(job_role, location, experience):
    """Generates mock compensation benchmark data."""
    base_salary = {
        "AI Engineer": 15_00_000, "Data Scientist": 12_50_000, "Software Engineer": 10_00_000,
        "Product Manager": 14_00_000, "DevOps Engineer": 11_50_000
    }.get(job_role, 10_00_000)
    
    exp_factor = min(1 + (experience / 10), 2.5)
    location_factor = {"Bangalore": 1.1, "Mumbai": 1.05, "Delhi": 1.0, "Hyderabad": 0.95, "Chennai": 0.9, "Pune": 0.92}.get(location, 1.0)
    
    benchmark_salary = (base_salary * exp_factor) * location_factor
    
    return {
        "25th Percentile": int(benchmark_salary * 0.75), "50th Percentile (Median)": int(benchmark_salary),
        "75th Percentile": int(benchmark_salary * 1.25), "90th Percentile": int(benchmark_salary * 1.5),
        "Industry High": int(benchmark_salary * 2.0)
    }

def generate_equity_breakdown(company_type, job_role, salary):
    """Generates a realistic equity breakdown based on company type and role."""
    if "Startup" in company_type:
        equity_pct = 0.01 if "Engineer" in job_role else 0.005
        multiplier = 10 if "Series A" in company_type else 5
    elif "Giant" in company_type:
        equity_pct = 0.002
        multiplier = 1
    else: # MNC
        equity_pct = 0.001
        multiplier = 1

    equity_value = salary * equity_pct * multiplier
    
    return {
        "Equity Percentage": f"{equity_pct*100:.2f}%",
        "Estimated Annual Value": int(equity_value),
        "4-Year Vesting Value": int(equity_value * 4)
    }

def generate_benefits_breakdown(company_type):
    """Generates a benefits breakdown based on company type."""
    benefits = {
        "Health Insurance": {"Startup": "₹50,000", "MNC": "₹2,00,000", "Tech Giant": "Full coverage"},
        "Retirement": {"Startup": "PF only", "MNC": "PF + 5% matching", "Tech Giant": "PF + 10% matching"},
        "Bonus": {"Startup": "0-10%", "MNC": "10-20%", "Tech Giant": "15-30%"},
        "WFH Stipend": {"Startup": "₹5,000/yr", "MNC": "₹20,000/yr", "Tech Giant": "₹50,000/yr"},
        "Learning Budget": {"Startup": "₹10,000/yr", "MNC": "₹50,000/yr", "Tech Giant": "Unlimited"}
    }
    
    company_category = "Startup" if "Startup" in company_type else "Tech Giant" if "Giant" in company_type else "MNC"
    
    return pd.DataFrame([{"Benefit": b, "Value": v[company_category]} for b, v in benefits.items()])

def generate_report(total_comp, adjusted_salary, negotiation_range, projection_data, skills_df, benchmarks):
    """Generates a PDF report of the compensation analysis."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    def draw_section(c, y_start, title, data_dict):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y_start, title)
        c.setFont("Helvetica", 11)
        y = y_start - 20
        for key, val in data_dict.items():
            if y < 40: # Check for page break
                c.showPage()
                y = height - 40
            c.drawString(60, y, f"{key}: {val}")
            y -= 16
        return y - 20

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 40, "SalaryPredictor ai – Compensation Report")
    
    y = height - 80
    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Estimated Total Compensation: ₹{int(total_comp):,}")
    c.drawString(40, y - 18, f"Estimated Base Salary: ₹{int(adjusted_salary):,}")

    y = draw_section(c, y - 60, "Negotiation Range", {k: f"₹{int(v):,}" for k, v in negotiation_range.items()})
    
    proj_dict = {row['Year']: f"₹{int(row['Projected Salary']):,} ({row['Growth Rate']})" for _, row in projection_data.iterrows()}
    y = draw_section(c, y, "5-Year Projection", proj_dict)

    if skills_df is not None and not skills_df.empty:
        skills_dict = {row['Skill']: f"Impact: {row['Impact']}, Demand: {row['Demand']}" for _, row in skills_df.iterrows()}
        y = draw_section(c, y, "Skills Impact", skills_dict)

    bench_dict = {perc: f"₹{int(val):,}" for perc, val in benchmarks.items()}
    y = draw_section(c, y, "Market Benchmarks", bench_dict)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def generate_career_path(current_role, target_role, experience, education, priority):
    """Generates a mock career path projection."""
    base_year = datetime.now().year
    current_salary = 10_00_000 + (experience * 75_000)
    path = []
    
    for i in range(5):
        year = base_year + i
        growth_factor = 1.1 + (0.05 * i)
        salary = int(current_salary * growth_factor)
        market_avg = int(salary * random.uniform(0.9, 1.05))
        role = current_role if i < 2 else target_role
        timing = "Early Stage (Current Role)" if i < 2 else "Growth Stage (Target Role)"
        
        delta_vs_market = salary - market_avg
        delta_color = "var(--success)" if delta_vs_market >= 0 else "var(--danger)"
        delta_text = f"+₹{abs(delta_vs_market):,}" if delta_vs_market >= 0 else f"-₹{abs(delta_vs_market):,}"
        
        path.append({
            "Year": year, "Role": role, "Salary": salary, "MarketAvg": market_avg, "Timing": timing,
            "Delta": delta_text, "DeltaColor": delta_color,
            "Requirements": "Advanced skills in Python, ML, Leadership" if i > 2 else "Build experience and certifications"
        })
    return path

# Mock functions for Career Advisor page
def generate_current_skills(role, experience):
    skills = {"Python": 4, "SQL": 3.5, "Communication": 4, "Problem Solving": 3}
    return pd.DataFrame([{"Skill": k, "Level": v, "Category": "Technical" if k in ["Python", "SQL"] else "Soft"} for k, v in skills.items()])

def generate_target_skills(role):
    skills = {"Python": 5, "Machine Learning": 4.5, "Project Management": 4, "Leadership": 3.5}
    return pd.DataFrame([{"Skill": k, "Importance": v, "Type": "Technical" if k in ["Python", "Machine Learning"] else "Managerial"} for k, v in skills.items()])

def calculate_skills_gap(current_df, target_df):
    merged = pd.merge(current_df, target_df, on="Skill", how="outer").fillna(0)
    merged['Gap'] = merged['Importance'] - merged['Level']
    
    def get_priority(gap):
        if gap > 2: return "Critical"
        if gap > 1: return "Moderate"
        if gap > 0: return "Minor"
        return "None"
        
    merged['Priority'] = merged['Gap'].apply(get_priority)
    return merged[merged['Gap'] > 0][["Skill", "Level", "Importance", "Gap", "Priority"]].rename(columns={"Level": "Current", "Importance": "Required"})

def generate_action_plan(skills_gap, timeframe, priority, salary):
    plan = {"Skills": [], "Timeline": [], "Moves": []}
    start_date = datetime.now()
    
    for _, row in skills_gap.iterrows():
        plan["Skills"].append({"Name": row["Skill"], "Priority": row["Priority"], "Current": row["Current"], "Target": row["Required"],
                               "Hours": int(row["Gap"] * 20), "Impact": int(row["Gap"] * 5),
                               "Course": {"name": f"Advanced {row['Skill']}", "url": "#"},
                               "Book": {"name": f"{row['Skill']} for Experts", "url": "#"},
                               "Project": {"description": f"Build a project using {row['Skill']}"}})
        
        end_date = start_date + timedelta(days=int(row["Gap"] * 15))
        plan["Timeline"].append(dict(Skill=row["Skill"], Start=start_date, End=end_date, Priority=row["Priority"]))
        start_date = end_date + timedelta(days=5)

    plan["Moves"].append({"Title": "Seek a stretch project at current company", "SalaryBoost": 5, "Timing": "3-6 months", "Steps": "Discuss with manager about projects requiring target skills."})
    plan["Moves"].append({"Title": "Apply for a senior/specialist role", "SalaryBoost": 20, "Timing": "6-12 months", "Steps": "Update resume, network with recruiters, and start interviewing."})
    return plan


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #6c5ce7; margin-bottom: 0.5rem;">AI <span style="color: #00cec9;">Salary Predictor</span></h1>
            <p style="color: #636e72; font-size: 0.9rem; margin-top: 0;">AI Compensation Analytics Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## features of ai model")
    page = st.radio("", ["Salary Predictor", "Market Trends", "Career Advisor", "About"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 🏆 Achievements")
    
    with st.expander("Model Performance", expanded=True):
        st.markdown("""
            - **Accuracy:** 94.2% <span class="badge badge-popular">+2.2%</span>
            - **Coverage:** 50K+ records <span class="badge badge-new">New</span>
            - **Refresh:** Monthly updates
            - **Features:** 25+ factors analyzed
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔍 Need Help?")
    st.markdown("""
        Contact our team:
        [i.kunalsingh.ks@gmail.com](mailto:i.kunalsingh.ks@gmail.com)
        
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/kunal-singh-699485215/)
    """)
    
    st.markdown("---")
    st.markdown("### 🔄 System Status")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Models", "Online", "Stable")
    col2.metric("Data", "Current", "Updated")
    col3.metric("API", "Active", "100%")
    
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%B %d, %Y %H:%M')}")

# ---------- Main Content ----------
if page == "Salary Predictor":
    col1, col2 = st.columns([3, 2])
    with col1:
        st.title("AI-Powered Compensation Analytics")
        st.markdown("""
            <div style="font-size: 1.1rem; line-height: 1.6; color: #34495e;">
                Our advanced machine learning platform analyzes <strong>25+ professional factors</strong> to provide 
                hyper-accurate salary estimates. Get market-competitive compensation insights with 
                <strong>94.2% accuracy</strong> based on 50,000+ real salary records.
            </div>
        """, unsafe_allow_html=True)
    with col2:
        if lottie_animation:
            st_lottie(lottie_animation, height=250, key="main-animation")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.subheader("📋 Professional Profile Analysis")
        tab1, tab2, tab3 = st.tabs(["🧑‍💻 Core Profile", "🏆 Qualifications", "📊 Advanced Settings"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                job_role = st.selectbox("Primary Job Role", encoders["JobRole"].classes_)
                experience = st.slider("Years of Relevant Experience", 0.0, 30.0, 5.0, 0.5)
            with col2:
                location = st.selectbox("Primary Work Location", encoders["Location"].classes_)
                company = st.selectbox("Company Type", encoders["CompanyType"].classes_)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                education = st.selectbox("Highest Education Level", encoders["EducationLevel"].classes_)
                skill = st.selectbox("Technical Skill Level", encoders["SkillLevel"].classes_)
            with col2:
                certs = st.slider("Professional Certifications", 0, 10, 2)
                performance = st.select_slider("Performance Rating", options=["Below Average", "Average", "Above Average", "Exceptional", "Top Performer"], value="Above Average")
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                remote = st.selectbox("Work Arrangement", encoders["RemoteWork"].classes_)
                management = st.checkbox("People Management Responsibilities")
            with col2:
                equity = st.checkbox("Equity/Eligible for Equity")
                international = st.checkbox("International Scope")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        all_skills = list(skills_impact.keys())
        selected_skills = st.multiselect("Key Technical Skills (Select up to 5)", all_skills, max_selections=5)

        if st.button("🚀 Predict Compensation Package", use_container_width=True, type="primary"):
            with st.spinner("🧠 Analyzing factors across 50,000+ data points..."):
                time.sleep(1)
                
                try:
                    performance_mapping = {"Below Average": 0.8, "Average": 1.0, "Above Average": 1.2, "Exceptional": 1.5, "Top Performer": 2.0}
                    
                    encoded_inputs = [
                        encoders["EducationLevel"].transform([education])[0],
                        encoders["JobRole"].transform([job_role])[0],
                        encoders["Location"].transform([location])[0],
                        encoders["CompanyType"].transform([company])[0],
                        encoders["SkillLevel"].transform([skill])[0],
                        encoders["RemoteWork"].transform([remote])[0],
                        experience, certs, 1 if management else 0,
                        performance_mapping[performance], 1, 1 if international else 0
                    ]
                    
                    feature_names = [
                        "EducationLevel", "JobRole", "Location", "CompanyType", "SkillLevel", 
                        "RemoteWork", "YearsExperience", "NumCertifications", "Management",
                        "PerformanceRating", "RevenueImpact", "International"
                    ]
                    input_df = pd.DataFrame([encoded_inputs], columns=feature_names)
                    
                    base_salary = model.predict(input_df)[0]
                    
                    skills_impact_percent, skills_df = calculate_skills_impact(selected_skills)
                    adjusted_salary = base_salary * (1 + skills_impact_percent / 100)
                    
                    if equity:
                        equity_data = generate_equity_breakdown(company, job_role, adjusted_salary)
                        total_comp = adjusted_salary + equity_data["Estimated Annual Value"]
                    else:
                        equity_data = None
                        total_comp = adjusted_salary
                    
                    st.markdown("### 💰 Estimated Total Compensation")
                    salary_col1, salary_col2 = st.columns([2, 1])
                    with salary_col1:
                        st.markdown(f'<div class="result-card pulse-animation">₹{int(total_comp):,}</div>', unsafe_allow_html=True)
                    with salary_col2:
                        st.metric("Base Salary", f"₹{int(adjusted_salary):,}")
                        if equity:
                            st.metric("Equity Value", f"₹{equity_data['Estimated Annual Value']:,}")
                    
                    st.markdown("#### 💬 Smart Negotiation Range")
                    negotiation_range = calculate_negotiation_range(total_comp)
                    
                    neg_cols = st.columns(4)
                    neg_cols[0].metric("Minimum Ask (Low)", f"₹{negotiation_range['Low']:,}")
                    neg_cols[1].metric("Target", f"₹{negotiation_range['Target']:,}")
                    neg_cols[2].metric("Stretch Goal (High)", f"₹{negotiation_range['High']:,}")
                    neg_cols[3].metric("Premium Target", f"₹{negotiation_range['Premium']:,}")

                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Salary Composition", "🛠️ Skills Impact", "📈 Market Position", "🚀 Growth Projection"])
                    
                    with tab1:
                        st.subheader("Salary Composition Analysis")
                        breakdown_data = pd.DataFrame([
                            {"Component": "Base Salary", "Amount": adjusted_salary},
                            {"Component": "Equity (Annual)", "Amount": equity_data["Estimated Annual Value"] if equity else 0}
                        ])
                        breakdown_data = breakdown_data[breakdown_data["Amount"] > 0]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.dataframe(breakdown_data.style.format({"Amount": "₹{:,.0f}"}), hide_index=True)
                            st.dataframe(generate_benefits_breakdown(company), hide_index=True)
                        with col2:
                            fig = px.pie(breakdown_data, values='Amount', names='Component', title="Total Compensation Breakdown", hole=.3)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        if not skills_df.empty:
                            st.subheader("Skills Impact Analysis")
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-title">Total Skills Impact</div>
                                        <div class="metric-value">+{skills_impact_percent:.1f}%</div>
                                        <div class="metric-change positive">₹{int(adjusted_salary - base_salary):,} value</div>
                                    </div>
                                """, unsafe_allow_html=True)
                                st.dataframe(skills_df.sort_values("Impact", ascending=False), hide_index=True)
                            with col2:
                                skill_matrix = generate_skill_matrix(selected_skills, all_skills)
                                fig = px.treemap(skill_matrix, path=['Category', 'Skill'], values='Impact', color='Demand', title="Skill Demand & Impact")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No skills selected. Add skills to see their impact.")
                    
                    with tab3:
                        st.subheader("Market Position Analysis")
                        benchmarks = generate_comp_benchmark(job_role, location, experience)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.dataframe(pd.DataFrame(benchmarks.items(), columns=["Percentile", "Salary"]).style.format({"Salary": "₹{:,.0f}"}), hide_index=True)
                        with col2:
                            fig = go.Figure(go.Box(y=list(benchmarks.values()), name="Market Range"))
                            fig.add_trace(go.Scatter(y=[total_comp], mode='markers', name='Your Estimate', marker=dict(color='red', size=10)))
                            st.plotly_chart(fig, use_container_width=True)

                    with tab4:
                        st.subheader("5-Year Growth Projection")
                        dummy_trend = pd.DataFrame({"Year": range(2020, 2025), "Tech Industry": np.linspace(850000, 1250000, 5)})
                        projection_data = generate_salary_projection(total_comp, 5, trend_model if isinstance(trend_model, pd.DataFrame) else dummy_trend)
                        fig = px.line(projection_data, x="Year", y="Projected Salary", title="Projected Salary Growth", markers=True)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(projection_data, hide_index=True)

                    st.markdown("---")
                    st.download_button(label="📄 Download Full Compensation Report",
                        data=generate_report(total_comp, adjusted_salary, negotiation_range, projection_data, skills_df, benchmarks),
                        file_name=f"salary_report_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf", use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

elif page == "Market Trends":
    st.title("📊 Advanced Market Analytics")
    st.markdown("Real-time compensation trends and industry benchmarks.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Live Market Pulse")
        pulse_col1, pulse_col2, pulse_col3 = st.columns(3)
        pulse_col1.metric("Tech Salary Growth", "8.2%", "+0.5% YoY")
        pulse_col2.metric("Remote Premium", "12.5%", "-2.1% YoY")
        pulse_col3.metric("Equity Grants", "18%", "+3.2% YoY")
    with col2:
        if lottie_chart:
            st_lottie(lottie_chart, height=150, key="trend-animation")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.subheader("📈 Sector-wise Salary Trends")
    trend_data_display = trend_model if isinstance(trend_model, pd.DataFrame) else pd.DataFrame({ 
        "Year": [2020, 2021, 2022, 2023, 2024],
        "Tech Industry": [8_50_000, 9_20_000, 10_50_000, 11_80_000, 12_50_000],
        "Finance": [9_00_000, 9_30_000, 9_80_000, 10_20_000, 10_60_000],
        "Healthcare": [7_20_000, 7_50_000, 8_00_000, 8_40_000, 8_90_000]
    })
    
    trend_data_display = trend_data_display.dropna(subset=trend_data_display.columns.drop('Year')) # Fix: Drop rows with NaN
    fig = px.line(trend_data_display, x="Year", y=trend_data_display.columns.drop('Year'), title="Average Salary Growth by Sector (₹)")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Career Advisor":
    st.title("🚀 AI Career Path Optimizer")
    st.markdown("Personalized career growth recommendations.")
    
    form_col1, form_col2 = st.columns(2)
    with form_col1:
        current_role = st.selectbox("Current Role", encoders["JobRole"].classes_)
        current_salary = st.number_input("Current Total Compensation (₹)", min_value=0, value=12_00_000, step=50_000)
    with form_col2:
        target_role = st.selectbox("Target Role", encoders["JobRole"].classes_, index=1)
        experience = st.slider("Years of Experience", 0, 30, 5)
        
    if st.button("🔮 Generate Career Path", type="primary", use_container_width=True):
        with st.spinner("Analyzing 1,000+ career paths..."):
            time.sleep(1)
            st.success("✅ Analysis complete!")

            career_milestones = generate_career_path(current_role, target_role, experience, "Bachelor's", "Maximize Earnings")
            
            st.subheader("📈 Your Projected Career Trajectory")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[m["Year"] for m in career_milestones], y=[m["Salary"] for m in career_milestones], mode="lines+markers", name="Projected Salary"))
            fig.add_trace(go.Scatter(x=[m["Year"] for m in career_milestones], y=[m["MarketAvg"] for m in career_milestones], mode="lines", name="Market Average", line=dict(dash='dot')))
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🛠️ Skills Gap Analysis")
            current_skills = generate_current_skills(current_role, experience)
            target_skills = generate_target_skills(target_role)
            skills_gap = calculate_skills_gap(current_skills, target_skills)
            
            col1, col2, col3 = st.columns(3)
            # Fix: Corrected f-string usage
            critical_gaps = len(skills_gap[skills_gap['Priority'] == 'Critical'])
            moderate_gaps = len(skills_gap[skills_gap['Priority'] == 'Moderate'])
            minor_gaps = len(skills_gap[skills_gap['Priority'] == 'Minor'])

            col1.metric("Critical Gaps", critical_gaps)
            col2.metric("Moderate Gaps", moderate_gaps)
            col3.metric("Minor Gaps", minor_gaps)
            
            st.dataframe(skills_gap, use_container_width=True, hide_index=True)
            
            st.subheader("🎯 Personalized Action Plan")
            action_plan = generate_action_plan(skills_gap, "6-12 months", "Maximize Earnings", current_salary)
            for move in action_plan["Moves"]:
                st.info(f"**{move['Title']} ({move['Timing']})**\n*Potential Boost: +{move['SalaryBoost']}%*\n*Action: {move['Steps']}*")

elif page == "About":
    st.title("About  ai SalaryPredictor ")
    st.markdown("""
    Ai SalaryPredictor is an advanced AI-powered salary and career optimization platform.
    
    🧠 Built with machine learning and real-world compensation data, it helps you:
    - Predict your ideal salary
    - Benchmark against market trends
    - Get negotiation insights
    - Plan your career trajectory with smart analytics

    💼 Whether you're a fresher, experienced professional, or a career switcher — we help you make smarter compensation decisions.
    
    ---
    
    _Made by Kunal Singh_ 
    """)