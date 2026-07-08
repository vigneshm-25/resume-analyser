import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from dotenv import load_dotenv
from groq import Groq


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and get AI-powered insights ")


job_skills = {
    "Data Scientist": [
        "python", "machine learning", "deep learning", "nlp",
        "pandas", "numpy", "matplotlib", "scikit-learn"
    ],

    "Data Analyst": [
        "sql", "excel", "python", "tableau",
        "power bi", "data visualization", "statistics"
    ],

    "Machine Learning Engineer": [
        "python", "machine learning", "scikit-learn",
        "tensorflow", "pytorch", "model deployment", "mlops"
    ],

    "Software Engineer": [
        "java", "python", "c++", "data structures",
        "algorithms", "oop", "system design"
    ],

    "Backend Developer": [
        "python", "django", "flask", "node.js",
        "apis", "sql", "mongodb"
    ],

    "Frontend Developer": [
        "html", "css", "javascript", "react",
        "angular", "ui/ux"
    ],

    "DevOps Engineer": [
        "docker", "kubernetes", "aws", "azure",
        "ci/cd", "linux", "terraform"
    ],

    "AI Engineer": [
        "python", "deep learning", "nlp",
        "transformers", "huggingface", "llms"
    ],

    "Business Analyst": [
        "excel", "sql", "power bi", "tableau",
        "communication", "requirements gathering"
    ],

    "Cloud Engineer": [
        "aws", "azure", "gcp", "docker",
        "kubernetes", "networking"
    ]
}


skill_map = {
    "python": ["python"],
    
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "natural language processing": ["nlp", "natural language processing"],
    
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "matplotlib": ["matplotlib"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    
    "sql": ["sql"],
    "excel": ["excel", "ms excel"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "data visualization": ["data visualization", "data viz"],
    "statistics": ["statistics", "stats"],
    
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "model deployment": ["model deployment", "deployment"],
    "mlops": ["mlops", "ml ops"],
    
    "java": ["java"],
    "c++": ["c++"],
    "data structures": ["data structures", "dsa"],
    "algorithms": ["algorithms", "algo"],
    "oop": ["oop", "object oriented programming"],
    "system design": ["system design"],
    
    "django": ["django"],
    "flask": ["flask"],
    "node.js": ["node.js", "nodejs"],
    "apis": ["api", "apis", "rest api"],
    "mongodb": ["mongodb", "mongo"],
    
    "html": ["html"],
    "css": ["css"],
    "javascript": ["javascript", "js"],
    "react": ["react"],
    "angular": ["angular"],
    "ui/ux": ["ui", "ux", "ui/ux"],
    
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure"],
    "gcp": ["gcp", "google cloud"],
    "ci/cd": ["ci/cd", "ci cd"],
    "linux": ["linux"],
    "terraform": ["terraform"],
    
    "transformers": ["transformers"],
    "huggingface": ["huggingface", "hugging face"],
    "llms": ["llm", "llms", "large language models"],
    
    "communication": ["communication", "communication skills"],
    "requirements gathering": ["requirements gathering", "requirement analysis"],
    "networking": ["networking"]
}




def extract_text(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t
    return text


def get_skills(text):
    text = text.lower()
    skills = []
    for main, variations in skill_map.items():
        for v in variations:
            if v in text:
                skills.append(main)
                break
    return skills


def extract_experience(text):
    text = text.lower()
    found = re.findall(r'(\d+)\+?\s*(years|year|months|month)', text)
    exp = 0
    for num, dur in found:
        num = int(num)
        if "year" in dur:
            exp += num
        else:
            exp += num / 12
    return round(exp, 1)


def count_projects(text):
    matches = re.findall(r'\bprojects?\b', text.lower())
    return max(len(matches) - 1, 0)

def suggest_roles(skills):
    suggested = []

    for role, req_skills in job_skills.items():
        match_count = 0
        for s in skills:
            if s in req_skills:
                match_count += 1
        
        if match_count >= 2:
            suggested.append(role)

    return suggested

def calculate_score(skills, role, exp, projects):
    tf_skills = " ".join(skills)
    tf_role = " ".join(job_skills[role])

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([tf_skills, tf_role])
    sim = cosine_similarity(vectors[0], vectors[1])[0][0] * 100

    proj_score = min(projects / 5, 1) * 100
    exp_score = min(exp / 5, 1) * 100

    final = round(sim * 0.6 + exp_score * 0.2 + proj_score * 0.2, 2)
    return final


def generate_jd(role):
    return f"{role} with skills in {', '.join(job_skills[role])}"


def get_ai_suggestions(resume, job_desc):
    prompt = f"""
    Analyze this resume and suggest improvements.

    Resume:
    {resume}

    Job Description:
    {job_desc}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content




uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])





if uploaded_file:

    text = extract_text(uploaded_file)
    skills = get_skills(text)
    exp = extract_experience(text)
    projects = count_projects(text)

    st.success("✅ Resume uploaded successfully!")

    # 🎯 Filter roles based on skills
    filtered_roles = suggest_roles(skills)

    if len(filtered_roles) == 0:
        st.warning("⚠️ No strong role match found. Showing all roles.")
        filtered_roles = list(job_skills.keys())

   
    st.subheader("💼 Suggested Roles")
    st.write(filtered_roles)

    role = st.selectbox("🎯 Select Job Role", filtered_roles)

    if st.button("🚀 Analyze Resume"):

        score = calculate_score(skills, role, exp, projects)
        jd = generate_jd(role)


        st.subheader("📊 Match Score")
        st.progress(int(score))
        st.write(f"### {score} % match")


        st.subheader("🧠 Extracted Skills")
        st.write(skills)

   
        st.subheader("📅 Experience")
        st.write(f"{exp} years")

        # 📁 Projects
        st.subheader("📁 Projects")
        st.write(projects)

        with st.spinner("🤖 Generating AI suggestions..."):
            suggestions = get_ai_suggestions(text, jd)

        st.subheader("💡 AI Suggestions")
        st.write(suggestions)