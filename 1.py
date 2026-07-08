import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import groq

#Roll list

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

def get_skills_for_role(role,job_skills):
    role = role.lower()

    return job_skills.get(role, [])

#Skill list
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


def extraction_from_pdf():
    text=""

    with pdfplumber.open(r"sample_resume.pdf") as pdf:
        for page in pdf.pages:
            page_text=page.extract_text()
            if page_text:
                text+=page_text
    return text

#Filtering Skills

def get_skills(skill_map,text):
    text = text.lower()
    skills = []
    for main_skill,others in skill_map.items():
        for var in others:
            if var in text:
                skills.append(main_skill)
                break
    return skills


def suggest_roles(skills,job_skills):
    role_suggestions_list = []

    for role,need in job_skills.items():
        count=0
        for skill in skills:
            if skill in need:
                count+=1
        if count>=2:
            role_suggestions_list.append(role)

    return role_suggestions_list


#Getting experience from resume

def extract_experience(pdf_text):

    text = pdf_text.lower()

    found = re.findall(r'(\d+)\+?\s*(years|year|months|month)', text)
    if found:
        experience = 0

        for num,dur in found:
            num = int(num)
            if "year" in found or "years" in found:
                experience+=num
            else:
                experience+=num/12
        return max(experience)
    else:
        return 0
    
def count_projects(text):
    text = text.lower()
    
    matches = re.findall(r'\bprojects?\b', text)
    return len(matches) - 1 if len(matches) > 0 else 0

#predicting relation 
def prediction(tf_skills,tf_role_skills,exp,proj_count):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([tf_skills,tf_role_skills])
    similarity = cosine_similarity(vectors[0],vectors[1])
    sim_score = round(similarity[0][0]*100)
    proj_score = min(proj_count/5,1)*100
    exp_score = min(exp/5,1)*100
                      
    print("Match score: ",sim_score*0.6 + exp_score*0.2 + proj_score*0.2,"%")


#generatring job description
def generate_job_description(role, job_skills):
    skills = job_skills.get(role, [])
    
    jd = f"We are looking for a {role} with skills in "
    jd += ", ".join(skills)
    
    return jd

#extracting text from PDF
pdf_text = extraction_from_pdf()

#getting skill set
skills = get_skills(skill_map,pdf_text) 

#suggesting roles
suggestion = suggest_roles(skills,job_skills)

print(skills)
print("Here are some roles that matches your skills\n")
for i in suggestion:
    print(i )

#Getting experience
exp = extract_experience(pdf_text)
print(f"Experience:{exp} years")

#count projects
proj_count = count_projects(pdf_text)
print(f"No. of projects:{proj_count}")

role_sel = input("Enter which role do you prefer:")
tf_skills = " ".join(skills)
tf_role_skills = " ".join(job_skills[role_sel])

pred = prediction(tf_skills,tf_role_skills,exp,proj_count)

job_desc = generate_job_description(role_sel, job_skills)

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))



def get_ai_suggestions(resume, job_desc):
    prompt = f"""
    Analyze this resume and job description.
    Give improvements and missing skills.

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

suggest = get_ai_suggestions(pdf_text, job_desc)
print(suggest)