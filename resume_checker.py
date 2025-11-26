import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import re
from docx import Document
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import os

# ---------------------------------
# Utility: File Text Extractors
# ---------------------------------
def extract_text_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        print("PDF read error:", e)
    return text

def extract_text_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + " "
    except Exception as e:
        print("DOCX read error:", e)
    return text

# ----------------------------
# Preprocessing & Helpers
# ----------------------------
def preprocess(text):
    if text is None:
        return ""
    text = text.lower()
    # keep alphanum and spaces and + (e.g., c++)
    text = re.sub(r'[^a-z0-9\+\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Basic skill bank (you can expand this)
SKILL_BANK = [
    "python", "sql", "excel", "power bi", "tableau", "pandas", "numpy",
    "machine learning", "deep learning", "scikit-learn", "statistics",
    "data analysis", "data visualization", "r", "sas", "matlab",
    "nlp", "computer vision", "aws", "azure", "docker", "kubernetes",
    "communication", "presentation", "sql server", "postgresql",
    "bigquery", "hadoop", "spark", "etl", "powerpoint", "problem solving",
    "java", "c++", "c", "git", "github", "html", "css", "javascript",
    "keras", "tensorflow", "pytorch"
]

# Expandable roles database (role: descriptive skills string)
ROLES_DB = {
    "Data Analyst": "python sql excel tableau power bi data visualization pandas statistics data cleaning reporting",
    "Data Scientist": "python machine learning statistics deep learning pandas numpy scikit-learn modeling data preprocessing",
    "Business Analyst": "excel powerpoint communication requirements analysis stakeholder management data analysis",
    "ML Engineer": "python machine learning deep learning tensorflow pytorch docker aws model deployment",
    "Data Engineer": "sql hadoop spark etl airflow aws bigquery data pipeline",
    "BI Developer": "power bi tableau sql data visualization dax powerquery reporting",

}

# ----------------------------
# Skill Extraction
# ----------------------------
def extract_skills(text):
    text = preprocess(text)
    found = set()
    for skill in SKILL_BANK:
        # word boundary check
        pat = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pat, text):
            found.add(skill.lower())
    return sorted(found)

# ----------------------------
# Keyword Matching (weighted)
# ----------------------------
def keyword_match(job_desc, resume_text, important_keywords=None):
    if important_keywords is None:
        important_keywords = []
    jd_words = set(re.findall(r'\b\w+\b', job_desc.lower()))
    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    
    matches = jd_words.intersection(resume_words)
    # compute weighted score for important keywords
    weighted_score = 0
    for word in important_keywords:
        if word.lower() in resume_words:
            weighted_score += 2  # weight = 2 for important keywords

    # normal score: number of matched non-important jd words
    normal_matches = [w for w in matches if w.lower() not in [ik.lower() for ik in important_keywords]]
    normal_score = len(normal_matches)

    total_score = weighted_score + normal_score
    # define max possible score as (unique jd words) + (count of important keywords)
    max_score = len(jd_words) + len(important_keywords)
    if max_score == 0:
        match_percentage = 0.0
    else:
        match_percentage = (total_score / max_score) * 100

    missing_keywords = jd_words - resume_words
    return match_percentage, sorted(matches), sorted(missing_keywords)

# ----------------------------
# Role Recommendation (TF-IDF + Cosine)
# ----------------------------
def recommend_roles(resume_text, roles_db=ROLES_DB, top_n=1):
    # Build docs: roles descriptions + resume
    role_names = list(roles_db.keys())
    docs = [roles_db[r] for r in role_names]
    docs.append(resume_text)
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform(docs)
        resume_vec = vectors[-1]
        role_vecs = vectors[:-1]
        sims = cosine_similarity(resume_vec, role_vecs).flatten()
        # get top_n roles with highest similarity
        top_indices = sims.argsort()[::-1][:top_n]
        recommendations = [(role_names[i], float(sims[i])) for i in top_indices]
        return recommendations
    except Exception as e:
        print("Recommendation error:", e)
        return []

# ----------------------------
# Suggestion Generator
# ----------------------------
def generate_suggestions(missing_keywords, extracted_skills, match_percentage, recommended_roles):
    suggestions = []
    if match_percentage > 80:
        suggestions.append("Excellent match. Highlight relevant projects and achievements.")
    elif match_percentage > 50:
        suggestions.append("Good match. Add specific project bullets for missing skills.")
    else:
        suggestions.append("Low match. Add or emphasize skills/experience listed in job description.")
    
    if missing_keywords:
        # suggest up to 8 high-value missing items
        sample_missing = list(missing_keywords)[:8]
        suggestions.append("Missing keywords to consider adding: " + ", ".join(sample_missing))
        # give example resume bullets for common skills
        for kw in sample_missing:
            kw_clean = kw.lower()
            if kw_clean in ["python", "sql", "excel", "power bi", "tableau", "machine", "machine learning", "statistics"]:
                suggestions.append(f'Add a measurable bullet: "Used {kw} to analyze X records and improved Y by Z%."')
            else:
                suggestions.append(f'Add experience/ project mentioning "{kw}". Example: "Worked on {kw} for ...".')
    else:
        suggestions.append("No missing keywords detected relative to the JD text.")

    # suggest role-based improvements
    if recommended_roles:
        role, score = recommended_roles[0]
        suggestions.append(f"Top recommended role: {role} (similarity: {score:.2f})")
        if role == "Data Analyst":
            suggestions.append("Emphasize data cleaning, dashboarding, SQL queries, and Excel/Power BI projects.")
        elif role == "Data Scientist":
            suggestions.append("Emphasize modeling, ML algorithms, evaluation metrics, and coding reproducibility.")
        elif role == "Business Analyst":
            suggestions.append("Emphasize stakeholder communication, requirements gathering, and reporting.")
    return suggestions

# ----------------------------
# GUI & Interaction
# ----------------------------
class ResumeAnalyzerApp:
    def __init__(self, root):
        self.root = root
        root.title("AI-Powered Resume Analyzer & Job Recommender")
        root.geometry("760x640")

        # Left frame: file selection & JD input
        left = tk.Frame(root, padx=10, pady=10)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(left, text="Resume File (PDF / DOCX):").pack(anchor="w")
        file_frame = tk.Frame(left)
        file_frame.pack(fill=tk.X, pady=(0,8))
        self.file_entry = tk.Entry(file_frame, width=55)
        self.file_entry.pack(side=tk.LEFT, padx=(0,6))
        tk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT)

        tk.Label(left, text="(Optional) Paste Job Description here:").pack(anchor="w")
        self.jd_text = scrolledtext.ScrolledText(left, height=12, wrap=tk.WORD)
        self.jd_text.pack(fill=tk.BOTH, expand=False, pady=(4,8))

        # Example JD buttons
        ex_frame = tk.Frame(left)
        ex_frame.pack(fill=tk.X, pady=(0,12))
        tk.Label(ex_frame, text="Example JDs:").pack(side=tk.LEFT)
        tk.Button(ex_frame, text="Data Analyst JD", command=self.fill_example_analyst).pack(side=tk.LEFT, padx=4)
        tk.Button(ex_frame, text="Data Scientist JD", command=self.fill_example_scientist).pack(side=tk.LEFT, padx=4)
        tk.Button(ex_frame, text="Clear JD", command=lambda: self.jd_text.delete('1.0', tk.END)).pack(side=tk.LEFT, padx=4)

        # Important keywords (comma separated)
        tk.Label(left, text="Important Keywords (comma-separated, optional):").pack(anchor="w")
        self.important_entry = tk.Entry(left, width=60)
        self.important_entry.pack(pady=(2,8))

        # Buttons
        btn_frame = tk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=(6,12))
        tk.Button(btn_frame, text="Analyze Resume", command=self.check_resume, width=18).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Save Last Report", command=self.save_report, width=18).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Exit", command=root.quit, width=10).pack(side=tk.RIGHT, padx=6)

        # Right frame: output
        right = tk.Frame(root, padx=10, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        tk.Label(right, text="Analysis Report:").pack(anchor="w")
        self.report_box = scrolledtext.ScrolledText(right, height=35, wrap=tk.WORD)
        self.report_box.pack(fill=tk.BOTH, expand=True)

        # state
        self.last_report_text = ""
        self.last_analysis = None

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("Word Documents", "*.docx")])
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def fill_example_analyst(self):
        jd = """We are seeking a Data Analyst with strong skills in Python, SQL, Excel, Power BI, and Tableau.
Responsibilities include data cleaning, building dashboards, performing statistical analysis, and creating reports.
Candidates should be proficient in Pandas, SQL queries, data visualization and have experience working with large datasets."""
        self.jd_text.delete('1.0', tk.END)
        self.jd_text.insert(tk.END, jd)

    def fill_example_scientist(self):
        jd = """Looking for a Data Scientist with strong experience in Python, machine learning, statistics, scikit-learn, and deep learning frameworks.
Must be able to build models, evaluate them, work with large datasets, and deploy models in production. Knowledge of AWS or Docker is a plus."""
        self.jd_text.delete('1.0', tk.END)
        self.jd_text.insert(tk.END, jd)

    def save_report(self):
        if not self.last_report_text:
            messagebox.showinfo("Save Report", "No report available to save. Run analysis first.")
            return
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"resume_report_{now}.txt"
        save_path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=default_name, filetypes=[("Text files","*.txt")])
        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(self.last_report_text)
                messagebox.showinfo("Saved", f"Report saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {e}")

    def check_resume(self):
        resume_file = self.file_entry.get().strip()
        if not resume_file:
            messagebox.showwarning("Error", "Please select a resume file (PDF or DOCX).")
            return
        if not os.path.exists(resume_file):
            messagebox.showerror("Error", "Selected file does not exist.")
            return

        # Job description
        job_description = self.jd_text.get('1.0', tk.END).strip()
        if not job_description:
            # fallback generic JD
            job_description = "Data analyst role requiring python, sql, excel, power bi, statistics, and data visualization."

        important_raw = self.important_entry.get().strip()
        important_keywords = [k.strip() for k in important_raw.split(",") if k.strip()] if important_raw else []

        # Extract resume text
        resume_text = ""
        try:
            if resume_file.lower().endswith(".pdf"):
                resume_text = extract_text_pdf(resume_file)
            elif resume_file.lower().endswith(".docx"):
                resume_text = extract_text_docx(resume_file)
            else:
                messagebox.showerror("Error", "Unsupported file format. Use PDF or DOCX.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file: {e}")
            return

        resume_text_clean = preprocess(resume_text)
        jd_clean = preprocess(job_description)

        # Extract skills and matches
        extracted_skills = extract_skills(resume_text)
        match_percentage, matches, missing_keywords = keyword_match(jd_clean, resume_text_clean, important_keywords)

        # Role recommendation
        recommendations = recommend_roles(resume_text_clean, ROLES_DB, top_n=2)

        # Compose report
        report_lines = []
        report_lines.append("AI-Powered Resume Analyzer Report")
        report_lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("------------------------------------------------------------")
        report_lines.append(f"Resume file: {resume_file}")
        report_lines.append("")
        report_lines.append(f"Job Description (first 300 chars): {job_description[:300]}{'...' if len(job_description)>300 else ''}")
        report_lines.append("")
        report_lines.append(f"Important keywords (user-specified): {', '.join(important_keywords) if important_keywords else 'None'}")
        report_lines.append("")
        report_lines.append(f"Match Score: {match_percentage:.2f}%")
        report_lines.append(f"Matched Keywords (sample): {', '.join(matches[:30]) if matches else 'None'}")
        report_lines.append(f"Missing Keywords (sample): {', '.join(list(missing_keywords)[:30]) if missing_keywords else 'None'}")
        report_lines.append("")
        report_lines.append(f"Extracted Skills from Resume: {', '.join(extracted_skills) if extracted_skills else 'None detected'}")
        report_lines.append("")
        report_lines.append("Top Role Recommendations:")
        if recommendations:
            for role, score in recommendations:
                report_lines.append(f" - {role} (similarity: {score:.2f})")
        else:
            report_lines.append("No recommendations available.")
        report_lines.append("")
        suggestions = generate_suggestions(missing_keywords, extracted_skills, match_percentage, recommendations)
        report_lines.append("Suggestions / Actionable Edits:")
        for s in suggestions:
            report_lines.append(" * " + s)

        # Additional quick tips
        report_lines.append("")
        report_lines.append("Quick Resume Tips:")
        report_lines.append(" - Use quantified achievements (e.g., 'improved accuracy by 12%').")
        report_lines.append(" - Add a dedicated 'Skills' section with tools/technologies.")
        report_lines.append(" - Add 2-3 project bullets that show impact and tools used.")
        report_lines.append("")
        report_text = "\n".join(report_lines)

        # show in GUI
        self.report_box.delete('1.0', tk.END)
        self.report_box.insert(tk.END, report_text)
        self.last_report_text = report_text
        self.last_analysis = {
            "match_percentage": match_percentage,
            "matches": matches,
            "missing_keywords": missing_keywords,
            "extracted_skills": extracted_skills,
            "recommendations": recommendations,
            "suggestions": suggestions
        }
        
        # Optionally pop a message if score low/high
        if match_percentage >= 75:
            messagebox.showinfo("Analysis Complete", f"Good match: {match_percentage:.2f}% — see report at right.")
        elif match_percentage >= 45:
            messagebox.showinfo("Analysis Complete", f"Average match: {match_percentage:.2f}% — see suggestions to improve.")
        else:
            messagebox.showinfo("Analysis Complete", f"Low match: {match_percentage:.2f}% — consider editing resume and re-running analysis.")

# ----------------------------
# Run the app
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeAnalyzerApp(root)
    root.mainloop()
