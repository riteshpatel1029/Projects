"""
AI-Powered Resume Screening System
Author: Ritesh Patel
Tech Stack: Python · SpaCy · TF-IDF · Cosine Similarity · Flask
"""

import re
import json
import spacy
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model (run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# ─────────────────────────────────────────────
# 1. Text Cleaning
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# 2. SpaCy NER – Extract Skills / Experience / Qualifications
# ─────────────────────────────────────────────
SKILL_KEYWORDS = {
    "python", "java", "javascript", "sql", "html", "css", "react", "node",
    "machine learning", "deep learning", "nlp", "docker", "kubernetes",
    "aws", "azure", "gcp", "tensorflow", "pytorch", "scikit-learn", "flask",
    "django", "git", "linux", "pandas", "numpy", "spark", "hadoop",
}

DEGREE_PATTERNS = [
    r"\b(b\.?e|b\.?tech|bachelor|b\.?sc|m\.?tech|m\.?sc|mba|phd|diploma)\b"
]


def extract_entities(text: str) -> dict:
    """Extract skills, experience years, qualifications from resume text."""
    doc = nlp(text)
    text_lower = text.lower()

    # Skills
    found_skills = [skill for skill in SKILL_KEYWORDS if skill in text_lower]

    # Experience (look for patterns like "3 years", "2+ years")
    exp_matches = re.findall(r"(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience", text_lower)
    years_exp = max([int(x) for x in exp_matches], default=0)

    # Qualifications / degrees
    found_degrees = []
    for pattern in DEGREE_PATTERNS:
        found_degrees += re.findall(pattern, text_lower)

    # Named entities (ORG = companies, GPE = locations)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

    return {
        "skills": list(set(found_skills)),
        "years_experience": years_exp,
        "qualifications": list(set(found_degrees)),
        "organizations": orgs[:5],
        "locations": locations[:3],
    }


# ─────────────────────────────────────────────
# 3. TF-IDF + Cosine Similarity Ranking
# ─────────────────────────────────────────────
class ResumeScreener:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10_000,
        )

    def rank_resumes(self, job_description: str, resumes: list[dict]) -> list[dict]:
        """
        Rank resumes against a job description.

        Args:
            job_description: Raw JD text.
            resumes: List of dicts with keys 'id', 'name', 'text'.

        Returns:
            Sorted list with similarity scores and extracted entities.
        """
        jd_clean = clean_text(job_description)
        resume_texts = [clean_text(r["text"]) for r in resumes]

        # Fit TF-IDF on JD + all resumes together
        corpus = [jd_clean] + resume_texts
        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        jd_vector = tfidf_matrix[0]
        resume_vectors = tfidf_matrix[1:]

        scores = cosine_similarity(jd_vector, resume_vectors)[0]

        results = []
        for i, resume in enumerate(resumes):
            entities = extract_entities(resume["text"])
            results.append({
                "rank": 0,           # filled after sort
                "id": resume["id"],
                "name": resume["name"],
                "similarity_score": round(float(scores[i]), 4),
                "shortlisted": scores[i] >= 0.25,
                **entities,
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results


# ─────────────────────────────────────────────
# 4. Flask REST API
# ─────────────────────────────────────────────
app = Flask(__name__)
screener = ResumeScreener()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Resume Screening API"})


@app.route("/rank", methods=["POST"])
def rank():
    """
    POST /rank
    Body (JSON):
    {
      "job_description": "We are looking for a Python ML engineer...",
      "resumes": [
        {"id": "r1", "name": "Alice", "text": "..."},
        {"id": "r2", "name": "Bob",   "text": "..."}
      ]
    }
    """
    data = request.get_json(force=True)
    if not data or "job_description" not in data or "resumes" not in data:
        return jsonify({"error": "Provide 'job_description' and 'resumes'"}), 400

    ranked = screener.rank_resumes(data["job_description"], data["resumes"])
    shortlisted = [r for r in ranked if r["shortlisted"]]

    return jsonify({
        "total_resumes": len(ranked),
        "shortlisted_count": len(shortlisted),
        "ranked_resumes": ranked,
    })


@app.route("/extract", methods=["POST"])
def extract():
    """Extract entities from a single resume."""
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Provide 'text'"}), 400
    return jsonify(extract_entities(data["text"]))


# ─────────────────────────────────────────────
# 5. CLI Demo
# ─────────────────────────────────────────────
SAMPLE_JD = """
We are looking for a Machine Learning Engineer with strong Python skills.
Experience with scikit-learn, TensorFlow, and NLP is required.
Candidates should have at least 2 years of experience and a B.Tech or B.E. degree.
Flask or Django for API development is a plus.
"""

SAMPLE_RESUMES = [
    {
        "id": "r1", "name": "Alice",
        "text": "Python developer with 3 years experience. Skilled in scikit-learn, "
                "TensorFlow, NLP, Flask. B.Tech in CSE from IIT Bombay.",
    },
    {
        "id": "r2", "name": "Bob",
        "text": "Java backend developer. 4 years experience with Spring Boot and SQL. "
                "Good at Docker and Kubernetes. B.E. in IT.",
    },
    {
        "id": "r3", "name": "Carol",
        "text": "Data scientist with Python, pandas, numpy, machine learning. "
                "2 years experience. M.Sc. Statistics. Published NLP research.",
    },
]


def demo():
    print("=== Resume Screening CLI Demo ===\n")
    results = screener.rank_resumes(SAMPLE_JD, SAMPLE_RESUMES)
    for r in results:
        tag = "✅ SHORTLISTED" if r["shortlisted"] else "❌ Rejected"
        print(f"#{r['rank']}  {r['name']:<10}  Score: {r['similarity_score']:.4f}  "
              f"Skills: {r['skills']}  {tag}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        print("Starting Flask API on http://localhost:5000 ...")
        app.run(debug=True, port=5000)
    else:
        demo()
