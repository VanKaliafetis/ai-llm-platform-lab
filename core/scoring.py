import re
from difflib import SequenceMatcher


STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "for",
    "and",
    "or",
    "in",
    "on",
    "with",
    "using",
    "use",
    "this",
    "that",
    "it",
    "as",
    "by",
    "be",
    "from",
    "at",
}


def normalize(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def terms(s):
    words = re.findall(r"[a-zA-Z0-9_:-]+", normalize(s))
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def answer_similarity(expected, actual):
    e, a = normalize(expected), normalize(actual)

    if not e:
        return None

    ratio = SequenceMatcher(None, e, a).ratio()
    e_terms = terms(e)
    a_terms = terms(a)

    overlap = len(e_terms & a_terms) / max(1, len(e_terms))

    return round((ratio * 0.35 + overlap * 0.65) * 100, 2)


def groundedness(answer, context):
    a_terms = terms(answer)
    c_terms = terms(context)

    if not a_terms:
        return 0.0

    return round(100 * len(a_terms & c_terms) / len(a_terms), 2)


def retrieval_score(question, chunks):
    q_terms = terms(question)
    c_terms = terms(" ".join(chunks))

    if not q_terms:
        return 0.0

    return round(100 * len(q_terms & c_terms) / len(q_terms), 2)


def score_label(score):
    if score is None:
        return "N/A"

    score = float(score)

    if score >= 75:
        return "Strong"
    if score >= 45:
        return "Partial"
    return "Weak"