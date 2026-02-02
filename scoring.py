import re


STOPWORDS = {
    "the",
    "and",
    "or",
    "to",
    "for",
    "with",
    "a",
    "an",
    "of",
    "in",
    "on",
    "by",
    "is",
    "are",
    "from",
    "that",
    "this",
    "we",
    "our",
    "their",
    "they",
    "as",
    "at",
    "be",
    "it",
    "into",
    "across",
    "within",
    "around",
    "over",
    "need",
    "looking",
    "seeking",
}


def _tokenize(text):
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _flatten_credentials(expert):
    cred = expert.get("credentials", {})
    parts = []
    for key in ["education", "formerCompanies", "notableProjects", "publications"]:
        val = cred.get(key, [])
        if isinstance(val, list):
            parts.extend(val)
        elif isinstance(val, str):
            parts.append(val)
    return " ".join(parts).lower()


def score_expert(criteria, expert):
    industries = set(criteria.get("industries") or [])
    functions = set(criteria.get("functions") or [])
    levels = set(criteria.get("levels") or [])
    budget = criteria.get("budget") or 500
    free_text = criteria.get("free_text") or ""
    profile_text = criteria.get("profile_text") or ""

    score = 0
    reasons = []

    if industries:
        overlap = industries.intersection(expert.get("industryTags", []))
        industry_score = min(25, 25 * (len(overlap) / max(1, len(industries))))
        score += industry_score
        if overlap:
            reasons.append(f"Industry match: {', '.join(sorted(overlap))}.")
        else:
            score -= 12
            reasons.append("Industry mismatch: no overlap with requested industries.")

    if functions:
        overlap = functions.intersection(expert.get("functionTags", []))
        function_score = min(25, 25 * (len(overlap) / max(1, len(functions))))
        score += function_score
        if overlap:
            reasons.append(f"Function match: {', '.join(sorted(overlap))}.")

    if levels:
        if expert.get("roleLevel") in levels:
            score += 10
            reasons.append(f"Role level match: {expert.get('roleLevel')}.")

    free_tokens = set(_tokenize(free_text))
    expert_tokens = set(_tokenize(expert.get("expertiseSummary", "")))
    expert_tokens.update([t.lower() for t in expert.get("topicKeywords", [])])
    keyword_overlap = free_tokens.intersection(expert_tokens)
    if free_tokens:
        keyword_score = min(30, 30 * (len(keyword_overlap) / max(1, len(free_tokens))))
        score += keyword_score
        if keyword_overlap:
            reasons.append(f"Keyword overlap: {', '.join(sorted(list(keyword_overlap))[:6])}.")

    cred_text = _flatten_credentials(expert)
    cred_hits = [t for t in free_tokens if t in cred_text]
    if cred_hits:
        score += 10
        reasons.append(f"Credential signal: {', '.join(sorted(set(cred_hits))[:5])}.")

    profile_tokens = set(_tokenize(profile_text))
    if profile_tokens:
        profile_overlap = profile_tokens.intersection(expert_tokens)
        profile_score = min(10, 10 * (len(profile_overlap) / max(1, len(profile_tokens))))
        score += profile_score
        if profile_overlap:
            reasons.append(
                f"Profile match: {', '.join(sorted(list(profile_overlap))[:6])}."
            )

    rate = expert.get("ratePerHour", 0)
    if rate > budget:
        over = min(1.0, (rate - budget) / max(1, budget))
        penalty = 10 * over
        score -= penalty
        reasons.append("Rate above budget; slight penalty applied.")

    score = max(0, min(100, round(score, 1)))
    return score, reasons


def rank_experts(criteria, experts):
    ranked = []
    for expert in experts:
        score, reasons = score_expert(criteria, expert)
        ranked.append(
            {
                "expert": expert,
                "score": score,
                "match_reasons": reasons,
            }
        )
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked
