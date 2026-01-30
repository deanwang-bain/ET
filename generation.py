import hashlib
import json
import random
import re
from datetime import datetime


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


def _normalize_part(part):
    if isinstance(part, (dict, list)):
        try:
            return json.dumps(part, sort_keys=True)
        except Exception:
            return str(part)
    return str(part)


def _stable_seed(*parts):
    joined = "|".join([_normalize_part(p) for p in parts if p is not None])
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _pick_themes(criteria, expert, rng, max_themes=4):
    themes = []
    for item in (criteria.get("functions") or []):
        themes.append(item)
    for item in (criteria.get("industries") or []):
        if item not in themes:
            themes.append(item)
    free_tokens = _tokenize(criteria.get("free_text", ""))
    for token in free_tokens:
        if token not in themes:
            themes.append(token)
    if expert:
        for token in expert.get("topicKeywords", []):
            if token not in themes:
                themes.append(token)
    rng.shuffle(themes)
    return themes[:max_themes]


def generate_script(criteria, expert, length_minutes, tone, depth, refine_text=""):
    seed = _stable_seed(expert.get("id"), criteria, length_minutes, tone, depth, refine_text)
    rng = random.Random(seed)

    industries = ", ".join(criteria.get("industries") or ["General"])
    functions = ", ".join(criteria.get("functions") or ["General"])
    levels = ", ".join(criteria.get("levels") or ["Any level"])
    free_text = criteria.get("free_text") or "No additional context provided."

    objectives = [
        f"Validate practical experience across {industries}.",
        f"Understand key {functions} levers and trade-offs.",
        f"Capture constraints and risks for {levels} roles.",
        "Collect actionable benchmarks and illustrative metrics.",
        "Identify follow-on experts and sources.",
    ]
    rng.shuffle(objectives)
    objectives = objectives[: rng.randint(3, 5)]

    themes = _pick_themes(criteria, expert, rng, max_themes=4)
    if refine_text:
        themes.insert(0, refine_text.strip().capitalize())
        themes = themes[:4]

    qualification_questions = [
        "Describe your direct responsibility for the relevant scope.",
        "What is the most recent project you led in this domain?",
        "Which geographies and segments were in scope?",
        "What were the top constraints or risks you faced?",
        "How did you avoid conflicts of interest or sensitive topics?",
        "What metrics did you personally own or review?",
        "Which stakeholders were decisive in the outcome?",
        "What was out of scope that we should avoid here?",
        "What would be a red-flag sign that this approach fails?",
        "Which trade-offs were most material to decision making?",
    ]
    rng.shuffle(qualification_questions)
    qualification_questions = qualification_questions[: rng.randint(6, 9)]

    main_sections = []
    for theme in themes:
        base_questions = [
            f"What are the top value drivers in {theme} today?",
            f"How do you structure decision criteria for {theme} initiatives?",
            f"What data or metrics are essential to manage {theme} performance?",
            f"Where do teams typically underestimate effort in {theme}?",
            f"How should leaders sequence improvements in {theme}?",
            f"What are the most common failure modes in {theme}?",
        ]
        rng.shuffle(base_questions)
        q_count = 3 if depth == "High-level" else 5
        selected = base_questions[:q_count]
        followups = [
            "Can you share an illustrative example and the key outcomes?",
            "What would you measure in the first 30 days?",
            "How did you align incentives across teams?",
            "What was the fastest lever to change?",
        ]
        rng.shuffle(followups)
        main_sections.append(
            {
                "theme": theme,
                "rationale": f"Prioritized due to relevance to {theme} and the stated request.",
                "questions": selected,
                "followups": followups[:2],
            }
        )

    wrap_up = [
        "What metrics or artifacts should we request to validate assumptions?",
        "Who else should we speak with to triangulate this topic?",
        "What signals would change your recommendation?",
        "What is the most important watch-out for this engagement?",
        "Any final compliance reminders or boundaries we should respect?",
    ]
    rng.shuffle(wrap_up)
    wrap_up = wrap_up[: rng.randint(3, 5)]

    script = []
    script.append(f"# Expert Interview Script: {expert.get('name')}")
    script.append("")
    script.append("## Interview Objective")
    for obj in objectives:
        script.append(f"- {obj}")
    script.append("")
    script.append("## 1) Filtering / Qualification (first 8-10 min)")
    for q in qualification_questions:
        script.append(f"- Q: {q}")
    script.append("- Red flags to watch: lack of direct ownership, outdated experience, or compliance gaps.")
    script.append("- Follow-ups: clarify scope, recency, and avoided topics.")
    script.append("")
    script.append("## 2) Main Business Topics")
    for section in main_sections:
        script.append(f"### Theme: {section['theme']}")
        script.append(f"- Rationale: {section['rationale']}")
        for q in section["questions"]:
            script.append(f"- Q: {q}")
        for fup in section["followups"]:
            script.append(f"- Follow-up: {fup}")
        script.append("")
    script.append("## 3) Wrap-up")
    for q in wrap_up:
        script.append(f"- Q: {q}")
    script.append("- Request metrics or artifacts (clean, non-confidential).")
    script.append("- Suggested next experts: peer function lead, adjacent industry operator.")
    script.append("- Compliance reminder: avoid confidential or MNPI details.")

    return "\n".join(script)


def generate_transcript(criteria, expert, script_text, tone, depth, refine_text=""):
    seed = _stable_seed(
        expert.get("id"),
        criteria,
        script_text,
        tone,
        depth,
        refine_text,
    )
    rng = random.Random(seed)

    themes = _pick_themes(criteria, expert, rng, max_themes=4)
    if refine_text:
        themes.insert(0, refine_text.strip().capitalize())
        themes = themes[:4]

    intro = [
        "Consultant: Thanks for joining today. We will keep this high level and avoid confidential specifics.",
        f"Expert: Happy to help. I'll stay within compliance boundaries: {', '.join(expert.get('complianceFlags', []))}.",
    ]

    q_templates = [
        "Consultant: How would you describe the current state of {theme}?",
        "Consultant: What are the top two levers you see in {theme}?",
        "Consultant: Where do teams lose time or money in {theme}?",
        "Consultant: What metrics do you track for {theme}?",
    ]
    a_templates = [
        "Expert: In my experience, the biggest driver is alignment between teams and clear ownership. We saw illustrative gains of {num}% when governance was tightened.",
        "Expert: The baseline varies, but a reasonable illustrative range is {num}-{num2}%. The key is sequencing the work.",
        "Expert: The first 30 days should focus on data quality and quick wins. We typically measured cycle time and cost-to-serve.",
        "Expert: I would avoid any competitor-sensitive details, but broadly the pattern is consistent across operators.",
    ]

    lines = []
    lines.extend(intro)
    lines.append("")

    for theme in themes:
        for _ in range(2 if depth == "High-level" else 3):
            q = rng.choice(q_templates).format(theme=theme)
            n1 = rng.randint(5, 25)
            n2 = n1 + rng.randint(5, 20)
            a = rng.choice(a_templates).format(num=n1, num2=n2)
            lines.append(q)
            lines.append(a)
            if rng.random() < 0.4:
                lines.append("Consultant: Can you clarify the constraint behind that?")
                lines.append(
                    "Expert: Constraints were usually resource bandwidth and change management; "
                    "we kept efforts scoped to non-confidential data."
                )
        lines.append("")

    lines.append("Consultant: Any final caveats or compliance reminders?")
    lines.append(
        f"Expert: Yes, please avoid confidential client info and MNPI; "
        f"I can provide generalized patterns and illustrative figures only."
    )

    if tone == "Friendly":
        lines.insert(0, "Consultant: Appreciate you taking the time to share your experience.")
    elif tone == "Assertive":
        lines.insert(0, "Consultant: We will keep this focused and time-boxed.")

    return "\n".join(lines)


def _extract_metrics(transcript_text):
    if not transcript_text:
        return []
    metrics = []
    for match in re.finditer(r"(\d{1,3}(?:\.\d+)?)(%|x|k|m)?", transcript_text):
        value = match.group(0)
        start = max(0, match.start() - 40)
        end = min(len(transcript_text), match.end() + 40)
        snippet = transcript_text[start:end].replace("\n", " ")
        metrics.append({"value": value, "context": snippet.strip()})
    return metrics


def summarize_interview(criteria, expert, script_text, transcript_text):
    seed = _stable_seed(expert.get("id"), criteria, script_text, transcript_text, datetime.utcnow().date())
    rng = random.Random(seed)
    themes = _pick_themes(criteria, expert, rng, max_themes=5)

    exec_bullets = [
        "Expert has direct ownership experience aligned with the request.",
        "Insights reflect operational realities and trade-offs in execution.",
        "Illustrative metrics suggest material impact if levers are applied.",
        "Clear constraints and compliance boundaries were reinforced.",
        "Follow-on expert suggestions can deepen validation.",
        "Prioritize data quality and change management early.",
        "Stakeholder alignment is the most common blocker.",
    ]
    rng.shuffle(exec_bullets)
    exec_bullets = exec_bullets[: rng.randint(5, 7)]

    credibility_bullets = [
        f"Role level: {expert.get('roleLevel')} with {expert.get('yearsExperience')} years.",
        f"Industries covered: {', '.join(expert.get('industryTags', []))}.",
        f"Key credentials: {', '.join(expert.get('credentials', {}).get('formerCompanies', [])[:3])}.",
    ]
    credibility_paragraph = (
        f"{expert.get('name')} appears highly relevant based on direct scope ownership and "
        "recent leadership experience. Responses were consistent and bounded by compliance."
    )

    insights = []
    for theme in themes:
        bullets = [
            f"Primary value driver in {theme} is process clarity and ownership.",
            f"Sequencing changes in {theme} reduces risk and rework.",
            f"Metrics discipline is critical to sustain {theme} improvements.",
        ]
        rng.shuffle(bullets)
        insights.append({"theme": theme, "bullets": bullets[: rng.randint(2, 3)]})

    open_questions = [
        "What baseline metrics should be prioritized for benchmarking?",
        "Which constraints are unique to the target client context?",
        "What is the feasible timeline for impact realization?",
        "Which stakeholders will resist change and why?",
    ]
    rng.shuffle(open_questions)
    open_questions = open_questions[: rng.randint(3, 4)]

    next_steps = [
        "Validate themes with a peer expert in an adjacent industry.",
        "Collect anonymized metrics or public benchmarks for triangulation.",
        "Draft a hypothesis tree for the top two levers.",
        "Plan a follow-up with the expert to test open questions.",
    ]
    rng.shuffle(next_steps)
    next_steps = next_steps[: rng.randint(3, 4)]

    metrics = _extract_metrics(transcript_text)

    summary = {
        "executive_summary": exec_bullets,
        "credibility_bullets": credibility_bullets,
        "credibility_paragraph": credibility_paragraph,
        "insights_by_theme": insights,
        "open_questions": open_questions,
        "next_steps": next_steps,
        "metrics": metrics,
        "tags": {
            "industries": criteria.get("industries") or [],
            "functions": criteria.get("functions") or [],
            "levels": criteria.get("levels") or [],
            "topics": themes,
        },
    }
    return summary
