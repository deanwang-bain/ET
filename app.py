import json
import hashlib
import re
from datetime import date, datetime

import pandas as pd
import streamlit as st

from db import init_db, load_by_id, load_case, load_recent, save_case, save_interview
from et_data import (
    EXPERTS,
    FUNCTION_OPTIONS,
    GEOGRAPHY_OPTIONS,
    INDUSTRY_OPTIONS,
    ROLE_LEVELS,
)
from generation import generate_script, generate_transcript, summarize_interview
from llm import generate_json, generate_text, is_available
from scoring import rank_experts


st.set_page_config(page_title="Expert Tool (ET)", layout="wide")
st.markdown(
    """
    <style>
    .stButton > button {
        color: #CB2026;
        border-color: #CB2026;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _criteria_signature(criteria):
    payload = json.dumps(criteria, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _init_state():
    defaults = {
        "case_code": "",
        "case_details": {},
        "case_loaded": False,
        "criteria": {
            "industries": [],
            "functions": [],
            "levels": [],
            "free_text": "",
            "budget": 500,
            "geography": "Any",
            "profile_text": "",
            "linkedin_url": "",
        },
        "criteria_signature": None,
        "search_results": [],
        "top3": [],
        "agency_outreach_email": "",
        "agency_responses": [],
        "agency_synthesis": "",
        "agency_ranked": [],
        "agency_criteria_signature": None,
        "selected_networks": [],
        "selected_expert_id": None,
        "selected_expert": None,
        "selected_match_reasons": [],
        "interview_feedback": None,
        "selected_criteria_signature": None,
        "script_text": None,
        "script_params": {},
        "script_criteria_signature": None,
        "transcript_text": None,
        "transcript_params": {},
        "transcript_criteria_signature": None,
        "summary_text": None,
        "summary_data": None,
        "refine_script_history": [],
        "refine_transcript_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_chips(label, items):
    if not items:
        return
    chips = " ".join(
        [
            f"<span style='background:#eef2f6;border-radius:12px;padding:3px 10px;margin-right:6px;font-size:12px;'>{item}</span>"
            for item in items
        ]
    )
    st.markdown(f"**{label}:** {chips}", unsafe_allow_html=True)


def _render_expert_card(expert, score, reasons):
    st.markdown(f"### {expert['name']}")
    st.markdown(f"**{expert['headline']}**")
    _render_chips("Industry", expert.get("industryTags", []))
    _render_chips("Function", expert.get("functionTags", []))
    _render_chips("Level", [expert.get("roleLevel")])
    st.markdown(
        f"- {expert['yearsExperience']} yrs exp | {expert['geography']} | "
        f"Languages: {', '.join(expert['languages'])} | Rate: ${expert['ratePerHour']}/hr"
    )
    creds = expert.get("credentials", {})
    st.markdown(
        f"- **Credentials:** {creds.get('education')}; Former: {', '.join(creds.get('formerCompanies', []))}"
    )
    st.markdown(f"- **Publications:** {', '.join(creds.get('publications', []))}")
    st.markdown(f"- **Expertise summary:** {expert['expertiseSummary']}")
    st.markdown(f"- **Topic keywords:** {', '.join(expert.get('topicKeywords', []))}")
    st.markdown(f"- **Availability:** {expert['availability']}")
    st.markdown(f"- **Compliance:** {', '.join(expert.get('complianceFlags', []))}")
    if expert.get("cidCleared"):
        st.markdown(
            "<span style='color:#1a7f37;font-weight:600;'>CID cleared</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<a href='https://conflictid.bain.com/' style='color:#CB2026;font-weight:600;'>Clear CID</a>",
            unsafe_allow_html=True,
        )
    st.markdown(f"**Score:** {score}")
    if reasons:
        st.markdown("**Why this match:**")
        for reason in reasons:
            st.markdown(f"- {reason}")


def _stale_warning(message):
    st.warning(message, icon="⚠️")


def _build_summary_markdown(summary):
    lines = []
    lines.append("## Executive Summary")
    for bullet in summary["executive_summary"]:
        lines.append(f"- {bullet}")
    lines.append("")
    lines.append("## Expert Credibility & Relevance")
    for bullet in summary["credibility_bullets"]:
        lines.append(f"- {bullet}")
    lines.append(summary["credibility_paragraph"])
    lines.append("")
    lines.append("## Key Insights by Theme")
    for item in summary["insights_by_theme"]:
        lines.append(f"**{item['theme']}**")
        for bullet in item["bullets"]:
            lines.append(f"- {bullet}")
    lines.append("")
    lines.append("## Open Questions / Uncertainties")
    for bullet in summary["open_questions"]:
        lines.append(f"- {bullet}")
    lines.append("")
    lines.append("## Suggested Next Steps")
    for bullet in summary["next_steps"]:
        lines.append(f"- {bullet}")
    return "\n".join(lines)


def _llm_enabled():
    return is_available()


NETWORKS = [
    "NEXUS",
    "PRIME",
    "ALIGN",
    "WISDOM",
    "LENS",
    "OUTREACH",
    "BRIDGE",
    "CIRCLE",
]

AGENCY_TEMPLATE_BRAND = """BRAND - EXPERTS

{expert_name} - Candidate

Former {expert_headline}

BIOGRAPHY
Geo: {geo}

Credit: TBD

{bio_paragraph}

AVAILABILITY (WIB)
{availability}

SCREENING QUESTIONS
1. {screen_q1}

{screen_a1}

2. {screen_q2}

{screen_a2}

3. {screen_q3}

{screen_a3}
"""

AGENCY_TEMPLATE_SHORT = """Hello {contact_name},

Jumping in for the team here. We have screened {expert_name}, former {expert_headline}, who is {availability_short}. Would you like to book them?

Also, checking in on your timeline for calls and any new priorities.

Profile: {expert_name} | {expert_headline}
Relevant experience: {relevant_experience}
Screened: Yes

Top highlights:
- {highlight_1}
- {highlight_2}
- {highlight_3}
"""

PROFILE_STOPWORDS = {
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
}


def _parse_date(value, fallback):
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return fallback
    return fallback


def _summarize_profile_text(text, max_terms=8):
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    tokens = [t for t in tokens if t not in PROFILE_STOPWORDS and len(t) > 2]
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [term for term, _ in ranked[:max_terms]]


def _render_agency_email_body(
    template_style,
    expert,
    case_details,
    contact_name="Team",
):
    screen_q1 = case_details.get("screening_1") or "Are you familiar with this market?"
    screen_q2 = case_details.get("screening_2") or "Can you speak to major competitors?"
    screen_q3 = case_details.get("screening_3") or "Can you discuss recent shifts in the industry?"
    bio_paragraph = (
        f"{expert['name']} is a former {expert['headline']} with {expert['yearsExperience']} "
        "years of experience. Their expertise spans "
        f"{', '.join(expert.get('topicKeywords', []))}."
    )
    availability = (
        expert.get("availability")
        or "This expert has not provided availability. We can expedite upon request."
    )
    availability_short = (
        "available for a call this week" if "Available" in availability else "pending availability"
    )
    relevant_experience = "; ".join(expert.get("industryTags", [])) or expert.get("headline")
    highlight_1 = f"Led initiatives in {', '.join(expert.get('functionTags', []))}."
    highlight_2 = f"Experienced across {', '.join(expert.get('industryTags', []))}."
    highlight_3 = f"Languages: {', '.join(expert.get('languages', []))}."

    if template_style == "brand":
        return AGENCY_TEMPLATE_BRAND.format(
            expert_name=expert["name"],
            expert_headline=expert["headline"],
            geo=expert.get("geography", "N/A"),
            bio_paragraph=bio_paragraph,
            availability=availability,
            screen_q1=screen_q1,
            screen_a1="I can cover these on call.",
            screen_q2=screen_q2,
            screen_a2="Can cover this briefly.",
            screen_q3=screen_q3,
            screen_a3="Happy to provide more detail on call.",
        )

    return AGENCY_TEMPLATE_SHORT.format(
        contact_name=contact_name,
        expert_name=expert["name"],
        expert_headline=expert["headline"],
        availability_short=availability_short,
        relevant_experience=relevant_experience,
        highlight_1=highlight_1,
        highlight_2=highlight_2,
        highlight_3=highlight_3,
    )

def _suggest_template_defaults(case_details):
    industry = case_details.get("industry") or ""
    case_topic = case_details.get("case_topic") or ""
    industry_lower = industry.lower()
    defaults = {
        "question_1": f"What is the current market size and growth outlook for {industry or 'the sector'}?",
        "question_2": f"Who are the major competitors and how is market share shifting in {industry or 'this market'}?",
        "question_3": f"What recent trends are reshaping demand or pricing in {industry or 'the sector'}?",
        "company_type": industry or "relevant companies",
        "min_employees": "500",
        "target_geographies": "North America and Europe",
        "title_1": "Chief Operating Officer",
        "title_2": "Head of Strategy",
        "title_3": "VP/Director of Operations",
        "desired_level": "senior",
        "case_issue": case_topic or "the case topic",
        "screening_1": "Are you familiar with this market and its value chain?",
        "screening_2": "Can you outline the major competitors and their positioning?",
        "screening_3": "Can you speak to recent shifts in the market over the last 12-24 months?",
    }
    if "saas" in industry_lower:
        defaults.update(
            {
                "question_1": "What is the TAM/SAM/SOM and growth outlook for this SaaS segment?",
                "question_2": "Which competitors are leading and where are switching dynamics changing?",
                "question_3": "What trends (pricing, PLG, AI, security) are shaping buying decisions?",
                "title_3": "VP/Director of Product",
            }
        )
    if "health" in industry_lower:
        defaults.update(
            {
                "question_1": "What is the size and growth of the target care/service segment?",
                "question_2": "Who are the major providers and payers influencing dynamics?",
                "question_3": "What regulatory or reimbursement shifts matter most?",
                "title_2": "Director of Clinical Operations",
            }
        )
    if case_topic and case_topic not in defaults["case_issue"]:
        defaults["case_issue"] = case_topic
    return defaults


def _format_expert_list(experts):
    return [
        {
            "id": expert["id"],
            "name": expert["name"],
            "headline": expert["headline"],
            "industryTags": expert["industryTags"],
            "functionTags": expert["functionTags"],
            "roleLevel": expert["roleLevel"],
            "ratePerHour": expert["ratePerHour"],
            "availability": expert["availability"],
            "complianceFlags": expert["complianceFlags"],
        }
        for expert in experts
    ]


def _load_email_template():
    try:
        with open("email_template.md", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return ""


def _fill_email_template(template, case_details, criteria):
    question_1 = case_details.get("question_1") or ""
    question_2 = case_details.get("question_2") or ""
    question_3 = case_details.get("question_3") or ""
    company_type = case_details.get("company_type") or "relevant companies"
    min_employees = case_details.get("min_employees") or "N/A"
    geographies = case_details.get("target_geographies") or "global"
    ex1 = case_details.get("example_company_1") or "Example 1"
    ex2 = case_details.get("example_company_2") or "Example 2"
    ex3 = case_details.get("example_company_3") or "Example 3"
    suppliers = case_details.get("potential_suppliers") or ""
    customers = case_details.get("potential_customers") or ""
    title_1 = case_details.get("title_1") or "Title 1"
    title_2 = case_details.get("title_2") or "Title 2"
    title_3 = case_details.get("title_3") or "Title 3"
    desired_level = case_details.get("desired_level") or "senior"
    case_issue = case_details.get("case_issue") or "the case topic"
    screening_1 = case_details.get("screening_1") or ""
    screening_2 = case_details.get("screening_2") or ""
    screening_3 = case_details.get("screening_3") or ""

    background_line = (
        f"Case topic: {case_details.get('case_topic') or 'TBD'}. "
        f"Request context: {criteria.get('free_text') or 'No additional context provided.'}"
    )

    filled = template.replace(
        "Background / request",
        f"Background / request\n{background_line}",
    )
    if question_1:
        filled = filled.replace(
            "[Important case question #1 (e.g., market size)]", question_1
        )
    if question_2:
        filled = filled.replace(
            "[Important case question #2 (e.g., major competitors)]", question_2
        )
    if question_3:
        filled = filled.replace(
            "[Important case question #3 (e.g., recent shifts/trends in the market)]",
            question_3,
        )
    filled = filled.replace("[company type]", company_type)
    filled = filled.replace("[XXX]", str(min_employees))
    filled = filled.replace("[mix of geographies/targeted geographies]", geographies)
    filled = filled.replace("Title 1 (e.g., Chief Operating Officer)", title_1)
    filled = filled.replace("Title 2 (e.g., Public Relations Executive)", title_2)
    filled = filled.replace("Title 3 (e.g., Financial Director)", title_3)
    filled = filled.replace(
        "Any other [desired employee level, e.g., executive or decision-making] employees with [case issue]",
        f"Any other {desired_level} employees with {case_issue}",
    )
    if screening_1:
        filled = filled.replace("[Question 1 (e.g., are you familiar with this market?)]", screening_1)
    if screening_2:
        filled = filled.replace("[Question 2 (e.g., can you list the major competitors in this market?)]", screening_2)
    if screening_3:
        filled = filled.replace("[Question 3 (e.g., can you speak about recent shifts in the industry?)]", screening_3)
    filled = filled.replace(
        "Firms in the industry: Example 1 [12 month block], Example 2 [12 month block], Example 3etc.",
        f"Firms in the industry: {ex1} [12 month block], {ex2} [12 month block], {ex3} [12 month block]",
    )
    if suppliers:
        filled = filled.replace("Potential Suppliers:", f"Potential Suppliers: {suppliers}")
    if customers:
        filled = filled.replace("Potential Customers:", f"Potential Customers: {customers}")
    return filled


def _draft_agency_email(criteria, case_details, networks, ranked):
    industries = ", ".join(criteria.get("industries") or ["General"])
    functions = ", ".join(criteria.get("functions") or ["General"])
    levels = ", ".join(criteria.get("levels") or ["Any level"])
    free_text = criteria.get("free_text") or "No additional context provided."
    budget = criteria.get("budget")
    geography = criteria.get("geography")
    profile_text = criteria.get("profile_text") or ""
    linkedin_url = criteria.get("linkedin_url") or ""
    profile_terms = _summarize_profile_text(profile_text)
    template = _load_email_template()
    filled = _fill_email_template(template, case_details, criteria) if template else ""
    top_experts = ranked[:5] if ranked else []
    expert_summaries = []
    for item in top_experts:
        expert = item["expert"]
        expert_summaries.append(
            f"- {expert['name']} ({expert['headline']}) | "
            f"{expert['geography']} | {expert['yearsExperience']} yrs | "
            f"Rate ${expert['ratePerHour']}/hr | "
            f"Topics: {', '.join(expert.get('topicKeywords', []))}"
        )

    if _llm_enabled():
        system_prompt = "You are a consulting research coordinator drafting outreach emails to expert agencies."
        user_prompt = (
            "Fill the provided email template with the supplied details. "
            "Keep the structure and section headings unchanged. "
            "After the template, add a section titled 'Proposed Expert Shortlist (ET draft)' "
            "with brief expert summaries.\n\n"
            "Use a professional consulting tone.\n\n"
            f"Template:\n{template}\n\n"
            f"Industries: {industries}\nFunctions: {functions}\nLevels: {levels}\n"
            f"Budget: ${budget}/hr\nGeography: {geography}\nRequest: {free_text}\n"
            f"LinkedIn URL: {linkedin_url}\n"
            f"Profile keywords: {profile_terms}\n"
            f"Selected networks: {', '.join(networks) or 'All'}\n"
            f"Case details: {json.dumps(case_details, sort_keys=True)}\n"
            f"Top experts: {json.dumps(expert_summaries, ensure_ascii=False)}\n"
        )
        text = generate_text(system_prompt, user_prompt, max_tokens=900)
        if text:
            return text

    if filled:
        extra = []
        if linkedin_url:
            extra.append(f"LinkedIn URL: {linkedin_url}")
        if profile_terms:
            extra.append(f"Profile keywords: {', '.join(profile_terms)}")
        extra_block = ""
        if extra:
            extra_block = "\n\nProfile signal\n" + "\n".join(f"- {item}" for item in extra)
        if expert_summaries:
            return (
                f"{filled}"
                f"{extra_block}\n\nProposed Expert Shortlist (ET draft)\n"
                + "\n".join(expert_summaries)
            )
        if extra_block:
            return f"{filled}{extra_block}"
        return filled

    return f"Background / request\n{free_text}\n\n(Targeted populations and screening questions pending.)"


def _simulate_agency_responses(criteria, case_details, experts, ranked, networks):
    if _llm_enabled():
        system_prompt = "You are an expert-network agency responding to a consulting request."
        user_prompt = (
            "Simulate emails from the selected agencies using one of the two provided templates. "
            "Return JSON with key 'agencies', an array of objects "
            "with fields: agency_name, email_subject, email_body, recommended_experts (array). "
            "Each recommended_expert must include id, name, fit_reason, availability, rate. "
            "Use only the experts provided below by id/name.\n\n"
            f"Template A:\n{AGENCY_TEMPLATE_BRAND}\n\n"
            f"Template B:\n{AGENCY_TEMPLATE_SHORT}\n\n"
            f"Request criteria: {json.dumps(criteria, sort_keys=True)}\n"
            f"Case details: {json.dumps(case_details, sort_keys=True)}\n"
            f"Selected networks: {', '.join(networks) or 'All'}\n"
            f"Experts: {json.dumps(_format_expert_list(experts), ensure_ascii=False)}\n"
        )
        data = generate_json(system_prompt, user_prompt, max_tokens=900)
        if data and isinstance(data.get("agencies"), list):
            return data["agencies"]

    top = ranked[:6]
    agencies = []
    selected = networks or ["NEXUS", "PRIME", "ALIGN"]
    for idx, network in enumerate(selected):
        if idx * 2 + 1 >= len(top):
            break
        style = "brand" if idx % 2 == 0 else "short"
        email_body = _render_agency_email_body(
            style, top[idx * 2]["expert"], case_details
        )
        agencies.append(
            {
                "agency_name": network,
                "email_subject": "Re: Expert recommendations",
                "email_body": email_body,
                "recommended_experts": [
                    {
                        "id": top[idx * 2]["expert"]["id"],
                        "name": top[idx * 2]["expert"]["name"],
                        "fit_reason": "Strong match on industry and function.",
                        "availability": top[idx * 2]["expert"]["availability"],
                        "rate": top[idx * 2]["expert"]["ratePerHour"],
                    },
                    {
                        "id": top[idx * 2 + 1]["expert"]["id"],
                        "name": top[idx * 2 + 1]["expert"]["name"],
                        "fit_reason": "Recent leadership experience with similar scope.",
                        "availability": top[idx * 2 + 1]["expert"]["availability"],
                        "rate": top[idx * 2 + 1]["expert"]["ratePerHour"],
                    },
                ],
            }
        )
    return agencies


def _history_bonus(expert, recent_history):
    if not recent_history:
        return 0
    bonus = 0
    industry_tags = set(expert.get("industryTags", []))
    function_tags = set(expert.get("functionTags", []))
    topic_tags = set([t.lower() for t in expert.get("topicKeywords", [])])
    for item in recent_history:
        tags = item.get("tags", {})
        bonus += 2 * len(industry_tags.intersection(tags.get("industries", [])))
        bonus += 2 * len(function_tags.intersection(tags.get("functions", [])))
        bonus += len(topic_tags.intersection([t.lower() for t in tags.get("topics", [])]))
    return min(10, bonus)


def _synthesize_recommendations(criteria, experts, ranked, agency_responses, recent_history):
    agency_mentions = {}
    agency_reasons = {}
    for agency in agency_responses:
        for rec in agency.get("recommended_experts", []):
            expert_id = rec.get("id")
            if not expert_id:
                continue
            agency_mentions[expert_id] = agency_mentions.get(expert_id, 0) + 1
            reason = f"{agency.get('agency_name')}: {rec.get('fit_reason')}"
            agency_reasons.setdefault(expert_id, []).append(reason)

    base_scores = {item["expert"]["id"]: item["score"] for item in ranked}
    compiled = []
    for expert in experts:
        base = base_scores.get(expert["id"], 0)
        agency_bonus = min(15, 5 * agency_mentions.get(expert["id"], 0))
        history_bonus = _history_bonus(expert, recent_history)
        final_score = max(0, min(100, round(base + agency_bonus + history_bonus, 1)))
        reasons = []
        if agency_bonus:
            reasons.append(f"Agency recommendations: {agency_mentions.get(expert['id'], 0)}")
            reasons.extend(agency_reasons.get(expert["id"], []))
        if history_bonus:
            reasons.append("Aligned with past interview history tags.")
        compiled.append(
            {
                "expert": expert,
                "score": final_score,
                "match_reasons": reasons or ["Matches core criteria."],
            }
        )
    compiled.sort(key=lambda r: r["score"], reverse=True)
    return compiled


def _serialize_ranked(ranked):
    data = []
    for item in ranked:
        data.append(
            {
                "id": item["expert"]["id"],
                "score": item["score"],
                "match_reasons": item["match_reasons"],
            }
        )
    return data


def _deserialize_ranked(serialized, experts):
    expert_map = {expert["id"]: expert for expert in experts}
    ranked = []
    for item in serialized or []:
        expert = expert_map.get(item.get("id"))
        if not expert:
            continue
        ranked.append(
            {
                "expert": expert,
                "score": item.get("score", 0),
                "match_reasons": item.get("match_reasons", []),
            }
        )
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked


def _persist_case():
    case_code = st.session_state.get("case_code")
    if not case_code:
        return
    payload = {
        "case_details": st.session_state.get("case_details", {}),
        "criteria": st.session_state.get("criteria", {}),
        "criteria_signature": st.session_state.get("criteria_signature"),
        "agency_criteria_signature": st.session_state.get("agency_criteria_signature"),
        "selected_networks": st.session_state.get("selected_networks", []),
        "agency_outreach_email": st.session_state.get("agency_outreach_email", ""),
        "agency_responses": st.session_state.get("agency_responses", []),
        "agency_synthesis": st.session_state.get("agency_synthesis", ""),
        "agency_ranked": _serialize_ranked(st.session_state.get("agency_ranked", [])),
        "selected_expert_id": st.session_state.get("selected_expert_id"),
        "script_text": st.session_state.get("script_text"),
        "transcript_text": st.session_state.get("transcript_text"),
        "summary_text": st.session_state.get("summary_text"),
        "interview_feedback": st.session_state.get("interview_feedback"),
    }
    save_case(case_code, payload)


_init_state()
init_db()

st.title("Expert Tool (ET)")
st.info(
    "Compliance reminder: Do not solicit or share confidential client information or MNPI. "
    "Use only generalized, illustrative insights."
)

tab_case, tab_search, tab_script, tab_summary = st.tabs(
    [
        "0) Case Details",
        "1) Search + Agency",
        "2) Interview Script + Transcript",
        "3) Summary + Save",
    ]
)

with tab_case:
    st.subheader("Case details")
    case_details = st.session_state.get("case_details", {})
    template_defaults = _suggest_template_defaults(case_details)
    col1, col2 = st.columns(2)
    with col1:
        case_background_options = [
            "Active Case Request",
            "Case Pre-build",
            "Corporate Case Live",
            "Corporate Case CD",
            "PEG Live",
            "PEG CD",
            "Other",
        ]
        case_background = st.radio(
            "Case Background *",
            case_background_options,
            index=case_background_options.index(case_details.get("case_background"))
            if case_details.get("case_background") in case_background_options
            else 0,
            horizontal=True,
        )
        due_diligence_options = ["Yes", "No"]
        due_diligence = st.radio(
            "Due Diligence Case",
            due_diligence_options,
            index=due_diligence_options.index(case_details.get("due_diligence"))
            if case_details.get("due_diligence") in due_diligence_options
            else 0,
            horizontal=True,
        )
        case_code = st.text_input(
            "Case Code *",
            value=st.session_state.get("case_code", ""),
            help="Use a consistent code to access the same expert set.",
        )
        case_team_contact = st.text_input(
            "Case Team Contact",
            value=case_details.get("case_team_contact", ""),
        )
    with col2:
        industry_options = ["Select industry"] + INDUSTRY_OPTIONS
        selected_industry = st.selectbox(
            "Select Industry *",
            options=industry_options,
            index=industry_options.index(case_details.get("industry", "Select industry"))
            if case_details.get("industry") in industry_options
            else 0,
        )
        case_team_cc = st.text_input(
            "Case Team Members to be CC'd",
            value=", ".join(case_details.get("case_team_cc", [])),
        )
        case_topic = st.text_input(
            "Case Topic *",
            value=case_details.get("case_topic", ""),
        )

    st.markdown("**Off Limit Companies (former employees, 12 month out OK)**")
    off1, off2, off3 = st.columns(3)
    with off1:
        off_limit_1 = st.text_input("Off limit company 1", value=case_details.get("off_limit_1", ""))
    with off2:
        off_limit_2 = st.text_input("Off limit company 2", value=case_details.get("off_limit_2", ""))
    with off3:
        off_limit_3 = st.text_input("Off limit company 3", value=case_details.get("off_limit_3", ""))

    col3, col4, col5 = st.columns(3)
    with col3:
        target_calls_value = case_details.get("target_calls", 3)
        if not isinstance(target_calls_value, int) or target_calls_value < 1:
            target_calls_value = 3
        target_calls_value = min(10, target_calls_value)
        target_calls = st.selectbox(
            "Target Calls",
            options=list(range(1, 11)),
            index=target_calls_value - 1,
        )
    with col4:
        start_date = st.date_input(
            "Start Date Call *",
            value=_parse_date(case_details.get("start_date"), date.today()),
        )
    with col5:
        end_date = st.date_input(
            "End Date Call *",
            value=_parse_date(case_details.get("end_date"), date.today()),
        )

    with st.expander("Email template inputs"):
        q1 = st.text_input(
            "Important case question #1",
            value=case_details.get("question_1") or template_defaults["question_1"],
        )
        q2 = st.text_input(
            "Important case question #2",
            value=case_details.get("question_2") or template_defaults["question_2"],
        )
        q3 = st.text_input(
            "Important case question #3",
            value=case_details.get("question_3") or template_defaults["question_3"],
        )
        company_type = st.text_input(
            "Company type",
            value=case_details.get("company_type") or template_defaults["company_type"],
        )
        min_employees = st.text_input(
            "Minimum employees",
            value=case_details.get("min_employees") or template_defaults["min_employees"],
        )
        target_geographies = st.text_input(
            "Target geographies",
            value=case_details.get("target_geographies")
            or template_defaults["target_geographies"],
        )
        st.markdown("**Examples of relevant companies**")
        ex1 = st.text_input("Example company 1", value=case_details.get("example_company_1", ""))
        ex2 = st.text_input("Example company 2", value=case_details.get("example_company_2", ""))
        ex3 = st.text_input("Example company 3", value=case_details.get("example_company_3", ""))
        suppliers = st.text_input("Potential Suppliers", value=case_details.get("potential_suppliers", ""))
        customers = st.text_input("Potential Customers", value=case_details.get("potential_customers", ""))
        st.markdown("**Targeted positions**")
        title_1 = st.text_input(
            "Title 1", value=case_details.get("title_1") or template_defaults["title_1"]
        )
        title_2 = st.text_input(
            "Title 2", value=case_details.get("title_2") or template_defaults["title_2"]
        )
        title_3 = st.text_input(
            "Title 3", value=case_details.get("title_3") or template_defaults["title_3"]
        )
        desired_level = st.text_input(
            "Desired employee level",
            value=case_details.get("desired_level") or template_defaults["desired_level"],
        )
        case_issue = st.text_input(
            "Case issue", value=case_details.get("case_issue") or template_defaults["case_issue"]
        )
        st.markdown("**Screening questions**")
        screening_1 = st.text_input(
            "Screening question 1",
            value=case_details.get("screening_1") or template_defaults["screening_1"],
        )
        screening_2 = st.text_input(
            "Screening question 2",
            value=case_details.get("screening_2") or template_defaults["screening_2"],
        )
        screening_3 = st.text_input(
            "Screening question 3",
            value=case_details.get("screening_3") or template_defaults["screening_3"],
        )

    st.session_state["case_code"] = case_code.strip()
    st.session_state["case_details"] = {
        "case_background": case_background,
        "due_diligence": due_diligence,
        "industry": "" if selected_industry == "Select industry" else selected_industry,
        "case_team_contact": case_team_contact,
        "case_team_cc": [x.strip() for x in case_team_cc.split(",") if x.strip()],
        "case_topic": case_topic,
        "off_limit_1": off_limit_1,
        "off_limit_2": off_limit_2,
        "off_limit_3": off_limit_3,
        "target_calls": target_calls,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "question_1": q1,
        "question_2": q2,
        "question_3": q3,
        "company_type": company_type,
        "min_employees": min_employees,
        "target_geographies": target_geographies,
        "example_company_1": ex1,
        "example_company_2": ex2,
        "example_company_3": ex3,
        "potential_suppliers": suppliers,
        "potential_customers": customers,
        "title_1": title_1,
        "title_2": title_2,
        "title_3": title_3,
        "desired_level": desired_level,
        "case_issue": case_issue,
        "screening_1": screening_1,
        "screening_2": screening_2,
        "screening_3": screening_3,
    }

    col_save, col_load = st.columns(2)
    with col_save:
        if st.button("Save case details"):
            if not st.session_state["case_code"]:
                st.error("Case code is required to save.")
            else:
                _persist_case()
                st.success("Case details saved.")
    with col_load:
        if st.button("Load case"):
            if not st.session_state["case_code"]:
                st.error("Enter a case code to load.")
            else:
                record = load_case(st.session_state["case_code"])
                if record:
                    data = record.get("data", {})
                    st.session_state["case_details"] = data.get("case_details", {})
                    st.session_state["criteria"] = data.get("criteria", st.session_state["criteria"])
                    st.session_state["criteria_signature"] = data.get("criteria_signature")
                    st.session_state["agency_criteria_signature"] = data.get("agency_criteria_signature")
                    st.session_state["selected_networks"] = data.get("selected_networks", [])
                    st.session_state["agency_outreach_email"] = data.get("agency_outreach_email", "")
                    st.session_state["agency_responses"] = data.get("agency_responses", [])
                    st.session_state["agency_synthesis"] = data.get("agency_synthesis", "")
                    st.session_state["agency_ranked"] = _deserialize_ranked(
                        data.get("agency_ranked", []), EXPERTS
                    )
                    st.session_state["interview_feedback"] = data.get("interview_feedback")
                    st.session_state["selected_expert_id"] = data.get("selected_expert_id")
                    if st.session_state["selected_expert_id"]:
                        for expert in EXPERTS:
                            if expert["id"] == st.session_state["selected_expert_id"]:
                                st.session_state["selected_expert"] = expert
                                break
                    st.session_state["script_text"] = data.get("script_text")
                    st.session_state["transcript_text"] = data.get("transcript_text")
                    st.session_state["summary_text"] = data.get("summary_text")
                    st.session_state["case_loaded"] = True
                    st.success(f"Loaded case {record['case_code']} (updated {record['updated_at']}).")
                else:
                    st.session_state["case_loaded"] = True
                    _persist_case()
                    st.success("New case initialized.")

with tab_search:
    st.subheader("Search criteria and agency workflow")
    case_ready = bool(st.session_state.get("case_code"))
    if not case_ready:
        st.info("Enter a case code in the Case Details tab to begin.")
    else:
        if (
            st.session_state.get("agency_responses")
            and st.session_state["criteria_signature"]
            != st.session_state.get("agency_criteria_signature")
        ):
            _stale_warning(
                "Stale context: agency responses tied to older criteria. Regenerate recommended."
            )
        col1, col2, col3 = st.columns(3)
        with col1:
            default_industries = st.session_state["criteria"].get("industries", [])
            if not default_industries and st.session_state["case_details"].get("industry"):
                default_industries = [st.session_state["case_details"]["industry"]]
            industries = st.multiselect(
                "Industry",
                options=INDUSTRY_OPTIONS,
                default=default_industries,
            )
            functions = st.multiselect(
                "Function",
                options=FUNCTION_OPTIONS,
                default=st.session_state["criteria"].get("functions", []),
            )
        with col2:
            levels = st.multiselect(
                "Role level",
                options=ROLE_LEVELS,
                default=st.session_state["criteria"].get("levels", []),
            )
            geography = st.selectbox(
                "Geography preference",
                options=GEOGRAPHY_OPTIONS,
                index=GEOGRAPHY_OPTIONS.index(
                    st.session_state["criteria"].get("geography", "Any")
                ),
            )
        with col3:
            budget = st.slider(
                "Budget per hour",
                min_value=100,
                max_value=1500,
                value=st.session_state["criteria"].get("budget", 500),
                step=50,
            )

        default_free_text = st.session_state["criteria"].get("free_text", "")
        if not default_free_text:
            case_topic = st.session_state["case_details"].get("case_topic", "")
            default_free_text = case_topic
        free_text = st.text_area(
            "Further request / context",
            value=default_free_text,
            height=120,
        )

        st.markdown("**Expert profile matching (optional)**")
        cv_file = st.file_uploader(
            "Upload CV (TXT/MD only)",
            type=["txt", "md"],
            help="Upload a text CV to help match similar expert backgrounds.",
        )
        resume_text = ""
        if cv_file is not None:
            try:
                resume_text = cv_file.read().decode("utf-8")
            except UnicodeDecodeError:
                st.warning("Unable to read CV file. Please upload a UTF-8 text file.")
        linkedin_url = st.text_input(
            "LinkedIn profile link (optional)",
            value=st.session_state["criteria"].get("linkedin_url", ""),
        )
        profile_notes = st.text_area(
            "Profile notes or pasted LinkedIn summary (optional)",
            value=st.session_state["criteria"].get("profile_text", ""),
            height=100,
        )
        profile_text = "\n".join([resume_text, profile_notes]).strip()

        if st.button("Find experts", type="primary"):
            criteria = {
                "industries": industries,
                "functions": functions,
                "levels": levels,
                "free_text": free_text.strip(),
                "budget": budget,
                "geography": geography,
                "profile_text": profile_text,
                "linkedin_url": linkedin_url.strip(),
            }
            st.session_state["criteria"] = criteria
            st.session_state["criteria_signature"] = _criteria_signature(criteria)
            ranked = rank_experts(criteria, EXPERTS)
            st.session_state["search_results"] = ranked
            st.session_state["top3"] = ranked[:3]
            _persist_case()
            st.success("Search criteria saved. Awaiting agency responses.")

        st.markdown("---")
        st.subheader("Expert networks")
        select_all = st.checkbox(
            "Select all networks",
            value=(
                len(st.session_state.get("selected_networks", [])) == 0
                or len(st.session_state.get("selected_networks", [])) == len(NETWORKS)
            ),
        )
        if select_all:
            selected_networks = NETWORKS
        else:
            selected_networks = st.multiselect(
                "Networks",
                options=NETWORKS,
                default=st.session_state.get("selected_networks", []),
            )
        st.session_state["selected_networks"] = selected_networks

        st.markdown("---")
        st.subheader("Agency outreach workflow")
        if _llm_enabled():
            st.caption("LLM enabled via OPENAI_API_KEY.")
        else:
            st.caption("LLM not detected. Using template fallback.")

        if st.button("Draft agency outreach email"):
            if not st.session_state["search_results"]:
                st.session_state["search_results"] = rank_experts(
                    st.session_state["criteria"], EXPERTS
                )
            st.session_state["agency_outreach_email"] = _draft_agency_email(
                st.session_state["criteria"],
                st.session_state["case_details"],
                st.session_state["selected_networks"],
                st.session_state["search_results"],
            )
            st.session_state["agency_criteria_signature"] = st.session_state[
                "criteria_signature"
            ]
            _persist_case()
            st.success("Draft email generated.")

        if st.session_state.get("agency_outreach_email"):
            st.text_area(
                "Draft email to agencies",
                value=st.session_state["agency_outreach_email"],
                height=260,
            )

        if st.button("Simulate agency responses"):
            if not st.session_state["search_results"]:
                st.session_state["search_results"] = rank_experts(
                    st.session_state["criteria"], EXPERTS
                )
            agencies = _simulate_agency_responses(
                st.session_state["criteria"],
                st.session_state["case_details"],
                EXPERTS,
                st.session_state["search_results"],
                st.session_state["selected_networks"],
            )
            st.session_state["agency_responses"] = agencies
            st.session_state["agency_criteria_signature"] = st.session_state[
                "criteria_signature"
            ]
            _persist_case()
            st.success("Agency responses simulated.")

        if st.session_state.get("agency_responses"):
            with st.expander("View agency emails"):
                for agency in st.session_state["agency_responses"]:
                    st.markdown(f"**{agency.get('agency_name', 'Agency')}**")
                    st.markdown(f"Subject: {agency.get('email_subject', '')}")
                    st.markdown(agency.get("email_body", ""))
                    recs = agency.get("recommended_experts", [])
                    if recs:
                        st.markdown("Recommended experts:")
                        for rec in recs:
                            st.markdown(
                                f"- {rec.get('id')} {rec.get('name')}: {rec.get('fit_reason')}"
                            )
                    st.markdown("---")

            if st.button("Synthesize agency responses + interview history"):
                recent = load_recent(limit=20)
                compiled = _synthesize_recommendations(
                    st.session_state["criteria"],
                    EXPERTS,
                    st.session_state["search_results"],
                    st.session_state["agency_responses"],
                    recent,
                )
                st.session_state["agency_ranked"] = compiled
                st.session_state["agency_criteria_signature"] = st.session_state[
                    "criteria_signature"
                ]
                if _llm_enabled():
                    system_prompt = "You are a consulting research lead synthesizing expert recommendations."
                    user_prompt = (
                        "Provide a concise synthesis of agency responses and past interview history. "
                        "Highlight why the top experts stand out and any gaps.\n\n"
                        f"Criteria: {json.dumps(st.session_state['criteria'], sort_keys=True)}\n"
                        f"Agency responses: {json.dumps(st.session_state['agency_responses'], ensure_ascii=False)}\n"
                        f"Recent history tags: {json.dumps(recent, ensure_ascii=False)}\n"
                    )
                    st.session_state["agency_synthesis"] = generate_text(
                        system_prompt, user_prompt, max_tokens=400
                    ) or ""
                else:
                    st.session_state["agency_synthesis"] = (
                        "Synthesized recommendations prioritize experts with strong criteria match, "
                        "agency endorsements, and alignment to recent interview themes."
                    )
                _persist_case()
                st.success("ET synthesis completed.")
        else:
            st.info("Simulate agency responses to enable synthesis.")

        if st.session_state.get("agency_synthesis"):
            st.markdown("**Synthesis summary**")
            st.markdown(st.session_state["agency_synthesis"])

        if st.session_state.get("agency_ranked"):
            st.markdown("**ET recommended list (ranked)**")
            rows = []
            for item in st.session_state["agency_ranked"][:10]:
                expert = item["expert"]
                rows.append(
                    {
                        "ID": expert["id"],
                        "Name": expert["name"],
                        "Headline": expert["headline"],
                        "Industry": ", ".join(expert["industryTags"]),
                        "Function": ", ".join(expert["functionTags"]),
                        "Level": expert["roleLevel"],
                        "Rate": expert["ratePerHour"],
                        "Score": item["score"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.markdown("---")
            st.subheader("Top 3 selectable experts")
            cols = st.columns(3)
            for idx, item in enumerate(st.session_state["agency_ranked"][:3]):
                expert = item["expert"]
                with cols[idx]:
                    with st.container():
                        _render_expert_card(expert, item["score"], item["match_reasons"])

            st.markdown("---")
            st.subheader("Select expert")
            options = [
                f"{item['expert']['id']} - {item['expert']['name']}"
                for item in st.session_state["agency_ranked"]
            ]
            selected_label = st.radio(
                "Select one expert by ID",
                options=options,
                index=0 if options else None,
            )
            selected_id = selected_label.split(" - ")[0] if selected_label else None

            if st.button("Select expert & generate interview script", type="primary"):
                for item in st.session_state["agency_ranked"]:
                    if item["expert"]["id"] == selected_id:
                        st.session_state["selected_expert_id"] = selected_id
                        st.session_state["selected_expert"] = item["expert"]
                        st.session_state["selected_match_reasons"] = item["match_reasons"]
                        st.session_state["selected_criteria_signature"] = st.session_state[
                            "criteria_signature"
                        ]
                        _persist_case()
                        st.success(
                            f"Selected {item['expert']['name']}. Proceed to Tab 2."
                        )
                        break

            with st.expander("View all ranked experts"):
                search_term = st.text_input("Search by name, industry, or function")
                rows = []
                for item in st.session_state["agency_ranked"]:
                    expert = item["expert"]
                    haystack = " ".join(
                        [
                            expert["name"],
                            " ".join(expert["industryTags"]),
                            " ".join(expert["functionTags"]),
                        ]
                    ).lower()
                    if search_term and search_term.lower() not in haystack:
                        continue
                    rows.append(
                        {
                            "ID": expert["id"],
                            "Name": expert["name"],
                            "Headline": expert["headline"],
                            "Industry": ", ".join(expert["industryTags"]),
                            "Function": ", ".join(expert["functionTags"]),
                            "Level": expert["roleLevel"],
                            "Rate": expert["ratePerHour"],
                            "Score": item["score"],
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Recommended experts will appear after synthesis.")

with tab_script:
    st.subheader("Interview script and transcript")
    if not st.session_state.get("case_code"):
        st.info("Enter a case code in the Case Details tab to proceed.")
    elif not st.session_state["selected_expert"]:
        st.info("Select an expert in the Search + Agency tab to generate the script.")
    else:
        if st.session_state["criteria_signature"] != st.session_state.get(
            "selected_criteria_signature"
        ):
            _stale_warning(
                "Stale context: search criteria changed since expert selection. Regenerate recommended."
            )

        expert = st.session_state["selected_expert"]
        criteria = st.session_state["criteria"]

        st.markdown("### Context snapshot")
        st.markdown(
            f"- Industries: {', '.join(criteria.get('industries') or ['Any'])}\n"
            f"- Functions: {', '.join(criteria.get('functions') or ['Any'])}\n"
            f"- Levels: {', '.join(criteria.get('levels') or ['Any'])}\n"
            f"- Free text: {criteria.get('free_text') or 'None'}"
        )
        st.markdown(
            f"**Selected expert:** {expert['name']} - {expert['headline']}\n\n"
            f"- Credentials: {expert['credentials']['education']}; "
            f"Former: {', '.join(expert['credentials']['formerCompanies'])}\n"
            f"- Expertise: {expert['expertiseSummary']}\n"
            f"- Compliance: {', '.join(expert['complianceFlags'])}"
        )

        st.markdown("### Script generator")
        col1, col2, col3 = st.columns(3)
        with col1:
            length_minutes = st.selectbox("Interview length (min)", [30, 45, 60], index=2)
        with col2:
            tone = st.selectbox("Tone", ["Neutral", "Assertive", "Friendly"], index=0)
        with col3:
            depth = st.selectbox("Depth", ["High-level", "Deep-dive"], index=1)

        if st.button("Generate script"):
            script_text = None
            if _llm_enabled():
                system_prompt = "You are a consulting interviewer who drafts structured expert interview guides."
                user_prompt = (
                    "Create an interview script with the exact structure specified below. "
                    "Use a professional tone and keep it concise.\n\n"
                    "STRICT SCOPE: Only reference the industries/functions/levels provided. "
                    "Do not introduce other industries. If lists are empty, use 'General'.\n\n"
                    "Structure:\n"
                    "1) Title + Interview objective (3-5 bullets)\n"
                    "2) Filtering / Qualification (first 8-10 min) with 6-10 questions, red flags, follow-ups\n"
                    "3) Main Business Topics: 3-5 themes; each with rationale + 3-5 questions + probing follow-ups\n"
                    "4) Wrap-up: 3-5 questions + request for metrics/artifacts + recommended next experts + compliance reminder\n\n"
                    f"Criteria: {json.dumps(criteria, sort_keys=True)}\n"
                    f"Expert: {expert['name']} | {expert['headline']} | {expert['expertiseSummary']}\n"
                    f"Credentials: {expert['credentials']}\n"
                    f"Compliance: {expert['complianceFlags']}\n"
                    f"Length: {length_minutes} minutes\nTone: {tone}\nDepth: {depth}\n"
                )
                script_text = generate_text(system_prompt, user_prompt, max_tokens=900)
            if not script_text:
                script_text = generate_script(
                    criteria, expert, length_minutes, tone, depth, refine_text=""
                )
            st.session_state["script_text"] = script_text
            st.session_state["script_params"] = {
                "length_minutes": length_minutes,
                "tone": tone,
                "depth": depth,
                "refine_text": "",
                "llm": _llm_enabled(),
            }
            st.session_state["script_criteria_signature"] = st.session_state[
                "criteria_signature"
            ]
            st.success("Script generated.")
            _persist_case()

        refine_script = st.text_input("Refine script", placeholder="e.g., focus more on pricing")
        if st.button("Regenerate script"):
            script_text = None
            if _llm_enabled():
                system_prompt = "You are a consulting interviewer refining an expert interview guide."
                user_prompt = (
                    "Regenerate the script with the same structure, but apply the refinement request. "
                    "Keep the sections and make adjustments in emphasis.\n\n"
                    "STRICT SCOPE: Only reference the industries/functions/levels provided. "
                    "Do not introduce other industries.\n\n"
                    f"Refinement: {refine_script}\n"
                    f"Criteria: {json.dumps(criteria, sort_keys=True)}\n"
                    f"Expert: {expert['name']} | {expert['headline']} | {expert['expertiseSummary']}\n"
                    f"Length: {length_minutes} minutes\nTone: {tone}\nDepth: {depth}\n"
                )
                script_text = generate_text(system_prompt, user_prompt, max_tokens=900)
            if not script_text:
                script_text = generate_script(
                    criteria, expert, length_minutes, tone, depth, refine_text=refine_script
                )
            st.session_state["script_text"] = script_text
            st.session_state["script_params"] = {
                "length_minutes": length_minutes,
                "tone": tone,
                "depth": depth,
                "refine_text": refine_script,
                "llm": _llm_enabled(),
            }
            st.session_state["script_criteria_signature"] = st.session_state[
                "criteria_signature"
            ]
            if refine_script:
                st.session_state["refine_script_history"].append(refine_script)
            st.success("Script regenerated.")
            _persist_case()

        if st.session_state["script_text"]:
            st.text_area(
                "Current script",
                value=st.session_state["script_text"],
                height=400,
            )
        else:
            st.info("Generate a script to enable transcript creation.")

        st.markdown("### Mock transcript generator")
        refine_transcript = st.text_input(
            "Refine transcript",
            placeholder="e.g., shorter answers or more detail on go-to-market",
        )
        if st.button("Generate mock transcript"):
            transcript = None
            if _llm_enabled():
                system_prompt = "You are generating a mock expert interview transcript for consulting research."
                user_prompt = (
                    "Generate a realistic transcript that follows the script. "
                    "Include consultant questions, expert answers, occasional clarifications, "
                    "and a few illustrative numbers. Avoid confidential client info or MNPI.\n\n"
                    "STRICT SCOPE: Keep references aligned to the criteria industries/functions. "
                    "Do not introduce unrelated industries.\n\n"
                    f"Script:\n{st.session_state.get('script_text', '')}\n\n"
                    f"Expert: {expert['name']} | {expert['headline']}\n"
                    f"Compliance: {expert['complianceFlags']}\n"
                    f"Tone: {tone}\nDepth: {depth}\n"
                )
                transcript = generate_text(system_prompt, user_prompt, max_tokens=1100)
            if not transcript:
                transcript = generate_transcript(
                    criteria,
                    expert,
                    st.session_state.get("script_text", ""),
                    tone,
                    depth,
                    refine_text="",
                )
            st.session_state["transcript_text"] = transcript
            st.session_state["transcript_params"] = {
                "tone": tone,
                "depth": depth,
                "refine_text": "",
                "llm": _llm_enabled(),
            }
            st.session_state["transcript_criteria_signature"] = st.session_state[
                "criteria_signature"
            ]
            st.success("Transcript generated.")
            _persist_case()

        if st.button("Regenerate transcript"):
            transcript = None
            if _llm_enabled():
                system_prompt = "You are refining a mock expert interview transcript."
                user_prompt = (
                    "Regenerate the transcript based on the refinement request. "
                    "Keep compliance boundaries and include illustrative numbers.\n\n"
                    "STRICT SCOPE: Keep references aligned to the criteria industries/functions. "
                    "Do not introduce unrelated industries.\n\n"
                    f"Refinement: {refine_transcript}\n"
                    f"Script:\n{st.session_state.get('script_text', '')}\n\n"
                    f"Expert: {expert['name']} | {expert['headline']}\n"
                    f"Compliance: {expert['complianceFlags']}\n"
                    f"Tone: {tone}\nDepth: {depth}\n"
                )
                transcript = generate_text(system_prompt, user_prompt, max_tokens=1100)
            if not transcript:
                transcript = generate_transcript(
                    criteria,
                    expert,
                    st.session_state.get("script_text", ""),
                    tone,
                    depth,
                    refine_text=refine_transcript,
                )
            st.session_state["transcript_text"] = transcript
            st.session_state["transcript_params"] = {
                "tone": tone,
                "depth": depth,
                "refine_text": refine_transcript,
                "llm": _llm_enabled(),
            }
            st.session_state["transcript_criteria_signature"] = st.session_state[
                "criteria_signature"
            ]
            if refine_transcript:
                st.session_state["refine_transcript_history"].append(refine_transcript)
            st.success("Transcript regenerated.")
            _persist_case()

        if st.session_state["transcript_text"]:
            st.text_area(
                "Current transcript",
                value=st.session_state["transcript_text"],
                height=400,
            )
            st.markdown("### Interview feedback")
            rating = st.slider(
                "Score the interview quality (0-10)",
                min_value=0,
                max_value=10,
                value=st.session_state.get("interview_feedback") or 5,
            )
            if st.button("Save feedback"):
                st.session_state["interview_feedback"] = rating
                _persist_case()
                st.success("Interview feedback saved.")

with tab_summary:
    st.subheader("Summary + Save")
    if not st.session_state.get("case_code"):
        st.info("Enter a case code in the Case Details tab to proceed.")
    else:
        if st.session_state["criteria_signature"] != st.session_state.get(
            "script_criteria_signature"
        ):
            if st.session_state.get("script_text"):
                _stale_warning(
                    "Stale context: script generated with older criteria. Regenerate recommended."
                )
        if st.session_state["criteria_signature"] != st.session_state.get(
            "transcript_criteria_signature"
        ):
            if st.session_state.get("transcript_text"):
                _stale_warning(
                    "Stale context: transcript generated with older criteria. Regenerate recommended."
                )

        with st.expander("Current Script"):
            st.text_area(
                "Script text",
                value=st.session_state.get("script_text") or "",
                height=300,
            )
        with st.expander("Current Transcript"):
            st.text_area(
                "Transcript text",
                value=st.session_state.get("transcript_text") or "",
                height=300,
            )

        if st.button("Generate summary"):
            if not st.session_state.get("transcript_text"):
                st.error("Generate a transcript before summarizing.")
            else:
                summary = summarize_interview(
                    st.session_state["criteria"],
                    st.session_state["selected_expert"],
                    st.session_state.get("script_text", ""),
                    st.session_state.get("transcript_text", ""),
                )
                metrics = summary["metrics"]
                if metrics:
                    metrics_df = pd.DataFrame(metrics)
                else:
                    metrics_df = pd.DataFrame(
                        [{"value": "None", "context": "No metrics detected."}]
                    )
                summary_markdown = None
                if _llm_enabled():
                    system_prompt = "You are a consulting analyst summarizing expert interviews."
                    user_prompt = (
                        "Summarize the interview with the following sections:\n"
                        "1) Executive summary (5-8 bullets)\n"
                        "2) Expert credibility & relevance (bullets + short paragraph)\n"
                        "3) Key insights by theme (3-6 themes, each with bullets)\n"
                        "4) Open questions / uncertainties (bullets)\n"
                        "5) Suggested next steps (bullets)\n\n"
                        "Keep it concise and professional.\n\n"
                        f"Criteria: {json.dumps(st.session_state['criteria'], sort_keys=True)}\n"
                        f"Expert: {st.session_state['selected_expert']}\n"
                        f"Script:\n{st.session_state.get('script_text', '')}\n\n"
                        f"Transcript:\n{st.session_state.get('transcript_text', '')}\n"
                    )
                    summary_markdown = generate_text(
                        system_prompt, user_prompt, max_tokens=900
                    )
                if not summary_markdown:
                    summary_markdown = _build_summary_markdown(summary)
                summary_markdown += "\n\n## Metrics / Numbers Mentioned\n"
                summary_markdown += "| Value | Context |\n| --- | --- |\n"
                for row in metrics_df.to_dict(orient="records"):
                    value = str(row.get("value", "")).replace("|", " ")
                    context = str(row.get("context", "")).replace("|", " ")
                    summary_markdown += f"| {value} | {context} |\n"
                st.session_state["summary_text"] = summary_markdown
                st.session_state["summary_data"] = summary
                st.session_state["summary_data"]["metrics_df"] = metrics_df.to_dict(
                    orient="records"
                )
                st.success("Summary generated.")
                _persist_case()

    if st.session_state.get("case_code"):
        if st.session_state.get("summary_text"):
            st.markdown(st.session_state["summary_text"])
            st.markdown("## Metrics / Numbers Mentioned")
            metrics_rows = st.session_state["summary_data"].get("metrics_df", [])
            st.table(pd.DataFrame(metrics_rows))

            st.markdown("## Tags to Store")
            tags = st.session_state["summary_data"]["tags"]
            st.markdown(
                f"- Industry: {', '.join(tags.get('industries', []))}\n"
                f"- Function: {', '.join(tags.get('functions', []))}\n"
                f"- Level: {', '.join(tags.get('levels', []))}\n"
                f"- Topics: {', '.join(tags.get('topics', []))}"
            )

        if st.button("Add to ET database"):
            if not st.session_state.get("summary_text"):
                st.error("Generate a summary before saving.")
            else:
                payload = {
                    "industries": st.session_state["criteria"].get("industries", []),
                    "functions": st.session_state["criteria"].get("functions", []),
                    "levels": st.session_state["criteria"].get("levels", []),
                    "free_text": st.session_state["criteria"].get("free_text", ""),
                    "expert_id": st.session_state["selected_expert"]["id"],
                    "expert_name": st.session_state["selected_expert"]["name"],
                    "script_text": st.session_state.get("script_text", ""),
                    "transcript_text": st.session_state.get("transcript_text", ""),
                    "summary_text": st.session_state.get("summary_text", ""),
                    "tags": st.session_state["summary_data"]["tags"],
                    "interview_rating": st.session_state.get("interview_feedback"),
                }
                save_interview(payload)
                st.success("Saved interview summary.")
                _persist_case()

        st.markdown("### Saved interviews")
        recent = load_recent(limit=10)
        if recent:
            options = [
                f"{item['id']} | {item['created_at']} | {item['expert_name']}"
                for item in recent
            ]
            selected = st.selectbox("Select saved interview", options)
            selected_id = int(selected.split("|")[0].strip())
            record = load_by_id(selected_id)
            if record:
                st.markdown(
                    f"**{record['expert_name']}** | {record['created_at']}\n\n"
                    f"- Industries: {', '.join(record['industries'])}\n"
                    f"- Functions: {', '.join(record['functions'])}\n"
                    f"- Levels: {', '.join(record['levels'])}"
                )
                st.text_area("Stored summary", value=record["summary_text"], height=300)
        else:
            st.info("No saved interviews yet.")
