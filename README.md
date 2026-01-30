# Expert Tool (ET) - Streamlit Mockup

Expert Tool (ET) is a single-page Streamlit mockup for consulting teams to search experts, generate interview scripts and transcripts, and store summaries for reuse. The app uses local templates, deterministic randomness, and SQLite for persistence. LLM generation is optional via OpenAI.
Expert Tool now includes a Case Details tab to load/save case context by case code, and simulated agency workflows.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Notes

- Data is mock and stored in `et_data.py`.
- SQLite file is created locally as `et_interviews.db`.
- To enable LLM generation, set `OPENAI_API_KEY` before running.
- Agency email template lives in `email_template.md`.
