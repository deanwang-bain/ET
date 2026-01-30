import json
import os

from openai import OpenAI


DEFAULT_MODEL = "gpt-5.2"


def is_available():
    return bool(os.getenv("OPENAI_API_KEY"))


def _client():
    return OpenAI()


def generate_text(system_prompt, user_prompt, model=DEFAULT_MODEL, temperature=0.7, max_tokens=900):
    if not is_available():
        return None
    client = _client()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    text = response.output_text
    return text.strip() if text else None


def generate_json(system_prompt, user_prompt, model=DEFAULT_MODEL, temperature=0.4, max_tokens=900):
    if not is_available():
        return None
    client = _client()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    text = response.output_text
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
