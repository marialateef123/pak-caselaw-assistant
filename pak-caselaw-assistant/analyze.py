"""
analyze.py
----------
Step 3: the intelligence layer.

Takes a legal question, finds the most relevant judgments (reusing search.py),
then for each one asks Gemini to produce a plain-language summary and label
whether the excerpt supports the defence or the prosecution, with the case
reference attached.

    python analyze.py "self defence in a murder trial"
    python analyze.py "self defence in a murder trial" defence

The optional second word (defence / prosecution) just sharpens the search.
The side label for each case is judged objectively from the excerpt.

Your Gemini key is read from the .env file and never appears in this code.
"""

import os
import re
import sys
import json
from dotenv import load_dotenv
from google import genai

from search import search  # reuse the retrieval you already built

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("No GEMINI_API_KEY found. Put it in a .env file as GEMINI_API_KEY=your_key")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

# Model names change often. This picks a working Gemini flash model automatically.
PREFERRED = ("gemini-flash-latest", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.0-flash")


def resolve_model():
    try:
        names = [m.name.split("/")[-1] for m in client.models.list()]
    except Exception:
        return PREFERRED[0]
    for p in PREFERRED:
        if p in names:
            return p
    for n in names:               # fall back to any flash model available
        if "flash" in n:
            return n
    return PREFERRED[0]


MODEL = resolve_model()

PROMPT = """You are helping a lawyer research Pakistani case law.
Below is an excerpt from a Supreme Court of Pakistan judgment.

Case reference: {reference}
Excerpt:
\"\"\"{passage}\"\"\"

Based ONLY on this excerpt, reply with a JSON object and nothing else:
{{
  "summary": "one or two plain-language sentences on what this excerpt is about",
  "supports": "defence" | "prosecution" | "mixed" | "unclear",
  "reason": "one short sentence explaining the supports value"
}}
Use "unclear" if the excerpt does not clearly favour either side. Do not invent facts."""


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def label_judgment(reference, passage):
    prompt = PROMPT.format(reference=reference, passage=passage)
    try:
        resp = client.models.generate_content(model=MODEL, contents=prompt)
        return parse_json(resp.text)
    except Exception as e:
        return {"summary": f"(could not analyze: {e})", "supports": "unclear", "reason": ""}


def main():
    args = sys.argv[1:]
    side = None
    if args and args[-1].lower() in {"defence", "defense", "prosecution"}:
        side = args[-1].lower()
        args = args[:-1]
    query = " ".join(args).strip() or input("Enter a legal question: ").strip()
    if not query:
        print("No question given.")
        return

    full_query = f"{query} (for the {side} side)" if side else query
    print(f"\nQuery: {query}" + (f"   [side: {side}]" if side else ""))
    print(f"Model: {MODEL}\n" + "=" * 72)

    results = search(full_query)
    if not results:
        print("No results. Did you run build_index.py first?")
        return

    for i, r in enumerate(results, 1):
        a = label_judgment(r["reference"], r["snippet"])
        similarity = 1 - r["dist"]
        print(f"\n[{i}] {r['reference']}   (relevance {similarity:.2f})")
        print(f"    Supports : {a.get('supports', 'unclear').upper()}")
        print(f"    Summary  : {a.get('summary', '').strip()}")
        if a.get("reason"):
            print(f"    Why      : {a['reason'].strip()}")
        print(f"    Source   : {r['source']}")

    print("\n" + "=" * 72)
    print("The 'Supports' label is judged from a short excerpt only. It is a")
    print("research aid, not legal advice. Always read the full judgment before relying on it.")


if __name__ == "__main__":
    main()
