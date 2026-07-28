"""
app.py
------
Web interface for the Pakistan case-law assistant, with optional voice input.

Reuses the same vector store, embedding model, and Gemini labeling from
build_index.py / search.py / analyze.py.

Run it with:
    streamlit run app.py

Voice input needs one extra package:
    pip install streamlit-mic-recorder
The record button transcribes your speech with Gemini and drops it into the
question box. If the package is not installed, the app still works by typing.
"""

import os
import re
import json
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from google.genai import types

# voice recorder is optional; the app still runs without it
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC = True
except Exception:
    HAS_MIC = False

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DB_PATH     = "./chroma_db"
COLLECTION  = "pk_sc_judgments"
PREFERRED   = ("gemini-flash-latest", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.0-flash")

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

SUPPORT_STYLE = {
    "defence":     ("DEFENCE",     "#111111", "#111111"),
    "prosecution": ("PROSECUTION", "#ffffff", "#111111"),
    "mixed":       ("MIXED",       "#666666", "#666666"),
    "unclear":     ("UNCLEAR",     "#ffffff", "#999999"),
}


# ---------- cached resources ----------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION)


def _read_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_genai():
    key = _read_key()
    if not key:
        return None, None
    client = genai.Client(api_key=key)
    try:
        names = [m.name.split("/")[-1] for m in client.models.list()]
    except Exception:
        names = []
    model = next((p for p in PREFERRED if p in names), None)
    if not model:
        model = next((n for n in names if "flash" in n), PREFERRED[0])
    return client, model


# ---------- core logic ----------
def retrieve(query, top_cases=5, pool=25):
    embedder = get_embedder()
    col = get_collection()
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=pool)

    best = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        ref = meta["case_reference"]
        if ref not in best or dist < best[ref]["dist"]:
            best[ref] = {"dist": dist, "snippet": doc, "source": meta["source_id"], "reference": ref}
    return sorted(best.values(), key=lambda x: x["dist"])[:top_cases]


def _parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def label(reference, passage):
    client, model = get_genai()
    if not client:
        return None
    try:
        resp = client.models.generate_content(
            model=model, contents=PROMPT.format(reference=reference, passage=passage)
        )
        return _parse_json(resp.text)
    except Exception as e:
        return {"summary": f"(could not analyze: {e})", "supports": "unclear", "reason": ""}


def transcribe(audio_bytes):
    """Send recorded audio to Gemini and get back the text of what was said."""
    client, model = get_genai()
    if not client:
        return None
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[
                "Transcribe this audio to plain text. Return ONLY the words spoken, nothing else.",
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            ],
        )
        return (resp.text or "").strip()
    except Exception as e:
        st.warning(f"Could not transcribe audio: {e}")
        return None


def badge(supports):
    text, bg, border = SUPPORT_STYLE.get(supports, SUPPORT_STYLE["unclear"])
    fg = "#ffffff" if bg != "#ffffff" else "#111111"
    return (
        f'<span style="display:inline-block;padding:2px 10px;border:1px solid {border};'
        f'background:{bg};color:{fg};font-size:0.72rem;letter-spacing:0.08em;'
        f'font-weight:600;border-radius:2px;">{text}</span>'
    )


# ---------- page + styling ----------
st.set_page_config(page_title="Pakistan Case-Law Assistant", page_icon="§", layout="centered")

st.markdown(
    """
    <style>
      html, body, [class*="css"] { color: #111111; }
      .stApp { background: #ffffff; }
      h1, h2, h3, h4 { font-family: Georgia, 'Times New Roman', serif !important; color:#111111; }
      .lawtitle { font-family: Georgia, serif; font-size: 2.1rem; font-weight:700;
                  letter-spacing:-0.5px; margin-bottom:0.1rem; }
      .lawrule { border:none; border-top:2px solid #111111; margin:0.4rem 0 1.1rem 0; }
      .lawsub { color:#555; font-size:0.95rem; margin-bottom:1.4rem; }
      .caseref { font-family: Georgia, serif; font-size:1.05rem; font-weight:700; }
      .rel { color:#777; font-size:0.8rem; }
      .whytext { color:#555; font-size:0.85rem; }
      .srcfile { color:#999; font-size:0.75rem; }
      .stButton>button { background:#111111; color:#ffffff; border:1px solid #111111;
                         border-radius:2px; font-weight:600; }
      .stButton>button:hover { background:#ffffff; color:#111111; border:1px solid #111111; }
      [data-testid="stSidebar"] { background:#fafafa; border-right:1px solid #e5e5e5; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="lawtitle">§ &nbsp;Pakistan Case-Law Assistant</div>', unsafe_allow_html=True)
st.markdown('<hr class="lawrule">', unsafe_allow_html=True)
st.markdown(
    '<div class="lawsub">Ask a legal question by typing or voice and retrieve relevant '
    'Supreme Court of Pakistan judgments, each summarised and labelled by the side it '
    'appears to support.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### About")
    st.write(
        "Search real Supreme Court of Pakistan judgments by meaning rather than keywords. "
        "Each result is summarised and labelled DEFENCE, PROSECUTION, MIXED, or UNCLEAR."
    )
    _, model_name = get_genai()
    st.markdown("---")
    st.write("**AI labelling:** " + ("on" if model_name else "off (search still works)"))
    st.write("**Voice input:** " + ("available" if (HAS_MIC and model_name) else "off"))
    st.markdown("---")
    st.caption(
        "Research aid, not legal advice. Labels are judged from a short excerpt only. "
        "Always read the full judgment before relying on it."
    )

# keep the question text in session so voice can fill it in
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

# ----- voice input (optional) -----
if HAS_MIC and model_name:
    st.write("Speak your question:")
    audio = mic_recorder(start_prompt="🎤 Record", stop_prompt="⏹ Stop", format="wav", key="mic")
    if audio and audio.get("id") != st.session_state.get("last_audio_id"):
        st.session_state.last_audio_id = audio.get("id")
        with st.spinner("Transcribing your question..."):
            spoken = transcribe(audio["bytes"])
        if spoken:
            st.session_state.query_text = spoken
            st.rerun()
elif not HAS_MIC:
    st.caption("Tip: install streamlit-mic-recorder to enable voice input.")

# ----- question + controls -----
query = st.text_input("Legal question", key="query_text",
                      placeholder="e.g. self defence in a murder trial")
c1, c2 = st.columns([2, 1])
with c1:
    side = st.radio("Arguing for the", ["Either side", "Defence", "Prosecution"], horizontal=True)
with c2:
    n = st.slider("Results", 3, 8, 5)

go = st.button("Search judgments", use_container_width=True)

if go and query.strip():
    hint = "" if side == "Either side" else f" (for the {side.lower()} side)"
    with st.spinner("Searching and analysing judgments..."):
        results = retrieve(query.strip() + hint, top_cases=n)
        analyses = [label(r["reference"], r["snippet"]) for r in results]

    if not results:
        st.error("No results. Make sure the index has been built.")
    else:
        st.markdown(f"#### {len(results)} judgments found")
        for r, a in zip(results, analyses):
            with st.container(border=True):
                rel = 1 - r["dist"]
                top = f'<span class="caseref">{r["reference"]}</span> &nbsp; <span class="rel">relevance {rel:.2f}</span>'
                if a:
                    top += " &nbsp; " + badge(a.get("supports", "unclear").lower())
                st.markdown(top, unsafe_allow_html=True)
                if a:
                    st.write(a.get("summary", "").strip())
                    if a.get("reason"):
                        st.markdown(f'<div class="whytext">Why: {a["reason"].strip()}</div>', unsafe_allow_html=True)
                else:
                    st.write(r["snippet"][:400] + "...")
                    st.caption("AI labelling is off. Add a Gemini API key to enable summaries.")
                st.markdown(f'<div class="srcfile">Source: {r["source"]}</div>', unsafe_allow_html=True)
elif go:
    st.info("Type or record a question first.")
