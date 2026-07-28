# Pakistan Case-Law Assistant

An AI research assistant for Pakistani case law. Ask a legal question by typing or by voice and get back relevant Supreme Court of Pakistan judgments, each summarised and labelled by the side it appears to support, with the case reference attached.

Built as a working prototype to explore Retrieval-Augmented Generation (RAG) on real legal documents.

![App with voice input](screenshots/01-app-search.jpg)

<!-- DEMO VIDEO: edit this README on github.com and drag your .mp4 here, or add a demo.gif -->

## The problem

Legal research tools that let a lawyer search case law are expensive, and the paid ones still make you read each judgment in full to find out which side it actually favours. This project started from a real conversation with a law student, who explained that her biggest pain was not finding cases but discovering, only after reading a whole judgment, that a case she thought supported her actually cut against her.

So this tool does two things a plain search box does not: it summarises each judgment in plain language, and it labels whether the judgment appears to support the **defence**, the **prosecution**, both (**mixed**), or neither (**unclear**), so a lawyer can triage what is worth reading in full.

## Features

- Semantic search over real Supreme Court of Pakistan judgments (search by meaning, not keywords)
- **Voice input**: ask your question by speaking, transcribed with Gemini
- AI-generated plain-language summary of each result
- Side-labelling: defence / prosecution / mixed / unclear, with a one-line reason
- Case reference extracted from the judgment text (not the unreliable dataset citation field)
- Clean, black-and-white professional interface, plus a command-line version
- Honest "unclear" labelling when a short excerpt does not clearly favour a side

![Result cards](screenshots/02-app-results.jpg)

## How it works

1. **Ingestion** loads a public dataset of Supreme Court judgments and extracts a clean case reference from each one.
2. **Retrieval (RAG)** splits judgments into chunks, embeds them with a sentence-transformer model, and stores them in a Chroma vector database. A question is embedded and matched against the store to pull the most relevant passages.
3. **Reasoning** sends each retrieved judgment to a Google Gemini model, which returns the summary and the side-label. Citations come from the extracted reference, never from the model, to keep them verifiable.
4. **Voice** records spoken questions in the browser and transcribes them with Gemini, then runs the same pipeline.
5. **Interface** is a Streamlit web app presenting it all as ranked, labelled result cards.

## Tech stack

| Area | Tools |
|------|-------|
| Language | Python |
| Embeddings | sentence-transformers (BGE) |
| Vector database | Chroma |
| LLM + transcription | Google Gemini (google-genai SDK) |
| Voice capture | streamlit-mic-recorder |
| Web UI | Streamlit |
| Data | Supreme Court of Pakistan judgments (Hugging Face, MIT licensed) |

## Running it locally

Requires Python 3.10+.

```bash
# 1. set up
python -m venv venv
venv\Scripts\activate        # Windows  (use: source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt

# 2. add your free Gemini API key
#    copy .env.example to a new file named .env and paste your key
#    get one at https://aistudio.google.com/apikey

# 3. build the search index (one time)
python build_index.py

# 4a. run the web app (with voice)
streamlit run app.py

# 4b. or use the command line
python analyze.py "benefit of doubt acquittal in a murder case" defence
```

## Command-line version

The same pipeline runs in the terminal, useful for quick tests:

![CLI output](screenshots/03-cli-output.jpg)

## Honest limitations

This is a prototype, not a production legal tool.

- It currently indexes a subset of Supreme Court judgments, so coverage is partial and some queries return closer matches than others.
- Retrieval matches on semantic similarity, so it can occasionally surface a case that shares vocabulary but not the exact legal point. A reranking step is the planned next improvement.
- Side-labels are judged from a short retrieved excerpt, not the full judgment, so they are a triage aid, not a legal conclusion.
- It covers the Supreme Court only. A practising lawyer also needs High Court judgments.
- Uses the Gemini free tier, which is rate-limited to a small number of requests per day.

**This is a research aid, not legal advice. Always read the full judgment before relying on it.**

## Data source

Supreme Court of Pakistan judgments, `Ibtehaj10/supreme-court-of-pak-judgments` on Hugging Face (MIT licensed). Judgments are public record; the tool generates its own summaries rather than reusing any commercial legal database's editorial content.

## Roadmap

- Add a reranking step to improve retrieval precision
- Index the full dataset and add High Court judgments
- Deploy to a public URL
- Larger free LLM quota (e.g. via Groq) to remove the daily request limit
