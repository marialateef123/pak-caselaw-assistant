"""
build_index.py
--------------
Step 1 of the legal-search project.

Loads the public Supreme Court of Pakistan judgments dataset, extracts a clean
case reference from each judgment, splits each judgment into overlapping chunks,
embeds the chunks, and stores everything in a local Chroma vector database.

Run this ONCE (or whenever you want to rebuild). It creates a ./chroma_db folder
that search.py then reads from.

    python build_index.py
"""

import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb

# ------------------------- settings you can tweak -------------------------
DATASET      = "Ibtehaj10/supreme-court-of-pak-judgments"
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"  # small + fast; good first choice
DB_PATH      = "./chroma_db"
COLLECTION   = "pk_sc_judgments"
MAX_DOCS     = 300     # start small so the first run is quick. Set to None to index all 1,414.
CHUNK_WORDS  = 250     # size of each text chunk, in words
CHUNK_OVERLAP = 50     # overlap between consecutive chunks, so context is not cut mid-thought
EMBED_BATCH  = 256
# --------------------------------------------------------------------------


# ---- case-reference extraction (tested against real dataset samples) ----
CASE_TYPES = (
    r"Civil Appeal|Criminal Appeal|Civil Petition|Criminal Petition|"
    r"Jail Petition|Constitution Petition|Human Rights Case|Suo Motu Case|"
    r"Civil Review Petition|Criminal Review Petition|Criminal Original|Civil Misc"
)
TEXT_RE = re.compile(
    rf"({CASE_TYPES})\s*Nos?\.?\s*([\dA-Z\-]+)\s*of\s*((?:19|20)\d\d)",
    re.IGNORECASE,
)
ABBR = {
    "C.A": "Civil Appeal", "Crl.A": "Criminal Appeal", "C.P": "Civil Petition",
    "Crl.P": "Criminal Petition", "J.P": "Jail Petition", "Const.P": "Constitution Petition",
}
FILE_RE = re.compile(r"([A-Za-z.]+?)\.?(\d+)_((?:19|20)\d\d)\.pdf", re.IGNORECASE)


def extract_reference(text: str, filename: str) -> str:
    """Prefer the case number stated in the judgment text; fall back to the filename."""
    m = TEXT_RE.search(text or "")
    if m:
        return f"{m.group(1).title()} No.{m.group(2)} of {m.group(3)} (SC)"
    fm = FILE_RE.search(filename or "")
    if fm:
        abbr = fm.group(1).rstrip(".")
        name = ABBR.get(abbr, abbr)
        return f"{name} No.{fm.group(2)} of {fm.group(3)} (SC)"
    return "Reference not found (verify manually)"


def get_filename(case_details) -> str:
    """case_details may arrive as a dict or as its string form; handle both."""
    if isinstance(case_details, dict):
        return case_details.get("id", "")
    if isinstance(case_details, str):
        m = re.search(r"'id':\s*'([^']+)'", case_details)
        return m.group(1) if m else ""
    return ""


def chunk_text(text: str, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = (text or "").split()
    if not words:
        return []
    chunks, step = [], size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size]).strip()
        if chunk:
            chunks.append(chunk)
        if i + size >= len(words):
            break
    return chunks


def main():
    print("Loading dataset (first run downloads ~25 MB)...")
    ds = load_dataset(DATASET, split="train")
    if MAX_DOCS:
        ds = ds.select(range(min(MAX_DOCS, len(ds))))
    print(f"  {len(ds)} judgments selected.")

    print(f"Loading embedding model '{MODEL_NAME}' (first run downloads it)...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_PATH)
    # start clean each build
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    for row_i, row in enumerate(ds):
        text = row["text"] or ""
        filename = get_filename(row.get("case_details"))
        reference = extract_reference(text, filename)
        for ci, chunk in enumerate(chunk_text(text)):
            ids.append(f"{row_i}_{ci}")
            docs.append(chunk)
            metas.append({
                "case_reference": reference,
                "source_id": filename,
                "chunk_index": ci,
            })

    print(f"  {len(docs)} chunks created. Embedding and indexing (the slow part)...")
    for start in range(0, len(docs), EMBED_BATCH):
        batch = docs[start:start + EMBED_BATCH]
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False).tolist()
        col.add(
            ids=ids[start:start + EMBED_BATCH],
            embeddings=embeddings,
            documents=batch,
            metadatas=metas[start:start + EMBED_BATCH],
        )
        print(f"    indexed {min(start + EMBED_BATCH, len(docs))}/{len(docs)}")

    print(f"\nDone. Vector store saved to {DB_PATH}")
    print("Next: run  python search.py \"your legal question here\"")


if __name__ == "__main__":
    main()
