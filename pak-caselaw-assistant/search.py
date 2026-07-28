"""
search.py
---------
Step 2 of the legal-search project.

Takes a legal question, finds the most relevant judgments in the vector store
built by build_index.py, and prints each one with its case reference and the
most relevant passage.

    python search.py "self defence in a murder trial"
    python search.py            (then type your question when prompted)

This is retrieval only. The next phase adds an LLM step that summarizes each
judgment and labels which side it supports.
"""

import sys
from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # must match build_index.py
DB_PATH    = "./chroma_db"
COLLECTION = "pk_sc_judgments"

TOP_CASES  = 5    # how many distinct judgments to show
POOL       = 25   # how many chunks to pull before grouping into cases


def search(query: str, top_cases=TOP_CASES, pool=POOL):
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_collection(COLLECTION)

    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=pool)

    # A judgment can appear in several chunks; keep only its single best-matching chunk.
    best = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        ref = meta["case_reference"]
        if ref not in best or dist < best[ref]["dist"]:
            best[ref] = {"dist": dist, "snippet": doc, "source": meta["source_id"], "reference": ref}

    ranked = sorted(best.values(), key=lambda x: x["dist"])[:top_cases]
    return ranked


def main():
    query = " ".join(sys.argv[1:]).strip() or input("Enter a legal question: ").strip()
    if not query:
        print("No question given.")
        return

    print(f"\nQuery: {query}\n" + "=" * 72)
    results = search(query)
    if not results:
        print("No results. Did you run build_index.py first?")
        return

    for i, r in enumerate(results, 1):
        similarity = 1 - r["dist"]              # cosine distance -> similarity
        snippet = r["snippet"][:400].replace("\n", " ")
        print(f"\n[{i}] {r['reference']}   (relevance {similarity:.2f})")
        print(f"    source file: {r['source']}")
        print(f"    passage: {snippet}...")

    print("\n" + "=" * 72)
    print("Reminder: this is a research aid, not legal advice. Always read the full judgment.")


if __name__ == "__main__":
    main()
