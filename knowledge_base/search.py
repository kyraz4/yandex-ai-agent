#!/usr/bin/env python3
"""
Простой CLI для поиска по FAISS-индексу. Используется для проверки
качества ретрива в Задании 3 и как ядро RAG-бота в следующих заданиях.
"""

import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("vector_index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load():
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    metadata: list[dict] = []
    with (INDEX_DIR / "metadata.jsonl").open(encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    model = SentenceTransformer(MODEL_NAME)
    return index, metadata, model


def search(query: str, k: int = 3):
    index, metadata, model = load()
    q = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    scores, ids = index.search(q, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        m = metadata[idx]
        results.append({
            "score": float(score),
            "chunk_id": m["chunk_id"],
            "source": m["source"],
            "text": m["text"],
        })
    return results


if __name__ == "__main__":
    test_queries = [
        # общие вопросы по миру (с вымышленными терминами из Задания 2)
        "Who leads the human resistance against the automatons?",
        "What is the Simulacrum used for?",
        "Where do the free humans live after escaping the Lattice?",
        # контрольные запросы с оригинальными терминами — индекс не должен
        # «находить» их, потому что в текстах этих слов нет
        "Who is Neo?",
        "What is Zion?",
    ]
    for q in test_queries:
        print(f"\n🔍 {q}")
        for r in search(q):
            snippet = r["text"][:180].replace("\n", " ")
            print(f"  [{r['score']:.3f}] {r['chunk_id']:30} ← {r['source']}")
            print(f"     {snippet}…")
