#!/usr/bin/env python3
"""Строит FAISS-индекс из обезличенных документов в prepared_data/."""

import json
import time
from pathlib import Path

import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

PREPARED_DIR = Path("prepared_data")
OUTPUT_DIR = Path("vector_index")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_documents():
    docs = []
    for path in sorted(PREPARED_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append({
            "title": path.stem,
            "source": path.name,
            "text": text,
        })
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{doc['title']}__{i:03d}",
                "chunk_index": i,
                "total_chunks": len(pieces),
                "title": doc["title"],
                "source": doc["source"],
                "text": piece,
            })
    return chunks


def main():
    t0 = time.time()

    print("1/4 · Загрузка документов…")
    docs = load_documents()
    if not docs:
        raise SystemExit("❌ В prepared_data/ нет .txt файлов.")
    print(f"    документов: {len(docs)}")

    print("2/4 · Чанкинг…")
    chunks = chunk_documents(docs)
    print(f"    чанков: {len(chunks)}")

    print(f"3/4 · Загрузка модели {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"    размерность эмбеддинга: {dim}")

    print("4/4 · Генерация эмбеддингов и построение индекса…")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(OUTPUT_DIR / "faiss.index"))

    with (OUTPUT_DIR / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"\n✅ Готово за {elapsed:.1f} с")
    print(f"   → {OUTPUT_DIR / 'faiss.index'}")
    print(f"   → {OUTPUT_DIR / 'metadata.jsonl'}  ({len(chunks)} чанков, {len(docs)} документов)")


if __name__ == "__main__":
    main()
