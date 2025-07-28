import json
import os
import dotenv  # ✅ NEW

dotenv.load_dotenv()  # ✅ NEW

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


docs = [Document(page_content="Batman is a vigilante in Gotham."),
        Document(page_content="Superman is the last son of Krypton, and the protector of Metropolis.")]


def parse_volumes(input_file="comic_corpus.json", output_file="parsed_comics.txt"):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    for vol in data:
        name = vol.get("name", "Unknown Title")
        publisher = vol.get("publisher", "Unknown Publisher")
        issues = vol.get("count_of_issues", "?")
        year = vol.get("start_year", "?")
        desc = vol.get("description", "No description available.").strip()

        text_chunk = f"""
Title: {name}
Publisher: {publisher}
Year: {year}
Issues: {issues}
Description: {desc}
"""
        chunks.append(text_chunk.strip())

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(chunks))

    print(f"Parsed {len(chunks)} volumes to {output_file}")


if __name__ == "__main__":
    parse_volumes()

# Embed and store in FAISS
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding)
vectorstore.save_local("comicvine_index")
print("FAISS index saved to comicvine_index/")
