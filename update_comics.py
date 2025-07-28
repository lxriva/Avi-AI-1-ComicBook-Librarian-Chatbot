import os
from datetime import datetime

# update_comics.py
from fetch_comicvine_volumes import fetch_volumes
import json
from datetime import datetime

def update_dc_marvel():
    data = fetch_volumes(pages=5)  # Adjust to only fetch recent pages
    filtered = [v for v in data if str(v.get("publisher", "")).lower() in ["dc comics", "marvel"]]

    os.makedirs("updates", exist_ok=True)

    filename = f"updates/update_{datetime.now().strftime('%Y_%m')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(filtered)} updates to {filename}")

    # After saving to updates/
    with open("comic_corpus.json", "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print("Updated comic_corpus.json with latest DC/Marvel comics")

if __name__ == "__main__":
    update_dc_marvel()
