import requests
import time
import json

API_KEY = "12a10514a29ee249fdcde1c533ac7347c4833727"
API_URL = "https://comicvine.gamespot.com/api/volumes/"
PROXY = "https://api.allorigins.win/raw?url="

HEADERS = {
    "User-Agent": "ComicShelfBot/1.0"
}


def fetch_volumes(pages=5, limit=100):
    all_volumes = []
    for page in range(pages):
        offset = page * limit
        url = f"{API_URL}?api_key={API_KEY}&format=json&limit={limit}&offset={offset}"
        proxy_url = url  # Direct API call at home


        print(f"Fetching page {page + 1} from: {proxy_url}")
        res = requests.get(proxy_url, headers=HEADERS)
        if res.status_code != 200:
            print("Failed to fetch:", res.status_code, res.text)
            break

        data = res.json()
        results = data.get("results", [])
        for vol in results:
            publisher = (vol.get("publisher") or {}).get("name", "").lower()
            if publisher not in ["dc comics", "marvel"]:
                continue  # skip non-DC/Marvel books

            all_volumes.append({
                "id": vol.get("id"),
                "name": vol.get("name"),
                "description": (vol.get("description") or "").strip(),
                "count_of_issues": vol.get("count_of_issues"),
                "publisher": publisher.title(),
                "start_year": vol.get("start_year"),
                "image_url": vol.get("image", {}).get("original_url"),
                "api_detail_url": vol.get("api_detail_url")
            })


        time.sleep(1)  # be nice to the proxy/api

    return all_volumes


if __name__ == "__main__":
    print("Starting fetch...")
    data = fetch_volumes(pages=120)  # fetch 12,000 volumes
    with open("comic_corpus.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} volumes to comic_corpus.json")
