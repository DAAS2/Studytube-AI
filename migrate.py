# rebuild_index.py
import os, json
from pytubefix import YouTube

SHARED_DIR = "video_library/_shared"
index = {}

for folder in os.listdir(SHARED_DIR):
    if not folder.startswith("faiss_"):
        continue
    video_id = folder.replace("faiss_", "")
    print(f"🔍 Fetching metadata for {video_id}...")
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        index[video_id] = {
            "video_id":  video_id,
            "url":       f"https://www.youtube.com/watch?v={video_id}",
            "title":     yt.title or "Untitled",
            "author":    yt.author or "Unknown",
            "thumbnail": yt.thumbnail_url or f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            "duration":  yt.length or 0,
            "chunks":    0,   # unknown, harmless
            "added_at":  "migrated",
        }
        print(f"  ✅ {yt.title}")
    except Exception as e:
        print(f"  ⚠️  Failed: {e} — adding stub entry")
        index[video_id] = {
            "video_id":  video_id,
            "url":       f"https://www.youtube.com/watch?v={video_id}",
            "title":     video_id,
            "author":    "Unknown",
            "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            "duration":  0, "chunks": 0, "added_at": "migrated",
        }

with open(os.path.join(SHARED_DIR, "index.json"), "w") as f:
    json.dump(index, f, indent=2)

print(f"\n✅ Rebuilt index with {len(index)} video(s).")