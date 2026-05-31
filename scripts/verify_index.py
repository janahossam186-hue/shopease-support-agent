"""Quick verification script — checks ChromaDB collection counts without loading any ML model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
import chromadb
from chromadb.config import Settings as CS

client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=CS(anonymized_telemetry=False),
)

collections = [
    "product_catalog", "faqs", "policies",
    "product_manuals", "cosmetics_catalog",
    "recommendations", "store_info", "all_docs",
]

print("\nChromaDB collection stats:")
print("-" * 38)
subtotal = 0
for name in collections:
    try:
        c = client.get_collection(name)
        n = c.count()
        if name != "all_docs":
            subtotal += n
        print(f"  {name:<25} {n:>4} docs")
    except Exception as e:
        print(f"  {name:<25}  MISSING ({e})")

print("-" * 38)
print(f"  {'subtotal (excl. all_docs)':<25} {subtotal:>4}")
print()

# Spot-check a document from each new collection
print("Spot-check samples:")
for col_name in ["product_manuals", "cosmetics_catalog", "recommendations", "store_info"]:
    try:
        c = client.get_collection(col_name)
        result = c.peek(limit=1)
        if result["documents"]:
            preview = result["documents"][0][:80].replace("\n", " ")
            print(f"  [{col_name}] {preview}…")
    except Exception as e:
        print(f"  [{col_name}] error: {e}")

print("\nVerification complete.")
