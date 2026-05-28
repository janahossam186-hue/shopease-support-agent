"""
DocumentIndexer — loads product catalog, FAQs, policy documents,
product manuals, cosmetics catalog, recommendations, and store info,
chunks them, embeds them, and persists them in ChromaDB.

Run once (or after data changes) with:
    python scripts/index_documents.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from rag.embeddings import LocalEmbeddings

logger = logging.getLogger(__name__)

# Collection names — one per knowledge source
COLLECTION_PRODUCTS = "product_catalog"
COLLECTION_FAQS = "faqs"
COLLECTION_POLICIES = "policies"
COLLECTION_MANUALS = "product_manuals"
COLLECTION_COSMETICS = "cosmetics_catalog"
COLLECTION_RECOMMENDATIONS = "recommendations"
COLLECTION_STORE = "store_info"
COLLECTION_CUSTOMERS = "customers"
COLLECTION_ALL = "all_docs"  # unified collection for hybrid search


class DocumentIndexer:
    """
    Loads source data, splits it into chunks, and indexes into ChromaDB.
    """

    def __init__(self):
        self.embedder = LocalEmbeddings(model_name=settings.embedding_model)
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.data_dir = Path("./data")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def _upsert(self, collection, ids, documents, metadatas):
        embeddings = self.embedder.embed_documents(documents)
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Upserted %d docs into collection '%s'", len(ids), collection.name)

    # ── loaders ───────────────────────────────────────────────────────────────

    def _load_products(self) -> tuple[list, list, list]:
        path = self.data_dir / "product_catalog.json"
        products: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))

        ids, docs, metas = [], [], []
        for p in products:
            doc_text = (
                f"Product: {p['name']}\n"
                f"Category: {p['category']}\n"
                f"Price: ${p['price']:.2f}\n"
                f"Description: {p['description']}\n"
                f"Warranty: {p['warranty_years']} year(s)\n"
                f"In Stock: {'Yes' if p['in_stock'] else 'No'}\n"
                f"SKU: {p['sku']}"
            )
            ids.append(f"product_{p['product_id']}")
            docs.append(doc_text)
            metas.append(
                {
                    "source": "product_catalog",
                    "product_id": p["product_id"],
                    "category": p["category"],
                    "price": p["price"],
                    "in_stock": str(p["in_stock"]),
                }
            )
        return ids, docs, metas

    def _load_faqs(self) -> tuple[list, list, list]:
        path = self.data_dir / "faqs.json"
        faqs: list[dict] = json.loads(path.read_text(encoding="utf-8"))

        ids, docs, metas = [], [], []
        for faq in faqs:
            doc_text = (
                f"Question: {faq['question']}\n"
                f"Answer: {faq['answer']}"
            )
            ids.append(f"faq_{faq['id']}")
            docs.append(doc_text)
            metas.append(
                {
                    "source": "faq",
                    "faq_id": faq["id"],
                    "category": faq["category"],
                }
            )
        return ids, docs, metas

    def _load_policies(self) -> tuple[list, list, list]:
        """Chunk policy markdown files into ~500-token passages."""
        policy_files = {
            "shipping_policy": self.data_dir / "shipping_policy.md",
            "returns_policy": self.data_dir / "returns_policy.md",
        }

        ids, docs, metas = [], [], []
        for policy_name, path in policy_files.items():
            text = path.read_text(encoding="utf-8")
            chunks = self._chunk_text(text, chunk_size=500, overlap=50)
            for i, chunk in enumerate(chunks):
                ids.append(f"{policy_name}_chunk_{i}")
                docs.append(chunk)
                metas.append(
                    {
                        "source": policy_name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                )
        return ids, docs, metas

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Naive word-count chunker with overlap."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap
        return chunks

    # ── new data loaders ─────────────────────────────────────────────────────

    def _load_manuals(self) -> tuple[list, list, list]:
        """Index product manuals (usage steps + troubleshooting + maintenance)."""
        path = self.data_dir / "product_manuals.json"
        manuals: list[dict] = json.loads(path.read_text(encoding="utf-8"))

        ids, docs, metas = [], [], []
        for m in manuals:
            pid = m.get("product_id", "UNKNOWN")
            name = m.get("product_name", "Unknown Product")
            brand = m.get("brand", "")

            # Usage steps
            usage_text = "\n".join(m.get("step_by_step_usage", []))
            ids.append(f"manual_usage_{pid}")
            docs.append(
                f"Product: {name} ({brand})\nHow to use:\n{usage_text}"
            )
            metas.append({"source": "product_manuals", "product_id": pid,
                          "section": "usage", "product_name": name})

            # Troubleshooting
            trouble_text = "\n".join(m.get("troubleshooting_tips", []))
            ids.append(f"manual_trouble_{pid}")
            docs.append(
                f"Product: {name} ({brand})\nTroubleshooting:\n{trouble_text}"
            )
            metas.append({"source": "product_manuals", "product_id": pid,
                          "section": "troubleshooting", "product_name": name})

            # Maintenance
            maint_text = "\n".join(m.get("maintenance_tips", []))
            ids.append(f"manual_maint_{pid}")
            docs.append(
                f"Product: {name} ({brand})\nMaintenance tips:\n{maint_text}"
            )
            metas.append({"source": "product_manuals", "product_id": pid,
                          "section": "maintenance", "product_name": name})

        return ids, docs, metas

    def _load_cosmetics(self) -> tuple[list, list, list]:
        """Index cosmetics catalog (skin type, how to use, ingredients, benefits)."""
        path = self.data_dir / "cosmetics_catalog.json"
        items: list[dict] = json.loads(path.read_text(encoding="utf-8"))

        ids, docs, metas = [], [], []
        for item in items:
            pid = item.get("product_id", "COS-XXX")
            name = item.get("product_name", "Unknown")
            brand = item.get("brand", "")
            skin_type = ", ".join(item.get("skin_type", []))
            how_to = item.get("how_to_use", "")
            ingredients = ", ".join(item.get("ingredients", []))
            benefits = item.get("key_benefits", "")
            category = item.get("subcategory", item.get("category", ""))
            price = item.get("price", 0)

            doc_text = (
                f"Product: {name}\nBrand: {brand}\nCategory: {category}\n"
                f"Skin/Hair Type: {skin_type}\n"
                f"Key Benefits: {benefits}\n"
                f"How to Use: {how_to}\n"
                f"Ingredients: {ingredients}\n"
                f"Price: ${price:.2f}"
            )
            ids.append(f"cosmetic_{pid}")
            docs.append(doc_text)
            metas.append({
                "source": "cosmetics_catalog",
                "product_id": pid,
                "category": category,
                "brand": brand,
                "price": price,
                "skin_type": skin_type,
            })
        return ids, docs, metas

    def _load_recommendations(self) -> tuple[list, list, list]:
        """Index bundles, trending items, seasonal offers, and FBT suggestions."""
        path = self.data_dir / "recommendations.json"
        data: dict = json.loads(path.read_text(encoding="utf-8"))

        ids, docs, metas = [], [], []

        # Bundles
        for b in data.get("product_bundles", []):
            bid = b["bundle_id"]
            products_str = ", ".join(b.get("products", []))
            tags = ", ".join(b.get("tags", []))
            doc_text = (
                f"Bundle: {b['name']}\n"
                f"Description: {b['description']}\n"
                f"Products included: {products_str}\n"
                f"Bundle price: ${b.get('bundle_price', 0):.2f} "
                f"(save ${b.get('savings', 0):.2f})\n"
                f"Tags: {tags}"
            )
            ids.append(f"bundle_{bid}")
            docs.append(doc_text)
            metas.append({"source": "recommendations", "type": "bundle",
                          "bundle_id": bid})

        # Trending items
        trending_parts = []
        for t in data.get("trending_items", []):
            trending_parts.append(
                f"#{t['rank']}: {t['product_name']} — {t['reason']}"
            )
        if trending_parts:
            doc_text = "Trending Products at ShopEase:\n" + "\n".join(trending_parts)
            ids.append("trending_items")
            docs.append(doc_text)
            metas.append({"source": "recommendations", "type": "trending"})

        # Seasonal offers
        for offer in data.get("seasonal_offers", []):
            oid = offer["offer_id"]
            products_str = ", ".join(offer.get("applicable_products", []))
            code = offer.get("promo_code", "N/A")
            doc_text = (
                f"Seasonal Offer: {offer['name']}\n"
                f"Description: {offer['description']}\n"
                f"Discount: {offer.get('discount', offer.get('discount_percent', ''))}% off\n"
                f"Promo code: {code}\n"
                f"Applicable products: {products_str}\n"
                f"Valid until: {offer.get('valid_until', 'See website')}"
            )
            ids.append(f"offer_{oid}")
            docs.append(doc_text)
            metas.append({"source": "recommendations", "type": "seasonal_offer",
                          "offer_id": oid})

        # Frequently bought together
        for fbt in data.get("frequently_bought_together", []):
            pid = fbt["primary_product"]
            bought_with_parts = [
                f"{x['product_name']} ({int(x['match_rate']*100)}% of customers also buy this)"
                for x in fbt.get("bought_with", [])
            ]
            doc_text = (
                f"Customers who buy {fbt['primary_name']} also frequently buy:\n"
                + "\n".join(bought_with_parts)
            )
            ids.append(f"fbt_{pid}")
            docs.append(doc_text)
            metas.append({"source": "recommendations", "type": "frequently_bought_together",
                          "primary_product": pid})

        # Personalised routine tips
        for tip in data.get("personalised_tips", []):
            cat = tip["category"].replace(" ", "_").lower()
            ids.append(f"tip_{cat}")
            docs.append(f"Skincare/Beauty Tip — {tip['category']}:\n{tip['tip']}")
            metas.append({"source": "recommendations", "type": "beauty_tip"})

        return ids, docs, metas

    def _load_customers(self) -> tuple[list, list, list]:
        """Index customer profiles for RAG — names and IDs only.
        Emails and phone numbers are intentionally excluded for privacy."""
        path = self.data_dir / "customers.json"
        customers: dict = json.loads(path.read_text(encoding="utf-8"))

        ids, docs, metas = [], [], []
        for cid, c in customers.items():
            doc_text = (
                f"Customer: {c['name']}\n"
                f"Customer ID: {cid}"
            )
            ids.append(f"customer_{cid}")
            docs.append(doc_text)
            metas.append({
                "source": "customers",
                "customer_id": cid,
            })
        return ids, docs, metas

    def _load_store_info(self) -> tuple[list, list, list]:
        """Index store locations, delivery info, promotions, and website help."""
        path = self.data_dir / "store_info.json"
        data: dict = json.loads(path.read_text(encoding="utf-8"))

        ids, docs, metas = [], [], []

        # Store locations
        for branch in data.get("store_locations", []):
            bid = branch["branch_id"]
            services = ", ".join(branch.get("services", []))
            doc_text = (
                f"ShopEase Store: {branch['name']}\n"
                f"City: {branch['city']}, {branch['area']}\n"
                f"Address: {branch['address']}\n"
                f"Phone: {branch['phone']}\n"
                f"Hours: {branch['hours']}\n"
                f"Services: {services}\n"
                f"Parking: {branch.get('parking', 'See website')}"
            )
            ids.append(f"store_{bid}")
            docs.append(doc_text)
            metas.append({"source": "store_info", "type": "store_location",
                          "city": branch["city"], "branch_id": bid})

        # Delivery info
        delivery = data.get("delivery", {})
        delivery_text = (
            f"ShopEase Delivery Options:\n"
            f"Same-Day Delivery: Available in {', '.join(delivery.get('same_day_delivery', {}).get('available_cities', []))}. "
            f"Order before {delivery.get('same_day_delivery', {}).get('cutoff_time', '2 PM')}. "
            f"Fee: EGP {delivery.get('same_day_delivery', {}).get('fee', 49.99):.0f}.\n"
            f"Next-Day Delivery: {', '.join(delivery.get('next_day_delivery', {}).get('available_cities', []))}. "
            f"Fee: EGP {delivery.get('next_day_delivery', {}).get('fee', 29.99):.0f}.\n"
            f"Standard Delivery: {delivery.get('standard_delivery', {}).get('coverage', 'All governorates')}. "
            f"{delivery.get('standard_delivery', {}).get('days', '2-5 days')}. "
            f"Free above EGP {delivery.get('standard_delivery', {}).get('free_above', 500):.0f}."
        )
        ids.append("delivery_info")
        docs.append(delivery_text)
        metas.append({"source": "store_info", "type": "delivery"})

        # Payment methods
        payment_methods = "\n".join(data.get("payment_methods", []))
        ids.append("payment_methods")
        docs.append(f"ShopEase Payment Methods:\n{payment_methods}")
        metas.append({"source": "store_info", "type": "payment"})

        # Promotions
        for promo in data.get("promotions", []):
            pid = promo["promo_id"]
            code = promo.get("promo_code", "No code needed")
            doc_text = (
                f"Promotion: {promo['name']}\n"
                f"Details: {promo['description']}\n"
                f"Promo code: {code}\n"
                f"Valid: {promo.get('valid_until', 'See website')}"
            )
            ids.append(f"promo_{pid}")
            docs.append(doc_text)
            metas.append({"source": "store_info", "type": "promotion",
                          "promo_id": pid})

        # Website help
        website_help = data.get("website_help", {})
        for topic, content in website_help.items():
            ids.append(f"webhelp_{topic}")
            docs.append(f"Website Help — {topic.replace('_', ' ').title()}:\n{content}")
            metas.append({"source": "store_info", "type": "website_help",
                          "topic": topic})

        # Contact info
        contact = data.get("contact", {})
        contact_text = (
            f"ShopEase Egypt Contact Information:\n"
            f"Phone: {contact.get('customer_support_phone', '19123')}\n"
            f"WhatsApp: {contact.get('whatsapp', '')}\n"
            f"Email: {contact.get('email', '')}\n"
            f"Support Hours: {contact.get('support_hours', '')}"
        )
        ids.append("contact_info")
        docs.append(contact_text)
        metas.append({"source": "store_info", "type": "contact"})

        return ids, docs, metas

    # ── public API ────────────────────────────────────────────────────────────

    def index_all(self) -> None:
        """Index all data sources into ChromaDB. Safe to call multiple times (upserts)."""
        logger.info("Starting document indexing…")

        prod_ids, prod_docs, prod_metas = self._load_products()
        faq_ids, faq_docs, faq_metas = self._load_faqs()
        pol_ids, pol_docs, pol_metas = self._load_policies()
        man_ids, man_docs, man_metas = self._load_manuals()
        cos_ids, cos_docs, cos_metas = self._load_cosmetics()
        rec_ids, rec_docs, rec_metas = self._load_recommendations()
        sto_ids, sto_docs, sto_metas = self._load_store_info()
        cust_ids, cust_docs, cust_metas = self._load_customers()

        # Per-source collections
        self._upsert(
            self._get_or_create_collection(COLLECTION_PRODUCTS),
            prod_ids, prod_docs, prod_metas,
        )
        self._upsert(
            self._get_or_create_collection(COLLECTION_FAQS),
            faq_ids, faq_docs, faq_metas,
        )
        self._upsert(
            self._get_or_create_collection(COLLECTION_POLICIES),
            pol_ids, pol_docs, pol_metas,
        )
        self._upsert(
            self._get_or_create_collection(COLLECTION_MANUALS),
            man_ids, man_docs, man_metas,
        )
        self._upsert(
            self._get_or_create_collection(COLLECTION_COSMETICS),
            cos_ids, cos_docs, cos_metas,
        )
        self._upsert(
            self._get_or_create_collection(COLLECTION_RECOMMENDATIONS),
            rec_ids, rec_docs, rec_metas,
        )
        self._upsert(
            self._get_or_create_collection(COLLECTION_STORE),
            sto_ids, sto_docs, sto_metas,
        )
        self._upsert(
            self._get_or_create_collection(COLLECTION_CUSTOMERS),
            cust_ids, cust_docs, cust_metas,
        )

        # Unified collection for cross-source hybrid retrieval
        all_col = self._get_or_create_collection(COLLECTION_ALL)
        all_ids = (prod_ids + faq_ids + pol_ids + man_ids +
                   cos_ids + rec_ids + sto_ids + cust_ids)
        all_docs = (prod_docs + faq_docs + pol_docs + man_docs +
                    cos_docs + rec_docs + sto_docs + cust_docs)
        all_metas = (prod_metas + faq_metas + pol_metas + man_metas +
                     cos_metas + rec_metas + sto_metas + cust_metas)
        self._upsert(all_col, all_ids, all_docs, all_metas)

        total = len(all_ids)
        logger.info("Indexing complete. Total documents indexed: %d", total)
        print(f"✓ Indexed {total} documents into ChromaDB.")
        print(f"  Products: {len(prod_ids)} | FAQs: {len(faq_ids)} | "
              f"Policies: {len(pol_ids)} | Manuals: {len(man_ids)} | "
              f"Cosmetics: {len(cos_ids)} | Recommendations: {len(rec_ids)} | "
              f"Store info: {len(sto_ids)} | Customers: {len(cust_ids)}")

    def get_collection_stats(self) -> dict:
        stats = {}
        for name in [COLLECTION_PRODUCTS, COLLECTION_FAQS, COLLECTION_POLICIES,
                     COLLECTION_MANUALS, COLLECTION_COSMETICS,
                     COLLECTION_RECOMMENDATIONS, COLLECTION_STORE,
                     COLLECTION_CUSTOMERS, COLLECTION_ALL]:
            try:
                col = self.client.get_collection(name)
                stats[name] = col.count()
            except Exception:
                stats[name] = 0
        return stats
