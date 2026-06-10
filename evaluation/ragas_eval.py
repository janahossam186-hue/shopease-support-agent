"""
RAGAS Evaluation — measures RAG pipeline quality using 4 metrics:
  1. Context Precision  — are retrieved docs relevant to the question?
  2. Context Recall     — do retrieved docs contain all needed information?
  3. Faithfulness       — is the generated answer grounded in retrieved docs?
  4. Answer Relevancy   — does the answer actually address the question?

Compares three retrievers: Naive, Hybrid, Agentic.

Run with:
    .venv\\Scripts\\python.exe evaluation/ragas_eval.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ── Ground truth dataset ──────────────────────────────────────────────────────

RAGAS_GROUND_TRUTH = [
    # Returns Policy
    {
        "question": "What is the return window?",
        "ground_truth": "ShopEase offers a standard return window of 30 days from the date of delivery. Items purchased between November 1 and December 31 may be returned until January 31 of the following year. After 30 days returns are generally not accepted and require manager approval with a 15% restocking fee."
    },
    {
        "question": "What items cannot be returned?",
        "ground_truth": "The following items cannot be returned: digital downloads and software licenses, perishable goods such as food and flowers, personal care items once opened including razors and cosmetics, hazardous materials, gift cards and store credit, items marked as Final Sale or Non-Returnable, customised or personalised items unless defective, and underwear and swimwear for hygiene reasons."
    },
    {
        "question": "How long does a refund take to process?",
        "ground_truth": "Once ShopEase receives the returned item it is inspected within 1 to 2 business days. The refund is then processed within 3 to 5 business days of inspection approval. Credit card refunds take an additional 3 to 5 business days to appear depending on your bank. The total expected time is 7 to 12 business days from when ShopEase receives the return."
    },
    {
        "question": "How do I return a damaged product?",
        "ground_truth": "If your package arrives damaged report it within 48 hours of delivery with photos or video of the damage. ShopEase will arrange a free return and priority replacement within 2 to 3 business days or issue a full refund. For packages damaged in transit ShopEase will file a carrier claim and investigations typically resolve within 5 to 7 business days."
    },

    # Shipping Policy
    {
        "question": "What shipping options are available?",
        "ground_truth": "ShopEase offers four shipping options: Standard Shipping in 5 to 7 business days which is free on orders of $50 or more and $5.99 otherwise, Express Shipping in 2 to 3 business days for $12.99, Overnight Shipping for the next business day for $24.99, and Same-Day Delivery in select cities for $14.99."
    },
    {
        "question": "How long does standard delivery take?",
        "ground_truth": "Standard shipping takes 5 to 7 business days. It is free for all domestic orders totalling $50 or more after discounts. For orders below $50 the cost is $5.99. Business days are Monday through Friday excluding federal holidays."
    },
    {
        "question": "Do you offer same-day delivery?",
        "ground_truth": "Yes ShopEase offers same-day delivery in select cities for $14.99. Orders must be placed before 2:00 PM EST on business days to qualify. In Egypt same-day delivery is available in Cairo and Giza with a fee of EGP 49.99 for orders placed before 2 PM."
    },

    # Product Manuals
    {
        "question": "How do I use the EcoBrew Coffee Maker?",
        "ground_truth": "Fill the water tank with fresh cold water up to the desired cup level. Place a filter in the filter basket and add ground coffee using 1 tablespoon per cup for standard brew or 2 tablespoons for stronger coffee. Slide the basket into place until it clicks. Press the Power button select your brew strength using the Strength button and press Brew to start. The thermal carafe keeps coffee hot for up to 2 hours."
    },
    {
        "question": "How do I fix the InstantPot burn warning?",
        "ground_truth": "The Burn warning on the CookMaster Instant Pot means the bottom is too dry. Add more liquid to the pot and deglaze the bottom by scraping any stuck food before retrying."
    },
    {
        "question": "How do I set up the FitTrack Smart Watch?",
        "ground_truth": "Charge the watch using the magnetic charging cable until full which takes approximately 2 hours. Press and hold the side button for 3 seconds to power on. Download the FitTrack app on your phone create an account and go to Settings then Bluetooth then Pair on the watch. Open the app and tap the plus button to add the device. Enter your profile details including height weight age and gender for accurate health tracking."
    },

    # Product Catalog
    {
        "question": "What is the price and specs of the ProBook Laptop 15?",
        "ground_truth": "The ProBook Laptop 15 is priced at $899.99. It features a 15-inch display with Intel Core i7 processor 16GB RAM and 512GB SSD with a full-HD IPS display. It comes with a 2-year warranty and is currently in stock."
    },
    {
        "question": "What are the features of the SoundWave Headphones?",
        "ground_truth": "The SoundWave Headphones are over-ear wireless headphones priced at $199.99. They feature active noise cancellation 40-hour battery life and Hi-Res audio certification. They come with a 1-year warranty and are currently in stock."
    },

    # Cosmetics
    {
        "question": "What are the ingredients in HydraGlow Vitamin C Serum?",
        "ground_truth": "The HydraGlow Vitamin C Serum by GlowLab contains 20% Vitamin C as L-Ascorbic Acid, Hyaluronic Acid, Vitamin E, Ferulic Acid, Niacinamide, and Aloe Vera Extract. It is priced at $49.99 for 30ml and is suitable for all skin types especially dull skin and hyperpigmentation."
    },
    {
        "question": "What skincare products are good for oily skin?",
        "ground_truth": "For oily skin ShopEase recommends the ClearSkin Salicylic Acid Face Wash with 2% Salicylic Acid for unclogging pores, the SkinBalance Oil-Control Toner with Niacinamide 5% to minimise pores and balance oil, and the ClearDerm Benzoyl Peroxide Spot Treatment for blemishes. The recommended routine is ClearSkin Face Wash morning and evening then SkinBalance Toner then HydraGlow Vitamin C Serum in the morning then AquaBoost Moisturiser then SunShield SPF 50+ every morning."
    },

    # Store Info
    {
        "question": "Where are ShopEase stores located in Cairo?",
        "ground_truth": "ShopEase has three stores in Cairo. ShopEase Cairo Festival City is at Cairo Festival City Mall on Ring Road in New Cairo open Sunday to Thursday 10 AM to 11 PM and Friday to Saturday 10 AM to midnight. ShopEase Maadi is at Degla Square Road 9 in Maadi open daily 10 AM to 10 PM. ShopEase Heliopolis is at City Stars Mall in Heliopolis open Sunday to Thursday 10 AM to 11 PM and Friday to Saturday 10 AM to midnight."
    },
    {
        "question": "What payment methods does ShopEase accept?",
        "ground_truth": "ShopEase Egypt accepts Cash on Delivery available nationwide, Credit and Debit Cards including Visa and Mastercard, Meeza Card, Vodafone Cash, Fawry payment at over 200000 locations, and Bank Instalment Plans with 0% interest for 12 months on orders above EGP 5000 with Banque Misr CIB and QNB."
    },
    {
        "question": "What is the customer support phone number and hours?",
        "ground_truth": "ShopEase Egypt customer support phone number is 19123. WhatsApp is available at +20 100 123 4567. Email is support@shopease.eg. Support hours are Saturday to Thursday 9 AM to 10 PM and Friday 12 PM to 8 PM."
    },

    # FAQs
    {
        "question": "How do I track my order?",
        "ground_truth": "You can track your order by visiting the My Orders section in your account dashboard. Once your order ships you will receive a confirmation email with a tracking number and a direct link to the carrier tracking page. Tracking updates typically appear within 24 hours of shipment on the carrier website. Carriers used include UPS FedEx USPS and regional carriers."
    },
    {
        "question": "Can I cancel my order after placing it?",
        "ground_truth": "Yes you can modify or cancel an existing order through the AI support agent while the order is still in Processing status. Available modifications include cancelling the order updating the shipping address changing item quantities or removing items. Once an order moves to Shipped or any later status modifications are no longer possible and you will need to initiate a return after delivery. New orders can only be placed through the website or mobile app."
    },

    # Recommendations
    {
        "question": "What are the current promotions at ShopEase?",
        "ground_truth": "ShopEase current promotions include the Mega Friday Sale with up to 70% off electronics and beauty products every last Friday of the month, the New Customer Welcome Gift giving first-time customers 10% off their first order plus free shipping using code WELCOME10, the Student Discount giving verified students 15% off laptops tablets and accessories using code STUDENT15, the Summer Glow Sale giving 20% off sunscreens and skincare using code SUMMER20, and the VIP Loyalty Programme where customers earn 1 point per EGP spent and can redeem 1000 points for EGP 50 off."
    },
]


# ── Step 1: Retrieve docs + generate answers ──────────────────────────────────

def generate_answers(retriever_fn, retriever_name: str, top_k: int = 3) -> list[dict]:
    """
    For each question in RAGAS_GROUND_TRUTH:
      1. Retrieve top-k docs using retriever_fn(question, top_k)
      2. Generate an answer using the LLM + retrieved docs as context
    Returns a list of {question, answer, contexts, ground_truth} dicts.
    """
    from langchain_groq import ChatGroq
    from config.settings import settings

    llm = ChatGroq(model=settings.model_name, temperature=0.0, api_key=settings.groq_api_key)

    dataset = []
    for i, item in enumerate(RAGAS_GROUND_TRUTH):
        question = item["question"]
        print(f"  [{retriever_name}] {i + 1}/{len(RAGAS_GROUND_TRUTH)}: {question[:55]}...")

        if i > 0:
            time.sleep(2)

        try:
            docs     = retriever_fn(question, top_k)
            contexts = [d.content for d in docs] if docs else ["No relevant documents found."]

            prompt = f"""You are a helpful ShopEase customer support agent.
Answer the customer question using ONLY the provided context.
If the context does not contain the answer, say you don't have that information.

Context:
{chr(10).join(contexts)}

Question: {question}
Answer:"""

            answer = llm.invoke(prompt).content.strip()

        except Exception as e:
            logger.warning("Failed for '%s': %s", question, e)
            answer   = "Unable to generate answer."
            contexts = []

        dataset.append({
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": item["ground_truth"],
        })

    return dataset


# ── Step 2: Run RAGAS ─────────────────────────────────────────────────────────

def run_ragas_evaluation(dataset: list[dict]) -> dict:
    """
    Score the dataset using RAGAS.
    Uses Groq as judge LLM and local HuggingFace embeddings.
    Evaluates one metric at a time to avoid overwhelming Groq's rate limits.
    Returns a plain dict {metric_name: [per-sample scores]}.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from ragas.metrics._faithfulness import faithfulness
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from config.settings import settings

    # Groq only supports n=1; answer_relevancy defaults to strictness=3 (n=3)
    answer_relevancy.strictness = 1

    llm = ChatGroq(
        model=settings.model_name,
        temperature=0.0,
        api_key=settings.groq_api_key,
        n=1,
    )
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model, cache_folder="E:/hf_cache")
    )

    run_config = RunConfig(timeout=120, max_retries=3, max_workers=1)

    hf_dataset = Dataset.from_dict({
        "question":     [d["question"]     for d in dataset],
        "answer":       [d["answer"]       for d in dataset],
        "contexts":     [d["contexts"]     for d in dataset],
        "ground_truth": [d["ground_truth"] for d in dataset],
    })

    metrics_to_run = [
        ("context_precision", context_precision),
        ("context_recall",    context_recall),
        ("faithfulness",      faithfulness),
        ("answer_relevancy",  answer_relevancy),
    ]

    scores: dict[str, list] = {}
    for i, (metric_name, metric) in enumerate(metrics_to_run):
        if i > 0:
            print(f"  Sleeping 15s before next metric...")
            time.sleep(15)
        try:
            print(f"  Evaluating {metric_name}...")
            result = evaluate(
                dataset=hf_dataset,
                metrics=[metric],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                run_config=run_config,
            )
            scores[metric_name] = result[metric_name]
        except Exception as e:
            logger.warning("RAGAS metric '%s' failed: %s", metric_name, e)
            scores[metric_name] = []

    return scores


# ── Step 3: Save results ──────────────────────────────────────────────────────

def _mean(result, key: str) -> float:
    """Return the nanmean of a per-sample score list from an EvaluationResult."""
    import numpy as np
    vals = [v for v in result[key] if v is not None]
    if not vals:
        return 0.0
    m = np.nanmean(vals)
    return float(m) if not np.isnan(m) else 0.0


def save_results(result, dataset: list[dict], name: str) -> None:
    """Save scores to evaluation/ragas_results_{name}.json for the dashboard to read."""
    output = {
        "scores": {
            "context_precision": _mean(result, "context_precision"),
            "context_recall":    _mean(result, "context_recall"),
            "faithfulness":      _mean(result, "faithfulness"),
            "answer_relevancy":  _mean(result, "answer_relevancy"),
        },
        "num_queries": len(dataset),
    }
    path = Path(f"evaluation/ragas_results_{name.lower()}.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rag.naive_retriever import get_naive_retriever
    from rag.retriever import get_retriever
    from rag.agentic_retriever import get_agentic_retriever

    naive   = get_naive_retriever()
    hybrid  = get_retriever()
    agentic = get_agentic_retriever()

    print("Warming up retrievers...")
    naive.retrieve("return policy", 3)
    hybrid.retrieve("return policy", top_k_final=3)
    agentic.retrieve("return policy", top_k=3)
    print("Warmup complete.\n")

    all_results = {}

    retrievers = [
        #("Naive",   lambda q, k: naive.retrieve(q, k)),
        #("Hybrid",  lambda q, k: hybrid.retrieve(q, top_k_final=k)),
        ("Agentic", lambda q, k: agentic.retrieve(q, top_k=k)),
    ]

    for name, fn in retrievers:
        print(f"{'=' * 60}")
        print(f"Evaluating: {name} Retriever")
        print(f"{'=' * 60}")

        dataset = generate_answers(fn, name, top_k=3)
        result  = run_ragas_evaluation(dataset)
        all_results[name] = result

        cp  = _mean(result, "context_precision")
        cr  = _mean(result, "context_recall")
        f   = _mean(result, "faithfulness")
        ar  = _mean(result, "answer_relevancy")
        avg = (cp + cr + f + ar) / 4

        print(f"\nRAGAS Results — {name}")
        print(f"  Context Precision : {cp:.3f}")
        print(f"  Context Recall    : {cr:.3f}")
        print(f"  Faithfulness      : {f:.3f}")
        print(f"  Answer Relevancy  : {ar:.3f}")
        print(f"  Average           : {avg:.3f}\n")

        save_results(result, dataset, name)

    # Final comparison table — only show retrievers that were evaluated
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("RAGAS Comparison")
        print(f"{'=' * 60}")

        headers = list(all_results.keys())
        print(f"{'Metric':<25} " + " ".join(f"{h:>8}" for h in headers))
        print("-" * (25 + 9 * len(headers)))

        for metric in ["context_precision", "context_recall",
                       "faithfulness", "answer_relevancy"]:
            scores = [(h, _mean(all_results[h], metric)) for h in headers]
            winner = max(scores, key=lambda x: x[1])[0]
            row = f"{metric:<25} " + " ".join(f"{s:>8.3f}" for _, s in scores)
            print(f"{row}  ← {winner}")
    else:
        print(f"\nOnly {list(all_results.keys())[0]} evaluated. Run Hybrid and Agentic tomorrow.")