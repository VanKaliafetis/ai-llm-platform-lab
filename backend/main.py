import json
import tempfile

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from core.database import cleanup_mock_rows, init_db, insert, rows
from core.providers import complete
from core.rag_pipeline import (
    build_context,
    index_status,
    ingest_document,
    reset_index,
    retrieve,
)
from core.scoring import answer_similarity, groundedness, retrieval_score, score_label
from reports.report_export import build_markdown_report


OLLAMA_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen2.5:0.5b"
COMPARE_MODELS = ["qwen2.5:0.5b", "phi3:mini"]


init_db()
app = FastAPI(title="LLM Platform Lab API")


class PromptRun(BaseModel):
    prompt: str
    provider: str = OLLAMA_PROVIDER
    model: str = DEFAULT_MODEL


class RagQuery(BaseModel):
    question: str
    context: str
    provider: str = OLLAMA_PROVIDER
    model: str = DEFAULT_MODEL


class RagAsk(BaseModel):
    question: str
    provider: str = OLLAMA_PROVIDER
    model: str = DEFAULT_MODEL
    top_k: int = 5


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": OLLAMA_PROVIDER,
        "default_model": DEFAULT_MODEL,
        "compare_models": COMPARE_MODELS,
        "rag_index": index_status(),
    }


@app.post("/run")
def run(req: PromptRun):
    provider = OLLAMA_PROVIDER
    model = req.model or DEFAULT_MODEL

    r = complete(req.prompt, provider, model)

    row = {
        "module": "inference",
        "provider": provider,
        "model": model,
        "prompt": req.prompt,
        "response": r.text,
        "latency_ms": r.latency_ms,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "cost_usd": r.cost_usd,
        "tokens_per_second": r.tokens_per_second,
        "score": None,
        "metadata": json.dumps(
            {
                "provider_requested": req.provider,
                "model_requested": req.model,
                "backend_mode": "ollama_only",
            }
        ),
    }

    row["id"] = insert("runs", row)
    return row


@app.post("/compare")
def compare(req: PromptRun):
    out = []

    for model in COMPARE_MODELS:
        r = complete(req.prompt, OLLAMA_PROVIDER, model)

        data = {
            "module": "comparison",
            "provider": OLLAMA_PROVIDER,
            "model": model,
            "prompt": req.prompt,
            "response": r.text,
            "latency_ms": r.latency_ms,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost_usd": r.cost_usd,
            "tokens_per_second": r.tokens_per_second,
            "score": None,
            "metadata": json.dumps(
                {
                    "backend_mode": "ollama_only",
                    "compared_models": COMPARE_MODELS,
                }
            ),
        }

        data["id"] = insert("runs", data)
        out.append(data)

    return out


@app.post("/eval/upload")
async def eval_upload(
    provider: str = OLLAMA_PROVIDER,
    model: str = DEFAULT_MODEL,
    file: UploadFile = File(...),
):
    selected_model = model or DEFAULT_MODEL

    b = await file.read()
    suffix = file.filename.split(".")[-1].lower()

    path = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}").name

    with open(path, "wb") as f:
        f.write(b)

    if suffix == "jsonl":
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)

    results = []

    for _, row in df.iterrows():
        prompt = str(row.get("prompt") or row.get("question") or "")
        expected = str(row.get("expected") or row.get("answer") or "")

        if not prompt.strip():
            continue

        r = complete(prompt, OLLAMA_PROVIDER, selected_model)
        score = answer_similarity(expected, r.text)

        data = {
            "dataset_name": file.filename,
            "provider": OLLAMA_PROVIDER,
            "model": selected_model,
            "prompt": prompt,
            "expected": expected,
            "response": r.text,
            "score": score,
            "latency_ms": r.latency_ms,
            "tokens_per_second": r.tokens_per_second,
            "metadata": json.dumps(
                {
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost_usd": r.cost_usd,
                    "score_label": score_label(score),
                    "provider_requested": provider,
                    "model_requested": model,
                    "backend_mode": "ollama_only",
                }
            ),
        }

        data["id"] = insert("eval_results", data)
        results.append(data)

    return {
        "rows": len(results),
        "provider": OLLAMA_PROVIDER,
        "model": selected_model,
        "results": results,
    }


@app.post("/rag/evaluate")
def rag(req: RagQuery):
    selected_model = req.model or DEFAULT_MODEL

    chunks = [c.strip() for c in req.context.split("\n\n") if c.strip()][:5]
    ctx = "\n\n".join(chunks)

    prompt = (
        "You are evaluating a retrieved context for a RAG system.\n"
        "Answer the user's question using ONLY the provided context.\n"
        "Do not add outside information.\n"
        "If the answer is not clearly present, say the context does not contain enough information.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {req.question}\n\n"
        "Answer:"
    )

    r = complete(req.prompt if hasattr(req, "prompt") else prompt, OLLAMA_PROVIDER, selected_model)

    ret = retrieval_score(req.question, chunks)
    grd = groundedness(r.text, ctx)

    data = {
        "question": req.question,
        "answer": r.text,
        "retrieved_context": ctx,
        "retrieval_score": ret,
        "groundedness_score": grd,
        "metadata": json.dumps(
            {
                "provider": OLLAMA_PROVIDER,
                "model": selected_model,
                "latency_ms": r.latency_ms,
                "tokens_per_second": r.tokens_per_second,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
                "retrieval_label": score_label(ret),
                "groundedness_label": score_label(grd),
                "backend_mode": "manual_context_rag",
            }
        ),
    }

    data["id"] = insert("rag_results", data)
    return data


@app.post("/rag/ingest")
async def rag_ingest(file: UploadFile = File(...)):
    content = await file.read()
    return ingest_document(file.filename, content)


@app.get("/rag/status")
def rag_status():
    return index_status()


@app.delete("/rag/index")
def rag_reset():
    return reset_index()


@app.post("/rag/ask")
def rag_ask(req: RagAsk):
    selected_model = req.model or DEFAULT_MODEL

    retrieved = retrieve(req.question, top_k=req.top_k)
    ctx = build_context(retrieved)

    prompt = (
        "You are answering using a retrieved document index.\n"
        "Use ONLY the retrieved context below.\n"
        "Cite source names when useful.\n"
        "If the answer is not in the context, say the context does not contain enough information.\n\n"
        f"Retrieved context:\n{ctx}\n\n"
        f"Question: {req.question}\n\n"
        "Answer:"
    )

    r = complete(prompt, OLLAMA_PROVIDER, selected_model, num_predict=320)

    ret = retrieval_score(req.question, [item.text for item in retrieved])
    grd = groundedness(r.text, ctx)

    data = {
        "question": req.question,
        "answer": r.text,
        "retrieved_context": ctx,
        "retrieval_score": ret,
        "groundedness_score": grd,
        "metadata": json.dumps(
            {
                "provider": OLLAMA_PROVIDER,
                "model": selected_model,
                "latency_ms": r.latency_ms,
                "tokens_per_second": r.tokens_per_second,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
                "retrieval_label": score_label(ret),
                "groundedness_label": score_label(grd),
                "rag_mode": "automatic_vector_retrieval",
                "retrieved": [item.__dict__ for item in retrieved],
            }
        ),
    }

    data["id"] = insert("rag_results", data)

    return {
        **data,
        "retrieved_chunks": [item.__dict__ for item in retrieved],
    }


@app.get("/runs")
def get_runs():
    return rows("SELECT * FROM runs ORDER BY id DESC LIMIT 300")


@app.get("/evals")
def get_evals():
    return rows("SELECT * FROM eval_results ORDER BY id DESC LIMIT 300")


@app.get("/rag/results")
def get_rag():
    return rows("SELECT * FROM rag_results ORDER BY id DESC LIMIT 300")


@app.get("/reports/export")
def export_report():
    return build_markdown_report()


@app.delete("/admin/cleanup-mock")
def cleanup_mock():
    deleted = cleanup_mock_rows()
    return {"status": "ok", "deleted_rows": deleted}