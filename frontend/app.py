import html
import os
import requests

import pandas as pd
import plotly.express as px
import streamlit as st


API = os.getenv("API_URL", "http://127.0.0.1:8000")

OLLAMA_MODELS = [
    "qwen2.5:0.5b",
    "phi3:mini",
]


st.set_page_config(page_title="LLM Platform Lab", layout="wide")
st.title("LLM Platform Lab")
st.caption("Local Ollama inference • model benchmarking • evals • vector RAG • observability • fine-tuning lab")


def call_api(method, path, **kwargs):
    try:
        if method == "GET":
            response = requests.get(f"{API}{path}", timeout=120, **kwargs)
        elif method == "DELETE":
            response = requests.delete(f"{API}{path}", timeout=120, **kwargs)
        else:
            response = requests.post(f"{API}{path}", timeout=300, **kwargs)

        response.raise_for_status()
        return response.json(), None

    except requests.exceptions.ConnectionError:
        return None, "Backend is not running. Start it with: uvicorn backend.main:app --reload"
    except requests.exceptions.Timeout:
        return None, "Request timed out. Ollama may still be loading the model."
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json()
        except Exception:
            detail = exc.response.text
        return None, f"Backend error: {detail}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


def score_badge(score):
    if score is None:
        return "N/A"

    score = float(score)

    if score >= 75:
        return "Strong"
    if score >= 45:
        return "Partial"
    return "Weak"


def show_response_box(text):
    safe_text = html.escape(str(text or ""))

    st.markdown(
        f"""
<div style="
    padding: 1rem;
    border: 1px solid #ddd;
    border-radius: 0.6rem;
    background-color: #f8f9fa;
    max-height: 360px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.45;
">
{safe_text}
</div>
""",
        unsafe_allow_html=True,
    )


def show_metric_row(result):
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Model", result.get("model", "N/A"))
    m2.metric("Latency", f"{float(result.get('latency_ms', 0)) / 1000:.2f}s")
    m3.metric("Input tokens", result.get("input_tokens", 0))
    m4.metric("Output tokens", result.get("output_tokens", 0))
    m5.metric("Tokens/sec", f"{float(result.get('tokens_per_second', 0)):.2f}")


with st.sidebar:
    st.header("Settings")

    provider = st.selectbox("Provider", ["ollama"])
    model = st.selectbox("Model", OLLAMA_MODELS)

    st.info("Using real local Ollama inference. Make sure Ollama is running and models are pulled.")

    st.code(
        "ollama list\n"
        "ollama pull qwen2.5:0.5b\n"
        "ollama pull phi3:mini",
        language="bash",
    )

    st.divider()

    if st.button("Clear old mock/placeholder rows", use_container_width=True):
        result, error = call_api("DELETE", "/admin/cleanup-mock")
        if error:
            st.error(error)
        else:
            st.success(f"Deleted {result.get('deleted_rows', 0)} old rows.")

    if st.button("Reset RAG index", use_container_width=True):
        result, error = call_api("DELETE", "/rag/index")
        if error:
            st.error(error)
        else:
            st.success("RAG index reset.")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Prompt Playground",
        "Benchmark & Evals",
        "Vector RAG",
        "Manual RAG Eval",
        "Observability",
        "Fine-Tuning Lab",
    ]
)


with tab1:
    st.subheader("Prompt Playground")
    st.caption("Run a single prompt or compare both local Ollama models.")

    prompt = st.text_area(
        "Prompt",
        "Explain vector databases in simple terms and why they are useful for RAG.",
        height=150,
    )

    c1, c2 = st.columns(2)

    if c1.button("Run selected model", use_container_width=True):
        with st.spinner(f"Running {model} locally with Ollama..."):
            result, error = call_api(
                "POST",
                "/run",
                json={"prompt": prompt, "provider": provider, "model": model},
            )

        if error:
            st.error(error)
        else:
            show_metric_row(result)
            st.subheader("Response")
            show_response_box(result.get("response", ""))

    if c2.button("Compare qwen2.5:0.5b vs phi3:mini", use_container_width=True):
        with st.spinner("Running both Ollama models..."):
            result, error = call_api(
                "POST",
                "/compare",
                json={"prompt": prompt, "provider": "ollama", "model": model},
            )

        if error:
            st.error(error)
        else:
            df = pd.DataFrame(result)
            st.subheader("Comparison Results")

            show_cols = [
                "model",
                "latency_ms",
                "tokens_per_second",
                "input_tokens",
                "output_tokens",
                "response",
            ]
            existing_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(df[existing_cols], use_container_width=True, height=260)

            chart_df = df.copy()
            chart_df["latency_seconds"] = chart_df["latency_ms"] / 1000

            st.plotly_chart(
                px.bar(
                    chart_df,
                    x="model",
                    y="latency_seconds",
                    title="Latency comparison",
                    text_auto=".2f",
                ),
                use_container_width=True,
            )

            if "tokens_per_second" in chart_df.columns:
                st.plotly_chart(
                    px.bar(
                        chart_df,
                        x="model",
                        y="tokens_per_second",
                        title="Generation speed comparison",
                        text_auto=".2f",
                    ),
                    use_container_width=True,
                )


with tab2:
    st.subheader("Batch Evaluation")
    st.caption("Upload prompts and expected answers, then evaluate one local model.")

    example = pd.DataFrame(
        [
            {
                "prompt": "What is RAG?",
                "expected": "Retrieval augmented generation combines retrieved context with generation.",
            },
            {
                "prompt": "What is latency in inference?",
                "expected": "Latency is the time taken for the model to return a response.",
            },
        ]
    )

    with st.expander("Example CSV format"):
        st.dataframe(example, use_container_width=True)

    f = st.file_uploader("Upload CSV/JSONL with prompt and expected columns", type=["csv", "jsonl"])

    selected_eval_model = st.selectbox("Evaluation model", OLLAMA_MODELS, key="eval_model")

    if f and st.button("Run evaluation dataset", use_container_width=True):
        with st.spinner(f"Evaluating dataset with {selected_eval_model}..."):
            result, error = call_api(
                "POST",
                "/eval/upload",
                params={"provider": "ollama", "model": selected_eval_model},
                files={"file": (f.name, f.getvalue())},
            )

        if error:
            st.error(error)
        else:
            df = pd.DataFrame(result.get("results", []))
            st.success(f"Evaluated {result.get('rows', len(df))} rows")

            if not df.empty:
                st.dataframe(df, use_container_width=True, height=320)

                if "score" in df.columns:
                    avg = df["score"].dropna().mean()
                    st.metric("Average similarity score", f"{avg:.2f}", score_badge(avg))

                    st.plotly_chart(
                        px.histogram(df, x="score", title="Answer similarity score"),
                        use_container_width=True,
                    )


with tab3:
    st.subheader("Vector RAG")
    st.caption("Upload documents, build a local retrieval index, and ask questions over retrieved chunks.")

    status, status_error = call_api("GET", "/rag/status")

    if status_error:
        st.error(status_error)
    else:
        s1, s2, s3 = st.columns(3)
        s1.metric("Index status", status.get("status", "unknown"))
        s2.metric("Chunks", status.get("chunks", 0))
        s3.metric("Embedding backend", status.get("embedding_backend", "unknown"))

        if status.get("sources"):
            st.write("Sources:", ", ".join(status.get("sources", [])))

    upload = st.file_uploader(
        "Upload document for RAG index",
        type=["txt", "md", "csv", "json", "jsonl", "py"],
        key="rag_upload",
    )

    if upload and st.button("Ingest document", use_container_width=True):
        with st.spinner("Chunking, embedding and indexing document..."):
            result, error = call_api(
                "POST",
                "/rag/ingest",
                files={"file": (upload.name, upload.getvalue())},
            )

        if error:
            st.error(error)
        else:
            st.success(
                f"Indexed {result.get('chunks_added', 0)} chunks from {result.get('filename')}."
            )
            st.json(result)

    st.divider()

    rag_model = st.selectbox("Answer model", OLLAMA_MODELS, key="auto_rag_model")
    rag_question = st.text_input(
        "Question over indexed documents",
        "What are the main issues discussed in the document?",
    )
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=5)

    if st.button("Ask indexed documents", use_container_width=True):
        with st.spinner("Retrieving chunks and generating grounded answer..."):
            result, error = call_api(
                "POST",
                "/rag/ask",
                json={
                    "question": rag_question,
                    "provider": "ollama",
                    "model": rag_model,
                    "top_k": top_k,
                },
            )

        if error:
            st.error(error)
        else:
            st.subheader("Answer")
            show_response_box(result.get("answer", ""))

            c1, c2, c3 = st.columns(3)
            ret = result.get("retrieval_score", 0)
            grd = result.get("groundedness_score", 0)
            c1.metric("Retrieval score", ret, score_badge(ret))
            c2.metric("Groundedness score", grd, score_badge(grd))
            c3.metric("Model", rag_model)

            chunks = pd.DataFrame(result.get("retrieved_chunks", []))
            if not chunks.empty:
                st.subheader("Retrieved chunks")
                st.dataframe(chunks, use_container_width=True, height=260)


with tab4:
    st.subheader("Manual RAG Retrieval + Groundedness Scoring")
    st.caption("Paste retrieved context and test whether the model answer is grounded.")

    selected_rag_model = st.selectbox("RAG answer model", OLLAMA_MODELS, key="manual_rag_model")

    question = st.text_input("Question", "What is the closure action for the RFI?")

    context = st.text_area(
        "Retrieved context chunks separated by blank lines",
        height=220,
        value=(
            "RFI-104 requires missing load calculation evidence. Owner: design team. "
            "Closure requires signed calculation note.\n\n"
            "The due date is Friday and risk is medium until evidence is approved."
        ),
    )

    if st.button("Evaluate manual RAG answer", use_container_width=True):
        with st.spinner(f"Generating grounded answer with {selected_rag_model}..."):
            result, error = call_api(
                "POST",
                "/rag/evaluate",
                json={
                    "question": question,
                    "context": context,
                    "provider": "ollama",
                    "model": selected_rag_model,
                },
            )

        if error:
            st.error(error)
        else:
            st.subheader("Generated answer")
            show_response_box(result.get("answer", ""))

            c1, c2, c3 = st.columns(3)
            ret = result.get("retrieval_score", 0)
            grd = result.get("groundedness_score", 0)
            c1.metric("Retrieval score", ret, score_badge(ret))
            c2.metric("Groundedness score", grd, score_badge(grd))
            c3.metric("Model", selected_rag_model)

            st.text_area("Retrieved context used", result.get("retrieved_context", ""), height=150)


with tab5:
    st.subheader("Observability Dashboard")
    st.caption("View previous inference, evaluation and RAG runs.")

    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("Export markdown report", use_container_width=True):
            report, error = call_api("GET", "/reports/export")
            if error:
                st.error(error)
            else:
                st.success(f"Report generated: {report.get('path')}")
                st.download_button(
                    "Download markdown report",
                    data=report.get("markdown", ""),
                    file_name="platform_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

    runs_data, runs_error = call_api("GET", "/runs")
    evals_data, evals_error = call_api("GET", "/evals")
    rag_data, rag_error = call_api("GET", "/rag/results")

    if runs_error:
        st.error(runs_error)
    else:
        runs = pd.DataFrame(runs_data)

        if not runs.empty:
            st.markdown("### Inference Runs")

            runs["latency_seconds"] = runs["latency_ms"] / 1000
            if "tokens_per_second" not in runs.columns:
                runs["tokens_per_second"] = runs.apply(
                    lambda r: round(r["output_tokens"] / max(r["latency_ms"] / 1000, 0.001), 2),
                    axis=1,
                )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total runs", len(runs))
            k2.metric("Models tested", runs["model"].nunique())
            k3.metric("Avg latency", f"{runs['latency_seconds'].mean():.2f}s")
            k4.metric("Avg tokens/sec", f"{runs['tokens_per_second'].mean():.2f}")

            display_cols = [
                "id",
                "created_at",
                "module",
                "provider",
                "model",
                "latency_ms",
                "tokens_per_second",
                "input_tokens",
                "output_tokens",
                "prompt",
                "response",
            ]
            existing_cols = [c for c in display_cols if c in runs.columns]
            st.dataframe(runs[existing_cols], use_container_width=True, height=320)

            st.plotly_chart(
                px.line(
                    runs.sort_values("id"),
                    x="id",
                    y="latency_seconds",
                    color="model",
                    title="Latency over runs",
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                px.bar(
                    runs,
                    x="model",
                    y="tokens_per_second",
                    title="Tokens/sec by model",
                ),
                use_container_width=True,
            )
        else:
            st.info("No inference runs yet. Use the Prompt Playground first.")

    if not evals_error:
        evals = pd.DataFrame(evals_data)

        if not evals.empty:
            st.markdown("### Evaluation Results")
            st.dataframe(evals, use_container_width=True, height=260)

            if "score" in evals.columns:
                st.plotly_chart(
                    px.box(evals, x="model", y="score", title="Eval score distribution"),
                    use_container_width=True,
                )

    if not rag_error:
        rag = pd.DataFrame(rag_data)

        if not rag.empty:
            st.markdown("### RAG Results")
            st.dataframe(rag, use_container_width=True, height=260)


with tab6:
    st.subheader("Specialist Model Fine-Tuning Lab")
    st.caption("Local LoRA fine-tuning workflow for a small specialist assistant.")

    st.markdown(
        """
This project currently uses **Ollama for real local inference**.

For actual fine-tuning, Ollama models are not the training target directly.

Recommended workflow:

1. Fine-tune the Hugging Face base model with LoRA.
2. Save the adapter.
3. Evaluate base vs fine-tuned model.
4. Optionally export/convert for local serving later.

Best starter target:

`Qwen/Qwen2.5-0.5B-Instruct`
"""
    )

    st.code(
        "python fine_tuning/validate_dataset.py data/sample_eval/engineering_train.jsonl\n"
        "python fine_tuning/train_lora.py "
        "--dataset data/sample_eval/engineering_train.jsonl "
        "--base_model Qwen/Qwen2.5-0.5B-Instruct "
        "--output_dir fine_tuning/outputs/qwen-engineering-lora\n"
        "python fine_tuning/compare_base_vs_adapter.py "
        "--adapter fine_tuning/outputs/qwen-engineering-lora",
        language="bash",
    )

    st.warning(
        "Fine-tuning on Windows can be painful. Use WSL2/Linux or a cloud GPU for training. "
        "The Ollama inference, benchmarking, evals and RAG tabs run locally now."
    )