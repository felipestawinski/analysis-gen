"""
FastAPI application — route handlers and request models only.
All business logic lives in the sibling modules.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl
import uvicorn
import httpx
import asyncio
import io
import pandas as pd
import os
import time
from dotenv import load_dotenv
from typing import List, Optional

from normalization import detect_file_type, dataframe_from_bytes, dataframe_preview, _is_text_dtype

# Load environment variables BEFORE imports that read env vars at module level
load_dotenv()

from rag import rag_store
from cache import (
    get_or_download_dataframe, get_or_download_file_bytes,
    _get_cached_dataframe,
    MAX_FILES_PER_ANALYSIS_REQUEST, MAX_PARALLEL_DOWNLOADS,
)
from context import (
    is_large_dataset, _build_supplementary_context, _reduce_df_for_chart,
    CHART_MAX_ROWS,
)
from analysis import (
    analyze_dataframe_with_openai, reset_request_tokens, get_request_tokens,
)
from chart_gen import generate_chart_from_prompt, generate_chart_fallback
from data_quality import perform_data_health_check, clean_dataframe

app = FastAPI()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatHistoryMessage(BaseModel):
    role: str
    content: str

class URLRequest(BaseModel):
    fileAddresses: List[HttpUrl]
    fileTypes: Optional[List[str]] = None
    prompt: str
    generateChart: bool = False
    chartRecommendation: bool = False
    chatId: Optional[str] = None
    forceRefresh: bool = False
    model: str = "gpt-4o"
    chatHistory: Optional[List[ChatHistoryMessage]] = None

class PreviewRequest(BaseModel):
    fileAddress: HttpUrl
    fileType: Optional[str] = None
    maxRows: int = 20
    maxCols: int = 12
    forceRefresh: bool = False

class PreloadRequest(BaseModel):
    fileAddress: str
    fileType: Optional[str] = None


# ---------------------------------------------------------------------------
# /preload-file
# ---------------------------------------------------------------------------

@app.post("/preload-file")
async def preload_file(request: PreloadRequest):
    try:
        file_address = request.fileAddress.strip()
        file_type = detect_file_type(
            explicit_type=request.fileType,
            filename=file_address,
        )
        async with httpx.AsyncClient() as client_http:
            df = await get_or_download_dataframe(
                client_http, file_address, file_type, force_refresh=False,
            )
        rows, cols = df.shape
        print(f"Preload complete: {file_address} -> shape=({rows}, {cols})")
        try:
            if rag_store.enabled:
                rag_store.index_dataframe(file_address, df)
        except Exception as rag_err:
            print(f"[RAG] WARNING: preload indexing failed (non-fatal): {rag_err}")
        return {"status": "ok", "shape": [rows, cols]}
    except Exception as e:
        print(f"Preload failed for {request.fileAddress}: {e}")
        raise HTTPException(status_code=500, detail=f"Preload failed: {str(e)}")


# ---------------------------------------------------------------------------
# /data-health-check
# ---------------------------------------------------------------------------

@app.post("/data-health-check", response_class=PlainTextResponse)
async def data_health_check(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_type = detect_file_type(filename=file.filename, content_type=file.content_type)
        df = dataframe_from_bytes(contents, file_type)
        report = perform_data_health_check(df)
        return report
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty or not a valid tabular file.")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="The file could not be decoded as UTF-8.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analysing file: {str(e)}")


# ---------------------------------------------------------------------------
# /data-health-check-clean
# ---------------------------------------------------------------------------

@app.post("/data-health-check-clean")
async def data_health_check_clean(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_type = detect_file_type(filename=file.filename, content_type=file.content_type)
        df = dataframe_from_bytes(contents, file_type)
        cleaned_df = clean_dataframe(df)
        output = io.StringIO()
        cleaned_df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=cleaned_{file.filename}"},
        )
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty or not a valid tabular file.")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="The file could not be decoded as UTF-8.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning file: {str(e)}")


# ---------------------------------------------------------------------------
# /preview-gen
# ---------------------------------------------------------------------------

@app.post("/preview-gen")
async def preview_gen(request: PreviewRequest):
    try:
        file_type = detect_file_type(explicit_type=request.fileType, filename=str(request.fileAddress))
        async with httpx.AsyncClient(timeout=90.0) as client_http:
            file_bytes = await get_or_download_file_bytes(
                client_http, str(request.fileAddress), file_type, request.forceRefresh,
            )
        df = dataframe_from_bytes(file_bytes, file_type)
        headers, rows = dataframe_preview(df, request.maxRows, request.maxCols)
        return {"headers": headers, "rows": rows, "fileType": file_type}
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Error downloading file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating preview: {str(e)}")


# ---------------------------------------------------------------------------
# /analysis-gen  (main endpoint)
# ---------------------------------------------------------------------------

@app.post("/analysis-gen")
async def download_csv(request: URLRequest):
    reset_request_tokens()
    try:
        started_at = time.perf_counter()
        print(f"User prompt: {request.prompt}")
        print(f"Number of files to process: {len(request.fileAddresses)}")
        print(f"Generate chart requested: {request.generateChart}")
        print(f"Chart recommendation requested: {request.chartRecommendation}")
        print(f"Chat ID: {request.chatId}")
        print(f"Force refresh: {request.forceRefresh}")

        if len(request.fileAddresses) > MAX_FILES_PER_ANALYSIS_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Máximo de {MAX_FILES_PER_ANALYSIS_REQUEST} arquivos por análise. Reduza a seleção e tente novamente.",
            )

        resolved_model = request.model.strip() if request.model and request.model.strip() else "gpt-4o"
        request.model = resolved_model

        async def load_single_dataframe(client_http, idx, file_address, semaphore):
            async with semaphore:
                incoming_file_type = (
                    request.fileTypes[idx]
                    if request.fileTypes and idx < len(request.fileTypes)
                    else None
                )
                resolved_file_type = detect_file_type(
                    explicit_type=incoming_file_type,
                    filename=str(file_address),
                )
                print(f"Processing file {idx + 1}/{len(request.fileAddresses)}: {file_address}")
                if not request.forceRefresh:
                    cached_df = _get_cached_dataframe(str(file_address).strip())
                    if cached_df is not None:
                        print(f"DF cache hit for file {idx + 1}, skipping download+parse.")
                        return idx, cached_df, resolved_file_type
                df_local = await get_or_download_dataframe(
                    client_http, str(file_address), resolved_file_type, request.forceRefresh,
                )
                print(f"Loaded DataFrame {idx + 1} with shape: {df_local.shape}")
                print(f"Columns: {list(df_local.columns[:10])}")
                return idx, df_local, resolved_file_type

        download_started_at = time.perf_counter()
        async with httpx.AsyncClient() as client_http:
            semaphore = asyncio.Semaphore(max(1, MAX_PARALLEL_DOWNLOADS))
            tasks = [
                load_single_dataframe(client_http, idx, file_address, semaphore)
                for idx, file_address in enumerate(request.fileAddresses)
            ]
            loaded_results = await asyncio.gather(*tasks)

        loaded_results.sort(key=lambda item: item[0])
        dataframes = [item[1] for item in loaded_results]
        processed_file_types = [item[2] for item in loaded_results]
        print(f"Download + parse stage latency: {time.perf_counter() - download_started_at:.2f}s")

        combine_started_at = time.perf_counter()
        if len(dataframes) == 1:
            combined_df = dataframes[0]
        else:
            try:
                combined_df = pd.concat(dataframes, ignore_index=True)
                print(f"Combined {len(dataframes)} dataframes vertically")
            except Exception as e:
                print(f"Could not concatenate vertically: {e}. Attempting horizontal merge...")
                combined_df = dataframes[0]
                for df in dataframes[1:]:
                    combined_df = pd.merge(combined_df, df, left_index=True, right_index=True, how='outer')
                print(f"Combined {len(dataframes)} dataframes horizontally")
        print(f"Combine stage latency: {time.perf_counter() - combine_started_at:.2f}s")

        print(f"Final combined DataFrame shape: {combined_df.shape}")
        print(f"Final columns: {list(combined_df.columns[:10])}")
        large_data_mode = is_large_dataset(combined_df)
        print(f"Large dataset mode: {large_data_mode}")

        # RAG indexing + retrieval
        rag_file_key = "|".join(sorted(str(a) for a in request.fileAddresses))
        retrieved_context = ""
        try:
            if rag_store.enabled:
                rag_started_at = time.perf_counter()
                if request.forceRefresh:
                    rag_store.invalidate(rag_file_key)
                rag_store.index_dataframe(rag_file_key, combined_df)
                print(f"RAG index stage latency: {time.perf_counter() - rag_started_at:.2f}s")
            if rag_store.enabled and not request.chartRecommendation:
                rag_retrieve_started_at = time.perf_counter()
                top_k = 20 if request.generateChart else 30
                retrieved_context = rag_store.retrieve(rag_file_key, request.prompt, top_k=top_k)
                print(f"RAG retrieval latency: {time.perf_counter() - rag_retrieve_started_at:.2f}s")
        except Exception as rag_err:
            print(f"[RAG] WARNING: RAG failed: {rag_err}")
            retrieved_context = ""

        # Supplementary context
        try:
            supplementary = _build_supplementary_context(combined_df, request.prompt, retrieved_context)
            if supplementary:
                sep = "\n\n--- Targeted data for columns mentioned in your query ---\n"
                retrieved_context = (
                    (retrieved_context + sep + supplementary) if retrieved_context else supplementary
                )
                print(f"[SUP] Supplementary context: {len(supplementary)} chars")
        except Exception as sup_err:
            print(f"[SUP] WARNING: supplementary extraction failed: {sup_err}")

        analysis_started_at = time.perf_counter()
        if request.generateChart:
            analysis_text = ""
        else:
            print("Generating streaming text analysis...")
            history_dicts = (
                [{"role": m.role, "content": m.content} for m in request.chatHistory]
                if request.chatHistory else None
            )
            text_generator = analyze_dataframe_with_openai(
                combined_df, request.prompt, request.model, stream=True,
                retrieved_context=retrieved_context, chat_history=history_dicts,
            )
            print(f"Analysis stage latency (stream started): {time.perf_counter() - analysis_started_at:.2f}s")
            return StreamingResponse(text_generator, media_type="text/plain")

        print(f"Analysis stage latency: {time.perf_counter() - analysis_started_at:.2f}s")

        chart_base64 = None
        chart_code = None
        response_message = None
        if request.generateChart:
            chart_started_at = time.perf_counter()
            chart_df = _reduce_df_for_chart(combined_df, request.prompt)
            if chart_df.shape[0] > CHART_MAX_ROWS:
                response_message = (
                    "O dataset possui muitas linhas para gerar o gráfico. "
                    "Refine o filtro para reduzir os dados."
                )
                print(f"Chart generation deferred: {chart_df.shape[0]} rows exceeds CHART_MAX_ROWS={CHART_MAX_ROWS}.")
            else:
                print(f"Generating chart (explicit request) with shape {chart_df.shape}...")
                chart_base64, chart_error, chart_code = generate_chart_from_prompt(
                    chart_df, request.prompt, "", request.model
                )
                if chart_error:
                    response_message = chart_error
            print(f"Chart stage latency: {time.perf_counter() - chart_started_at:.2f}s")

        data_summary = {
            "rows": len(combined_df),
            "columns": len(combined_df.columns),
            "column_names": list(combined_df.columns),
            "numeric_columns": list(combined_df.select_dtypes(include=['number']).columns),
            "categorical_columns": [c for c in combined_df.columns if _is_text_dtype(combined_df[c])],
            "files_processed": len(request.fileAddresses),
            "file_types": processed_file_types,
            "visualization_generated": chart_base64 is not None,
            "chart_deferred": bool(request.generateChart and chart_base64 is None and response_message is not None),
            "large_dataset_mode": large_data_mode,
        }
        print(f"Total /analysis-gen latency: {time.perf_counter() - started_at:.2f}s")

        total_tokens_used = get_request_tokens()
        print(f"DEBUG_TOKENS: Total tokens consumed in this request: {total_tokens_used}")

        return {
            "text_response": analysis_text,
            "chart_base64": chart_base64,
            "chart_code": chart_code,
            "data_summary": data_summary,
            "message": response_message,
            "tokens_used": total_tokens_used,
        }

    except httpx.RequestError as e:
        print('Error downloading file')
        raise HTTPException(status_code=400, detail=f"Error downloading file: {str(e)}")
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9090)