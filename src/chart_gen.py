"""Chart generation module with sandboxed exec()."""
from __future__ import annotations
import base64, builtins as _builtins_module, io, json, os, re, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openai import OpenAI
from normalization import _is_text_dtype
from context import (
    detect_language, _select_relevant_columns, _detect_column_roles,
    _reduce_df_for_chart, CHART_MAX_ROWS,
)
from prompts import CHART_GENERATION_SYSTEM_PROMPT
from analysis import _track_tokens

_client = None

def _get_chart_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _client = OpenAI(api_key=api_key)
    return _client

# ---------------------------------------------------------------------------
# Sandbox builtins whitelist
# ---------------------------------------------------------------------------
_SAFE_BUILTINS = {
    'abs','all','any','bool','bytes','chr','dict','divmod','enumerate',
    'filter','float','format','frozenset','hash','int','isinstance',
    'issubclass','iter','len','list','map','max','min','next','oct',
    'ord','pow','print','range','repr','reversed','round','set',
    'slice','sorted','str','sum','tuple','type','zip',
}
_restricted_builtins = {
    name: getattr(_builtins_module, name) for name in _SAFE_BUILTINS
    if hasattr(_builtins_module, name)
}
_restricted_builtins['__build_class__'] = __builtins__.__build_class__ if hasattr(__builtins__, '__build_class__') else _builtins_module.__build_class__
_restricted_builtins['True'] = True
_restricted_builtins['False'] = False
_restricted_builtins['None'] = None

# ---------------------------------------------------------------------------
# Regex lint patterns
# ---------------------------------------------------------------------------
_RE_INFER_DATETIME_FMT = re.compile(r',\s*infer_datetime_format\s*=\s*(?:True|False)', re.IGNORECASE)
_RE_SQUEEZE_KWARG = re.compile(r',\s*squeeze\s*=\s*(?:True|False)', re.IGNORECASE)
_RE_DF_APPEND = re.compile(r'(\w+)\s*=\s*\1\.append\((.+?)\)')
_RE_IMPORT_PANDAS = re.compile(r'^\s*(?:import\s+pandas(?:\s+as\s+\w+)?|from\s+pandas\s+import\s+.+)\s*$', re.MULTILINE)
_RE_IMPORT_MATPLOTLIB = re.compile(r'^\s*(?:import\s+matplotlib(?:\.pyplot)?(?:\s+as\s+\w+)?|from\s+matplotlib(?:\.pyplot)?\s+import\s+.+)\s*$', re.MULTILINE)
_RE_IMPORT_SEABORN = re.compile(r'^\s*(?:import\s+seaborn(?:\s+as\s+\w+)?|from\s+seaborn\s+import\s+.+)\s*$', re.MULTILINE)
_RE_IMPORT_NUMPY = re.compile(r'^\s*(?:import\s+numpy(?:\s+as\s+\w+)?|from\s+numpy\s+import\s+.+)\s*$', re.MULTILINE)
_RE_SERIES_STRFTIME = re.compile(r'(?<!\.dt)\.strftime\(')


def _lint_generated_code(code: str) -> str:
    original = code
    code = _RE_INFER_DATETIME_FMT.sub('', code)
    code = _RE_SQUEEZE_KWARG.sub('', code)
    code = _RE_DF_APPEND.sub(r'\1 = pd.concat([\1, \2])', code)
    code = _RE_IMPORT_PANDAS.sub('# (import removed — pd already available)', code)
    code = _RE_IMPORT_MATPLOTLIB.sub('# (import removed — plt already available)', code)
    code = _RE_IMPORT_SEABORN.sub('# (import removed — sns already available)', code)
    code = _RE_IMPORT_NUMPY.sub('# (import removed — np already available)', code)
    code = _RE_SERIES_STRFTIME.sub('.dt.strftime(', code)
    if code != original:
        print("[chart-lint] Pre-exec lint applied fixes to generated code")
    return code


def _validate_column_references(code: str, df_columns: set[str]) -> list[str]:
    pattern = r"""df\[['"](.+?)['"]]\s*"""
    referenced = set(re.findall(pattern, code))
    missing = [col for col in referenced if col not in df_columns]
    if missing:
        print(f"[chart-validate] Hallucinated columns detected: {missing}")
    return missing


def _inspect_df_quality(df: pd.DataFrame) -> str:
    notes = []
    MAX_COLS = 120
    timedelta_cols = [col for col in df.columns if pd.api.types.is_timedelta64_dtype(df[col])]
    if timedelta_cols:
        notes.append(
            f"# TIMEDELTA COLUMNS: {timedelta_cols}\n"
            f"#   NEVER call pd.to_datetime() on them. Use df[col].dt.total_seconds()"
        )
    obj_cols = [c for c in df.columns if _is_text_dtype(df[c])][:MAX_COLS]
    comma_cols = []
    _comma_re = r'^-?\d+,\d+$'
    for col in obj_cols:
        sample = df[col].dropna().astype(str).str.strip().head(50)
        if sample.empty:
            continue
        if sample.str.match(_comma_re).sum() >= max(1, len(sample) * 0.3):
            comma_cols.append(col)
    if comma_cols:
        notes.append(
            f"# COMMA-DECIMAL COLUMNS: {comma_cols}\n"
            f"#   Fix: df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')"
        )
    num_cols = [col for col in df.select_dtypes(include='number').columns if not pd.api.types.is_timedelta64_dtype(df[col])][:MAX_COLS]
    sentinel_cols = []
    for col in num_cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(series) < 5:
            continue
        zero_mask = series == 0
        zero_ratio = zero_mask.sum() / len(series)
        non_zero = series[~zero_mask]
        if zero_ratio >= 0.15 and len(non_zero) > 0 and non_zero.median() > 10:
            sentinel_cols.append((col, round(zero_ratio * 100, 1)))
    if sentinel_cols:
        pct_list = [f"{c}: {p}% zeros" for c, p in sentinel_cols]
        col_list = [c for c, _ in sentinel_cols]
        notes.append(
            f"# ZERO-SENTINEL COLUMNS: {'; '.join(pct_list)}\n"
            f"#   Fix: for col in {col_list}: df[col] = df[col].where(df[col] > 0, other=float('nan'))"
        )
    _METRIC_KEYWORDS = {
        'card','cards','yellow','red','booking','foul','fouls',
        'goal','goals','gol','gols','score','scored','assist','assists',
        'shot','shots','attempt','attempts','chute','chutes',
        'count','total','sum','num','number','qty','quantity',
        'point','points','ponto','pontos','win','wins','loss','losses','draw','draws',
        'pass','passes','tackle','tackles','interception','interceptions',
        'save','saves','clean','concede','conceded',
        'km','distance','minute','minutes','min','rating','rate','ratio','pct','percent',
    }
    def _is_metric_col(col_name):
        tokens = set(re.split(r'[\s_\-/]+', str(col_name).lower()))
        return bool(tokens & _METRIC_KEYWORDS)

    agg_cols = []
    for col in num_cols:
        if _is_metric_col(col):
            continue
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        unique_vals = series.unique()
        if len(unique_vals) > 10:
            continue
        unique_sorted = sorted(unique_vals)
        if (0 in unique_sorted and all(v >= 0 and v == int(v) for v in unique_sorted)
                and max(unique_sorted) <= 20 and (series == 0).sum() > 0 and (series != 0).sum() > 0):
            agg_cols.append((col, [int(v) for v in unique_sorted]))
    if agg_cols:
        for col, vals in agg_cols:
            notes.append(
                f"# AGGREGATION-LEVEL COLUMN: '{col}' contains values {vals}. "
                f"Value 0 typically marks aggregate rows.\n"
                f"# Filter: df = df[pd.to_numeric(df['{col}'], errors='coerce') == 0].copy()\n"
                f"# NOTE: skip this filter if '{col}' is itself a metric the user wants to plot."
            )
    if not notes:
        return "# No significant data-quality issues detected — df appears clean."
    return "\n".join(notes)


def _preprocess_df_for_chart(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    MAX_COLS = 120
    _comma_re = r'^-?\d+,\d+$'
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
    obj_cols = [c for c in df.columns if _is_text_dtype(df[c])][:MAX_COLS]
    for col in obj_cols:
        sample = df[col].dropna().astype(str).str.strip().head(50)
        if sample.empty:
            continue
        if sample.str.match(_comma_re).sum() >= max(1, len(sample) * 0.3):
            df[col] = pd.to_numeric(df[col].astype(str).str.strip().str.replace(',', '.', regex=False), errors='coerce')
    return df


def generate_visualization_code_with_ai(df: pd.DataFrame, user_prompt: str, model: str = "gpt-4o") -> tuple:
    MAX_DETAIL_COLS = 50
    MAX_SAMPLE_ROWS = 3
    all_col_names = list(df.columns)
    relevant_cols = _select_relevant_columns(df, user_prompt, max_columns=MAX_DETAIL_COLS)
    roles = _detect_column_roles(df)
    identity_cols = roles["identity"]
    for col in identity_cols:
        if col not in relevant_cols and col in df.columns:
            relevant_cols.append(col)
    relevant_dtypes = {col: str(df.dtypes[col]) for col in relevant_cols}
    relevant_sample = df[relevant_cols].head(MAX_SAMPLE_ROWS).to_dict(orient="records")
    relevant_numeric = [col for col in relevant_cols if pd.api.types.is_numeric_dtype(df[col])]
    relevant_categorical = [col for col in relevant_cols if _is_text_dtype(df[col])]
    id_col_details = {}
    for col in identity_cols[:15]:
        sample_vals = df[col].dropna().unique()[:8]
        id_col_details[col] = {"dtype": str(df[col].dtype), "nunique": int(df[col].nunique()), "sample_values": [str(v) for v in sample_vals]}
    metric_cols_preview = roles["metric"][:30]
    entity_model_section = f"""
# ENTITY MODEL:
# Identity/grouping columns: {json.dumps(id_col_details, ensure_ascii=False, default=str)}
# Metric columns: {metric_cols_preview}  (showing first {len(metric_cols_preview)} of {len(roles['metric'])})
# CRITICAL: Use ONLY column names from the lists above.
# When user says 'player': {[c for c in identity_cols if 'name' in c.lower() or 'player' in c.lower() or 'jogador' in c.lower()][:3] or identity_cols[:2]}
# When user says 'game'/'match': {[c for c in identity_cols if any(k in c.lower() for k in ('period','event','jogo','match','adversário','adversario'))][:3] or identity_cols[:2]}
"""
    system_prompt = CHART_GENERATION_SYSTEM_PROMPT
    quality_notes = _inspect_df_quality(df)
    user_message = f"""
# Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns
# ALL column names: {all_col_names}
# Detailed schema ({len(relevant_cols)} relevant columns):
# - Dtypes: {relevant_dtypes}
# - Numeric: {relevant_numeric}
# - Categorical: {relevant_categorical}
# - Sample ({MAX_SAMPLE_ROWS} rows): {relevant_sample}
{entity_model_section}
# Data Quality Notes:
{quality_notes}

User Request: {user_prompt}

Generate Python code to create the visualization. The DataFrame is already available as 'df'.
"""
    if os.getenv("OPENAI_API_KEY"):
        try:
            response = _get_chart_client().chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
                max_completion_tokens=3500,
            )
            _track_tokens(response)
            raw = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            print(f"[chart-gen] finish_reason={finish_reason!r}  raw_len={len(raw)} chars")
            if finish_reason == "length":
                print("[chart-gen] WARNING: response truncated")
                return None, "Não foi possível gerar o gráfico: o código gerado foi cortado antes de terminar."
            code = raw
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            code = code.strip()
            if not code:
                print(f"[chart-gen] ERROR: empty code. Raw response:\n{raw[:500]}")
                return None, "Não foi possível gerar o gráfico: a resposta do modelo não continha código válido."
            return code, None
        except Exception as e:
            msg = str(e)
            print(f"[chart-gen] ERROR calling OpenAI: {msg}")
            return None, f"Erro ao chamar a API de geração de gráfico: {msg}"
    else:
        print("[chart-gen] No OpenAI API key found")
        return None, "Chave da API OpenAI não configurada."


def execute_visualization_code(df: pd.DataFrame, code: str, output_path: str = "output_chart.png") -> tuple[bool, str | None]:
    """Execute AI-generated visualization code with sandboxed builtins."""
    _REMOVED_TO_DATETIME_KWARGS = {"infer_datetime_format"}
    _real_to_datetime = pd.to_datetime
    _real_read_csv = pd.read_csv

    def _compat_to_datetime(*args, **kwargs):
        for k in _REMOVED_TO_DATETIME_KWARGS:
            kwargs.pop(k, None)
        return _real_to_datetime(*args, **kwargs)

    def _compat_read_csv(*args, **kwargs):
        kwargs.pop("squeeze", None)
        return _real_read_csv(*args, **kwargs)

    code = _lint_generated_code(code)
    pd.to_datetime = _compat_to_datetime
    pd.read_csv = _compat_read_csv
    try:
        clean_df = _preprocess_df_for_chart(df)
        exec_globals = {
            '__builtins__': _restricted_builtins,
            'df': clean_df, 'pd': pd, 'plt': plt, 'sns': sns,
            'np': np, 'io': io, 'base64': base64,
        }
        exec(code, exec_globals)
        if os.path.exists(output_path):
            return True, None
        else:
            msg = f"Código executado mas '{output_path}' não foi criado."
            print(f"[chart-exec] {msg}")
            return False, msg
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        msg = f"{type(e).__name__}: {str(e)}"
        print(f"[chart-exec] ERROR: {msg}\n{tb}")
        return False, f"Erro ao executar o gráfico: {msg}"
    finally:
        pd.to_datetime = _real_to_datetime
        pd.read_csv = _real_read_csv


def generate_chart_from_prompt(df: pd.DataFrame, prompt: str, analysis_response: str = "", model: str = "gpt-4o") -> tuple[str | None, str | None, str | None]:
    MAX_ATTEMPTS = 2
    output_path = "output_chart.png"
    last_error: str | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt == 1:
            print("Attempting AI-generated visualization...")
            viz_code, gen_error = generate_visualization_code_with_ai(df, prompt, model)
        else:
            retry_prompt = (
                f"{prompt}\n\n[RETRY] Previous error:\n{last_error}\n"
                f"Fix the code. Remember: pandas >= 3.0, no deprecated APIs."
            )
            print(f"[chart-gen] Retry attempt {attempt}")
            viz_code, gen_error = generate_visualization_code_with_ai(df, retry_prompt, model)
        if gen_error:
            print(f"[chart-gen] Code generation failed (attempt {attempt}): {gen_error}")
            last_error = gen_error
            continue
        print(f"Generated visualization code (attempt {attempt}):")
        print(viz_code)
        print("-" * 50)
        missing_cols = _validate_column_references(viz_code, set(df.columns))
        if missing_cols:
            roles = _detect_column_roles(df)
            last_error = (
                f"Code references non-existent columns: {missing_cols}. "
                f"Identity columns: {roles['identity'][:10]}. "
                f"Metric columns: {roles['metric'][:10]}."
            )
            print(f"[chart-gen] Column validation failed (attempt {attempt}): {last_error}")
            continue
        if os.path.exists(output_path):
            os.remove(output_path)
        success, exec_error = execute_visualization_code(df, viz_code, output_path)
        if success:
            try:
                with open(output_path, 'rb') as img_file:
                    chart_base64 = base64.b64encode(img_file.read()).decode()
                os.remove(output_path)
                if attempt > 1:
                    print(f"AI-generated visualization succeeded on retry (attempt {attempt})!")
                else:
                    print("AI-generated visualization successful!")
                return chart_base64, None, viz_code
            except Exception as e:
                msg = f"Erro ao ler o gráfico gerado: {str(e)}"
                print(f"[chart-gen] {msg}")
                return None, msg, None
        last_error = exec_error or "Erro desconhecido ao executar o código de visualização."
        print(f"[chart-gen] Execution failed (attempt {attempt}): {last_error}")
    return None, last_error or "Erro desconhecido ao executar o código de visualização.", None


def generate_chart_fallback(df: pd.DataFrame, prompt: str, analysis_response: str = "") -> str:
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        prompt_lower = prompt.lower()
        lang = detect_language(prompt)
        chart_type = None
        if any(w in prompt_lower for w in ['bar chart','bar graph','barra','barras']):
            chart_type = 'bar'
        elif any(w in prompt_lower for w in ['line chart','line graph','linha','linhas','trend','tendência']):
            chart_type = 'line'
        elif any(w in prompt_lower for w in ['pie chart','pie graph','pizza','torta']):
            chart_type = 'pie'
        elif any(w in prompt_lower for w in ['scatter','dispersão','scatter plot']):
            chart_type = 'scatter'
        elif any(w in prompt_lower for w in ['histogram','histograma','distribution','distribuição']):
            chart_type = 'histogram'
        elif any(w in prompt_lower for w in ['heatmap','heat map','mapa de calor','correlação','correlation']):
            chart_type = 'heatmap'
        elif any(w in prompt_lower for w in ['box plot','boxplot','box','caixa']):
            chart_type = 'box'
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object','string','category']).columns
        if chart_type == 'heatmap' and len(numeric_cols) > 1:
            plt.close(); fig, ax = plt.subplots(figsize=(10, 8))
            correlation = df[numeric_cols].corr()
            sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap' if lang == 'en' else 'Mapa de Correlação', fontsize=14, fontweight='bold')
        elif chart_type == 'pie':
            if len(categorical_cols) > 0:
                col = categorical_cols[0]; data = df[col].value_counts().head(10)
                ax.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90)
                ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
            elif len(numeric_cols) > 0:
                col = numeric_cols[0]
                ax.pie(df[col].head(10), labels=df.index[:10], autopct='%1.1f%%')
                ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
        elif chart_type == 'histogram' and len(numeric_cols) > 0:
            col = numeric_cols[0]
            ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel(col, fontsize=12); ax.set_ylabel('Frequency' if lang == 'en' else 'Frequência', fontsize=12)
            ax.set_title(f'Distribution of {col}' if lang == 'en' else f'Distribuição de {col}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        elif chart_type == 'scatter' and len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
            ax.set_xlabel(x_col, fontsize=12); ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold'); ax.grid(True, alpha=0.3)
        elif chart_type == 'box' and len(numeric_cols) > 0:
            df[numeric_cols[:5]].boxplot(ax=ax)
            ax.set_ylabel('Value' if lang == 'en' else 'Valor', fontsize=12)
            ax.set_title('Box Plot', fontsize=14, fontweight='bold'); ax.grid(True, alpha=0.3)
        elif chart_type == 'line' and len(numeric_cols) > 0:
            col = numeric_cols[0]; df[col].plot(kind='line', ax=ax, linewidth=2, marker='o')
            ax.set_ylabel(col, fontsize=12); ax.set_xlabel('Index' if lang == 'en' else 'Índice', fontsize=12)
            ax.set_title(f'{col} Trend' if lang == 'en' else f'Tendência de {col}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        elif any(w in prompt_lower for w in ['top','scorer','highest','best','most','maior','melhor','artilheiro']):
            if len(numeric_cols) > 0:
                target_col = None
                for col in df.columns:
                    if any(w in col.lower() for w in ['goal','score','point','gol','ponto','valor','value']):
                        target_col = col; break
                if not target_col:
                    target_col = numeric_cols[0]
                name_cols = [c for c in df.columns if any(w in c.lower() for w in ['name','player','team','nome','jogador','time','equipe'])]
                if name_cols:
                    top_data = df.nlargest(15, target_col)
                    bars = ax.barh(range(len(top_data)), top_data[target_col])
                    ax.set_yticks(range(len(top_data))); ax.set_yticklabels(top_data[name_cols[0]], fontsize=10)
                    ax.set_xlabel(target_col, fontsize=12)
                    ax.set_title(f'Top 15 by {target_col}' if lang == 'en' else f'Top 15 por {target_col}', fontsize=14, fontweight='bold')
                    ax.invert_yaxis()
                    for i, (idx, bar) in enumerate(zip(top_data.index, bars)):
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}', ha='left', va='center', fontsize=9)
                else:
                    top_data = df.nlargest(15, target_col)
                    ax.bar(range(len(top_data)), top_data[target_col])
                    ax.set_xlabel('Rank' if lang == 'en' else 'Posição', fontsize=12)
                    ax.set_ylabel(target_col, fontsize=12)
                    ax.set_title(f'Top 15 {target_col}', fontsize=14, fontweight='bold')
        else:
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                cat_col = categorical_cols[0]; num_col = numeric_cols[0]
                grouped = df.groupby(cat_col)[num_col].mean().nlargest(15)
                grouped.plot(kind='barh', ax=ax)
                ax.set_xlabel(num_col, fontsize=12); ax.set_ylabel(cat_col, fontsize=12)
                ax.set_title(f'Average {num_col} by {cat_col}' if lang == 'en' else f'Média de {num_col} por {cat_col}', fontsize=14, fontweight='bold')
            elif len(numeric_cols) > 0:
                col = numeric_cols[0]; df[col].head(20).plot(kind='bar', ax=ax)
                ax.set_ylabel(col, fontsize=12); ax.set_xlabel('Index' if lang == 'en' else 'Índice', fontsize=12)
                ax.set_title(f'{col} Values', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        chart_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close('all')
        return chart_base64
    except Exception as e:
        print(f"Error generating fallback chart: {str(e)}")
        plt.close('all')
        return None
