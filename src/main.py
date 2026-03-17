from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, HttpUrl
import uvicorn
import httpx
import csv
import io
import matplotlib.pyplot as plt
import pandas as pd
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
from typing import List
import seaborn as sns

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class URLRequest(BaseModel):
    fileAddresses: List[HttpUrl]
    prompt: str


def perform_data_health_check(df: pd.DataFrame) -> str:
    """
    Analyze the health of a DataFrame and return a plain-text summary
    covering missing values, duplicates, data types, outliers, and basic stats.
    """
    lines: list[str] = []
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols

    # --- General overview ---
    lines.append("=== DATA HEALTH CHECK REPORT ===\n")
    lines.append(f"Rows: {total_rows}")
    lines.append(f"Columns: {total_cols}")
    lines.append(f"Total cells: {total_cells}\n")

    # --- Missing / null values ---
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    missing_pct = (total_missing / total_cells * 100) if total_cells else 0

    lines.append("--- Missing Values ---")
    lines.append(f"Total missing cells: {total_missing} ({missing_pct:.2f}%)")

    cols_with_missing = missing[missing > 0]
    if cols_with_missing.empty:
        lines.append("No columns with missing values.\n")
    else:
        lines.append(f"Columns affected: {len(cols_with_missing)}/{total_cols}")
        for col, count in cols_with_missing.items():
            pct = count / total_rows * 100
            lines.append(f"  • {col}: {count} missing ({pct:.1f}%)")
        lines.append("")

    # --- Duplicate rows ---
    dup_count = int(df.duplicated().sum())
    dup_pct = (dup_count / total_rows * 100) if total_rows else 0
    lines.append("--- Duplicate Rows ---")
    lines.append(f"Duplicate rows: {dup_count} ({dup_pct:.2f}%)\n")

    # --- Data types ---
    lines.append("--- Column Data Types ---")
    for col in df.columns:
        lines.append(f"  • {col}: {df[col].dtype}")
    lines.append("")

    # --- Numeric column stats & outlier detection ---
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        lines.append("--- Numeric Column Statistics ---")
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                lines.append(f"  {col}: all values are missing")
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())

            lines.append(f"  {col}:")
            lines.append(f"    min={series.min()}, max={series.max()}, "
                         f"mean={series.mean():.2f}, median={series.median():.2f}")
            lines.append(f"    outliers (IQR method): {outlier_count}")
        lines.append("")

    # --- Overall health score (simple heuristic) ---
    issues = 0
    if missing_pct > 5:
        issues += 1
    if dup_pct > 5:
        issues += 1
    if any(cols_with_missing.get(c, 0) / total_rows > 0.3 for c in df.columns):
        issues += 1

    if issues == 0:
        verdict = "GOOD – The dataset looks healthy."
    elif issues == 1:
        verdict = "FAIR – Minor quality issues detected."
    else:
        verdict = "POOR – Significant quality issues found. Review the details above."

    lines.append(f"Overall health verdict: {verdict}")

    return "\n".join(lines)


@app.post("/data-health-check", response_class=PlainTextResponse)
async def data_health_check(file: UploadFile = File(...)):
    """
    Receive a CSV file, analyse its data quality, and return a plain-text report.
    Called by the Go backend (sendToDataHealthCheck in upload-ipfs.go).
    """
    try:
        contents = await file.read()
        csv_text = contents.decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_text))

        report = perform_data_health_check(df)
        return report

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty or not a valid CSV.")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="The file could not be decoded as UTF-8.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analysing file: {str(e)}")


def detect_language(text: str) -> str:
    """
    Simple language detection based on common Portuguese words
    """
    portuguese_indicators = [
        'qual', 'quem', 'onde', 'quando', 'como', 'por que', 'porque',
        'jogador', 'melhor', 'maior', 'menor', 'média', 'total',
        'gols', 'pontos', 'time', 'equipe', 'artilheiro', 'temporada'
    ]
    
    text_lower = text.lower()
    portuguese_count = sum(1 for word in portuguese_indicators if word in text_lower)
    
    return 'pt' if portuguese_count > 0 else 'en'

def is_visualization_request(prompt: str) -> bool:
    """
    Detect if the user is asking for a visualization/chart/graph
    """
    visualization_keywords = [
        # English
        'visualize', 'visualization', 'chart', 'graph', 'plot', 'show me',
        'display', 'draw', 'create a chart', 'create a graph', 'bar chart',
        'line chart', 'pie chart', 'scatter plot', 'histogram', 'heatmap',
        # Portuguese
        'visualizar', 'visualização', 'gráfico', 'plotar', 'mostre',
        'exibir', 'desenhar', 'criar gráfico', 'criar um gráfico',
        'gráfico de barras', 'gráfico de linhas', 'gráfico de pizza',
        'dispersão', 'histograma', 'mapa de calor',
        # Spanish
        'visualizar', 'visualización', 'gráfica', 'mostrar', 'dibujar'
    ]
    
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in visualization_keywords)

def analyze_dataframe_with_openai(df: pd.DataFrame, user_prompt: str) -> str:
    """
    Send dataframe info and user prompt to OpenAI for analysis with fallback options
    """
    # Create a summary of the dataframe
    df_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "sample_data": df.to_dict(),
        "basic_stats": df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else None
    }
    
    system_prompt = """You are a data analyst expert. Analyze the provided CSV data and answer the user's question.
    IMPORTANT: Always respond in the same language as the user's question. If the user asks in Portuguese, respond in Portuguese. If in English, respond in English. If in Spanish, respond in Spanish, etc.
    Provide clear, concise insights based on the data."""
    
    user_message = f"""
    Dataset Information:
    - Shape: {df_info['shape']} (rows, columns)
    - Columns: {df_info['columns']}
    - Data types: {df_info['dtypes']}
    - Sample data: {df_info['sample_data']}
    - Basic statistics: {df_info['basic_stats']}
    
    User Question: {user_prompt}
    
    Please provide a detailed analysis answering the user's question in the same language as the question above.
    """
    
    # Try OpenAI first
    if os.getenv("OPENAI_API_KEY"):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI failed: {str(e)}")
            # Fall through to alternatives
    
    # Try Hugging Face as backup
    if os.getenv("HUGGINGFACE_API_KEY"):
        hf_result = analyze_dataframe_with_huggingface(df, user_prompt)
        if not hf_result.startswith("Error") and not hf_result.startswith("Hugging Face API error"):
            return hf_result
    
    # Use fallback analysis
    return analyze_dataframe_fallback(df, user_prompt)

def analyze_dataframe_with_huggingface(df: pd.DataFrame, user_prompt: str) -> str:
    """
    Use Hugging Face's free API as an alternative to OpenAI
    """
    # Create a summary of the dataframe
    df_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "sample_data": df.head(3).to_dict(),  # Reduced sample to save tokens
    }
    
    # Detect language and adjust prompt
    lang = detect_language(user_prompt)
    
    if lang == 'pt':
        prompt = f"""
        Analise estes dados CSV e responda à pergunta:
        
        Informações dos dados: {df.shape[0]} linhas, {df.shape[1]} colunas
        Colunas: {list(df.columns)}
        Amostra: {df_info['sample_data']}
        
        Pergunta: {user_prompt}
        
        Forneça uma análise clara com descobertas específicas em português.
        """
    else:
        prompt = f"""
        Analyze this CSV data and answer the question:
        
        Data info: {df.shape[0]} rows, {df.shape[1]} columns
        Columns: {list(df.columns)}
        Sample: {df_info['sample_data']}
        
        Question: {user_prompt}
        
        Provide a clear analysis with specific findings.
        """
    
    try:
        # Using Hugging Face's free inference API
        API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY', '')}"}
        
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        
        if response.status_code == 200:
            result = response.json()
            return result[0]['generated_text'] if result else "No response generated"
        else:
            return f"Hugging Face API error: {response.status_code}"
            
    except Exception as e:
        return f"Error with Hugging Face API: {str(e)}"

def analyze_dataframe_fallback(df: pd.DataFrame, user_prompt: str) -> str:
    """
    Fallback analysis without external APIs - purely based on pandas operations
    """
    lang = detect_language(user_prompt)
    analysis = []
    
    # Basic info in detected language
    if lang == 'pt':
        analysis.append(f"O conjunto de dados contém {len(df)} linhas e {len(df.columns)} colunas.")
        analysis.append(f"Colunas: {', '.join(df.columns)}")
    else:
        analysis.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.")
        analysis.append(f"Columns: {', '.join(df.columns)}")
    
    # Look for common patterns in the prompt
    prompt_lower = user_prompt.lower()
    
    # Portuguese and English keywords for "top/highest"
    top_keywords = ['top', 'highest', 'best', 'most', 'maior', 'melhor', 'artilheiro', 'líder']
    avg_keywords = ['average', 'mean', 'média', 'meio']
    total_keywords = ['total', 'sum', 'soma', 'somatório']
    
    if any(word in prompt_lower for word in top_keywords):
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if any(keyword in col.lower() for keyword in ['score', 'goal', 'point', 'value', 'gol', 'ponto', 'valor']):
                    top_value = df[col].max()
                    top_index = df[col].idxmax()
                    
                    if lang == 'pt':
                        analysis.append(f"Maior {col}: {top_value}")
                    else:
                        analysis.append(f"Highest {col}: {top_value}")
                    
                    # Try to find name column
                    name_cols = [c for c in df.columns if any(word in c.lower() for word in ['name', 'player', 'team', 'nome', 'jogador', 'time'])]
                    if name_cols:
                        top_name = df.loc[top_index, name_cols[0]]
                        if lang == 'pt':
                            analysis.append(f"Melhor desempenho: {top_name} com {top_value} {col}")
                        else:
                            analysis.append(f"Top performer: {top_name} with {top_value} {col}")
                    break
    
    elif any(word in prompt_lower for word in avg_keywords):
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            avg_val = df[col].mean()
            if lang == 'pt':
                analysis.append(f"Média de {col}: {avg_val:.2f}")
            else:
                analysis.append(f"Average {col}: {avg_val:.2f}")
    
    elif any(word in prompt_lower for word in total_keywords):
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            total_val = df[col].sum()
            if lang == 'pt':
                analysis.append(f"Total de {col}: {total_val}")
            else:
                analysis.append(f"Total {col}: {total_val}")
    
    # Add basic statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        if lang == 'pt':
            analysis.append("\nEstatísticas Básicas:")
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                analysis.append(f"{col}: mín={df[col].min()}, máx={df[col].max()}, média={df[col].mean():.2f}")
        else:
            analysis.append("\nBasic Statistics:")
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                analysis.append(f"{col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}")
    
    if lang == 'pt':
        return "\n".join(analysis) if analysis else "Não foi possível analisar os dados para esta pergunta específica."
    else:
        return "\n".join(analysis) if analysis else "Unable to analyze the data for this specific question."

def generate_visualization_code_with_ai(df: pd.DataFrame, user_prompt: str) -> str:
    """
    Use AI to generate Python code for visualization based on user prompt
    Returns only the Python code as a string
    """
    # Create a summary of the dataframe
    df_summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_data": df.head(3).to_dict(),
        "numeric_columns": list(df.select_dtypes(include=['number']).columns),
        "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns)
    }
    
    system_prompt = """You are a Python data visualization expert. Generate ONLY executable Python code to create a visualization based on the user's request.

IMPORTANT RULES:
1. Return ONLY Python code, no explanations, no markdown, no comments except necessary ones
2. The DataFrame is already loaded as 'df' - do NOT reload it
3. Use matplotlib and/or seaborn for visualizations
4. The code must save the figure to 'output_chart.png' using plt.savefig('output_chart.png', dpi=150, bbox_inches='tight')
5. Always include plt.close() at the end
6. Use proper labels, titles, and formatting
7. Handle any potential errors (missing data, etc.)
8. Make the visualization clear and professional
9. If the user's language is Portuguese, use Portuguese labels; if English, use English labels

Example output format:
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))
# Your visualization code here
plt.title('Chart Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.savefig('output_chart.png', dpi=150, bbox_inches='tight')
plt.close()
```"""
    
    user_message = f"""
Dataset Information:
- Shape: {df_summary['shape']} (rows, columns)
- Columns: {df_summary['columns']}
- Data types: {df_summary['dtypes']}
- Numeric columns: {df_summary['numeric_columns']}
- Categorical columns: {df_summary['categorical_columns']}
- Sample data (first 3 rows): {df_summary['sample_data']}

User Request: {user_prompt}

Generate Python code to create the visualization. Remember: the DataFrame is already available as 'df'.
"""
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1500,
                temperature=0.1
            )
            code = response.choices[0].message.content
            
            # Clean up the code - remove markdown code blocks if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            
            return code.strip()
        except Exception as e:
            print(f"Error generating visualization code with AI: {str(e)}")
            return None
    else:
        print("No OpenAI API key found")
        return None

def execute_visualization_code(df: pd.DataFrame, code: str, output_path: str = "output_chart.png") -> bool:
    """
    Execute the AI-generated visualization code safely
    Returns True if successful, False otherwise
    """
    try:
        # Create a safe execution environment with necessary imports
        exec_globals = {
            'df': df,
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'np': __import__('numpy'),
            'io': io,
            'base64': base64
        }
        
        # Execute the code
        exec(code, exec_globals)
        
        # Check if the output file was created
        if os.path.exists(output_path):
            return True
        else:
            print(f"Visualization code executed but {output_path} was not created")
            return False
            
    except Exception as e:
        print(f"Error executing visualization code: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_chart_from_prompt(df: pd.DataFrame, prompt: str, analysis_response: str = "") -> str:
    """
    Generate a chart based on the prompt using AI-generated code
    Falls back to rule-based generation if AI fails
    """
    # First, try AI-generated visualization
    print("Attempting AI-generated visualization...")
    viz_code = generate_visualization_code_with_ai(df, prompt)
    
    if viz_code:
        print("Generated visualization code:")
        print(viz_code)
        print("-" * 50)
        
        output_path = "output_chart.png"
        
        # Clean up any existing output file
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Execute the generated code
        if execute_visualization_code(df, viz_code, output_path):
            try:
                # Read the generated image and convert to base64
                with open(output_path, 'rb') as img_file:
                    chart_base64 = base64.b64encode(img_file.read()).decode()
                
                # Clean up the file
                os.remove(output_path)
                
                print("AI-generated visualization successful!")
                return chart_base64
            except Exception as e:
                print(f"Error reading generated chart: {str(e)}")
    
    # Fallback to rule-based visualization
    print("Falling back to rule-based visualization...")
    return generate_chart_fallback(df, prompt, analysis_response)

def generate_chart_fallback(df: pd.DataFrame, prompt: str, analysis_response: str = "") -> str:
    """
    Fallback chart generation using rule-based approach
    This is the original generate_chart_from_prompt logic
    """
    try:
        # Set style for better looking charts
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        prompt_lower = prompt.lower()
        lang = detect_language(prompt)
        
        # Detect chart type requested
        chart_type = None
        if any(word in prompt_lower for word in ['bar chart', 'bar graph', 'barra', 'barras']):
            chart_type = 'bar'
        elif any(word in prompt_lower for word in ['line chart', 'line graph', 'linha', 'linhas', 'trend', 'tendência']):
            chart_type = 'line'
        elif any(word in prompt_lower for word in ['pie chart', 'pie graph', 'pizza', 'torta']):
            chart_type = 'pie'
        elif any(word in prompt_lower for word in ['scatter', 'dispersão', 'scatter plot']):
            chart_type = 'scatter'
        elif any(word in prompt_lower for word in ['histogram', 'histograma', 'distribution', 'distribuição']):
            chart_type = 'histogram'
        elif any(word in prompt_lower for word in ['heatmap', 'heat map', 'mapa de calor', 'correlação', 'correlation']):
            chart_type = 'heatmap'
        elif any(word in prompt_lower for word in ['box plot', 'boxplot', 'box', 'caixa']):
            chart_type = 'box'
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Generate chart based on type or infer from data
        if chart_type == 'heatmap' and len(numeric_cols) > 1:
            plt.close()
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation = df[numeric_cols].corr()
            sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            title = 'Correlation Heatmap' if lang == 'en' else 'Mapa de Correlação'
            ax.set_title(title, fontsize=14, fontweight='bold')
            
        elif chart_type == 'pie':
            if len(categorical_cols) > 0:
                col = categorical_cols[0]
                data = df[col].value_counts().head(10)
                ax.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90)
                ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
            elif len(numeric_cols) > 0:
                col = numeric_cols[0]
                ax.pie(df[col].head(10), labels=df.index[:10], autopct='%1.1f%%')
                ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
                
        elif chart_type == 'histogram':
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Frequency' if lang == 'en' else 'Frequência', fontsize=12)
                title = f'Distribution of {col}' if lang == 'en' else f'Distribuição de {col}'
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
        elif chart_type == 'scatter':
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
                ax.set_xlabel(x_col, fontsize=12)
                ax.set_ylabel(y_col, fontsize=12)
                ax.set_title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
        elif chart_type == 'box':
            if len(numeric_cols) > 0:
                df[numeric_cols[:5]].boxplot(ax=ax)
                ax.set_ylabel('Value' if lang == 'en' else 'Valor', fontsize=12)
                ax.set_title('Box Plot', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
        elif chart_type == 'line':
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                df[col].plot(kind='line', ax=ax, linewidth=2, marker='o')
                ax.set_ylabel(col, fontsize=12)
                ax.set_xlabel('Index' if lang == 'en' else 'Índice', fontsize=12)
                title = f'{col} Trend' if lang == 'en' else f'Tendência de {col}'
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
        elif any(word in prompt_lower for word in ['top', 'scorer', 'highest', 'best', 'most', 'maior', 'melhor', 'artilheiro']):
            if len(numeric_cols) > 0:
                target_col = None
                for col in df.columns:
                    if any(word in col.lower() for word in ['goal', 'score', 'point', 'gol', 'ponto', 'valor', 'value']):
                        target_col = col
                        break
                
                if not target_col:
                    target_col = numeric_cols[0]
                
                name_cols = [c for c in df.columns if any(word in c.lower() for word in ['name', 'player', 'team', 'nome', 'jogador', 'time', 'equipe'])]
                
                if name_cols:
                    top_data = df.nlargest(15, target_col)
                    bars = ax.barh(range(len(top_data)), top_data[target_col])
                    ax.set_yticks(range(len(top_data)))
                    ax.set_yticklabels(top_data[name_cols[0]], fontsize=10)
                    ax.set_xlabel(target_col, fontsize=12)
                    title = f'Top 15 by {target_col}' if lang == 'en' else f'Top 15 por {target_col}'
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.invert_yaxis()
                    
                    for i, (idx, bar) in enumerate(zip(top_data.index, bars)):
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2, 
                               f'{width:.1f}', ha='left', va='center', fontsize=9)
                else:
                    top_data = df.nlargest(15, target_col)
                    ax.bar(range(len(top_data)), top_data[target_col])
                    ax.set_xlabel('Rank' if lang == 'en' else 'Posição', fontsize=12)
                    ax.set_ylabel(target_col, fontsize=12)
                    title = f'Top 15 {target_col}' if lang == 'en' else f'Top 15 {target_col}'
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    
        else:
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                grouped = df.groupby(cat_col)[num_col].mean().nlargest(15)
                grouped.plot(kind='barh', ax=ax)
                ax.set_xlabel(num_col, fontsize=12)
                ax.set_ylabel(cat_col, fontsize=12)
                title = f'Average {num_col} by {cat_col}' if lang == 'en' else f'Média de {num_col} por {cat_col}'
                ax.set_title(title, fontsize=14, fontweight='bold')
            elif len(numeric_cols) > 0:
                col = numeric_cols[0]
                df[col].head(20).plot(kind='bar', ax=ax)
                ax.set_ylabel(col, fontsize=12)
                ax.set_xlabel('Index' if lang == 'en' else 'Índice', fontsize=12)
                ax.set_title(f'{col} Values', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save to base64
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

@app.post("/analysis-gen")
async def download_csv(request: URLRequest):
    try:
        print(f"User prompt: {request.prompt}")
        print(f"Number of files to process: {len(request.fileAddresses)}")
        
        # Check if this is a visualization request
        is_viz_request = is_visualization_request(request.prompt)
        print(f"Visualization request detected: {is_viz_request}")
        
        # Download and parse all CSV files
        dataframes = []
        async with httpx.AsyncClient() as client_http:
            for idx, file_address in enumerate(request.fileAddresses):
                print(f"Downloading file {idx + 1}/{len(request.fileAddresses)}: {file_address}")
                response = await client_http.get(str(file_address))
                response.raise_for_status()
                
                # Parse CSV
                csv_content = response.text
                df = pd.read_csv(io.StringIO(csv_content))
                print(f"Loaded DataFrame {idx + 1} with shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                dataframes.append(df)
        
        # Combine all dataframes into one
        if len(dataframes) == 1:
            combined_df = dataframes[0]
        else:
            # Try to concatenate vertically if columns match, otherwise merge horizontally
            try:
                combined_df = pd.concat(dataframes, ignore_index=True)
                print(f"Combined {len(dataframes)} dataframes vertically")
            except Exception as e:
                print(f"Could not concatenate vertically: {e}. Attempting horizontal merge...")
                combined_df = dataframes[0]
                for df in dataframes[1:]:
                    combined_df = pd.merge(combined_df, df, left_index=True, right_index=True, how='outer')
                print(f"Combined {len(dataframes)} dataframes horizontally")
        
        print(f"Final combined DataFrame shape: {combined_df.shape}")
        print(f"Final columns: {list(combined_df.columns)}")
        
        # Analyze with OpenAI (or fallback)
        analysis_text = analyze_dataframe_with_openai(combined_df, request.prompt)
        
        # Generate chart if it's a visualization request OR if the prompt suggests graphical analysis
        chart_base64 = None
        if is_viz_request:
            print("Generating chart for visualization request...")
            chart_base64 = generate_chart_from_prompt(combined_df, request.prompt, analysis_text)
        else:
            # Also try to generate chart for common analytical questions even if not explicitly asking for viz
            prompt_lower = request.prompt.lower()
            should_auto_chart = any(word in prompt_lower for word in [
                'top', 'best', 'highest', 'most', 'compare', 'comparison',
                'maior', 'melhor', 'artilheiro', 'comparar', 'comparação'
            ])
            if should_auto_chart:
                print("Generating chart for analytical query...")
                chart_base64 = generate_chart_from_prompt(combined_df, request.prompt, analysis_text)
        
        # Create data summary
        data_summary = {
            "rows": len(combined_df),
            "columns": len(combined_df.columns),
            "column_names": list(combined_df.columns),
            "numeric_columns": list(combined_df.select_dtypes(include=['number']).columns),
            "categorical_columns": list(combined_df.select_dtypes(include=['object']).columns),
            "files_processed": len(request.fileAddresses),
            "visualization_generated": chart_base64 is not None
        }
        
        # Return JSON response
        return {
            "text_response": analysis_text,
            "chart_base64": chart_base64,
            "data_summary": data_summary
        }
        
    except httpx.RequestError as e:
        print('Error downloading file')
        raise HTTPException(status_code=400, detail=f"Error downloading file: {str(e)}")
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9090)