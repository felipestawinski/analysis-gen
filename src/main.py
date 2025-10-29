from fastapi import FastAPI, HTTPException
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

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class URLRequest(BaseModel):
    fileAddress: HttpUrl
    prompt: str

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
    analysis = []
    
    # Basic info
    analysis.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.")
    analysis.append(f"Columns: {', '.join(df.columns)}")
    
    # Look for common patterns in the prompt
    prompt_lower = user_prompt.lower()
    
    if any(word in prompt_lower for word in ['top', 'highest', 'best', 'most']):
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if any(keyword in col.lower() for keyword in ['score', 'goal', 'point', 'value']):
                    top_value = df[col].max()
                    top_index = df[col].idxmax()
                    analysis.append(f"Highest {col}: {top_value}")
                    
                    # Try to find name column
                    name_cols = [c for c in df.columns if any(word in c.lower() for word in ['name', 'player', 'team'])]
                    if name_cols:
                        top_name = df.loc[top_index, name_cols[0]]
                        analysis.append(f"Top performer: {top_name} with {top_value} {col}")
                    break
    
    elif any(word in prompt_lower for word in ['average', 'mean']):
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            avg_val = df[col].mean()
            analysis.append(f"Average {col}: {avg_val:.2f}")
    
    elif any(word in prompt_lower for word in ['total', 'sum']):
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            total_val = df[col].sum()
            analysis.append(f"Total {col}: {total_val}")
    
    # Add basic statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        analysis.append("\nBasic Statistics:")
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            analysis.append(f"{col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}")
    
    return "\n".join(analysis) if analysis else "Unable to analyze the data for this specific question."

def analyze_dataframe_with_openai(df: pd.DataFrame, user_prompt: str) -> str:
    """
    Send dataframe info and user prompt to OpenAI for analysis with fallback options
    """
    # Create a summary of the dataframe
    df_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "sample_data": df.head().to_dict(),
        "basic_stats": df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else None
    }
    
    system_prompt = """You are a data analyst expert. Analyze the provided CSV data and answer the user's question.
    Provide clear, concise insights based on the data. If the question requires visualization, specify what type of chart would be appropriate.
    Always mention specific numbers and findings from the data."""
    
    user_message = f"""
    Dataset Information:
    - Shape: {df_info['shape']} (rows, columns)
    - Columns: {df_info['columns']}
    - Data types: {df_info['dtypes']}
    - Sample data: {df_info['sample_data']}
    - Basic statistics: {df_info['basic_stats']}
    
    User Question: {user_prompt}
    
    Please provide a detailed analysis answering the user's question.
    """
    
    # Try OpenAI first
    if os.getenv("OPENAI_API_KEY"):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000,
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

def generate_chart_from_prompt(df: pd.DataFrame, prompt: str, analysis_response: str) -> str:
    """
    Generate a chart based on the prompt and analysis response
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Simple logic to determine chart type based on prompt keywords
        prompt_lower = prompt.lower()
        analysis_lower = analysis_response.lower()
        
        if any(word in prompt_lower for word in ['top', 'scorer', 'highest', 'best', 'most']):
            # Look for numeric columns to create bar charts
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                # Try to find relevant column based on prompt
                target_col = None
                for col in df.columns:
                    if any(word in col.lower() for word in ['goal', 'score', 'point']):
                        target_col = col
                        break
                
                if target_col and target_col in numeric_cols:
                    # Group by player/name column if exists, otherwise use index
                    name_cols = [col for col in df.columns if any(word in col.lower() for word in ['name', 'player', 'team'])]
                    if name_cols:
                        top_data = df.nlargest(10, target_col)
                        plt.bar(range(len(top_data)), top_data[target_col])
                        plt.xticks(range(len(top_data)), top_data[name_cols[0]], rotation=45)
                        plt.ylabel(target_col)
                        plt.title(f'Top 10 {target_col}')
                    else:
                        top_data = df[target_col].nlargest(10)
                        plt.bar(range(len(top_data)), top_data.values)
                        plt.ylabel(target_col)
                        plt.title(f'Top 10 {target_col}')
        
        elif any(word in prompt_lower for word in ['distribution', 'histogram', 'spread']):
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                df[numeric_cols[0]].hist(bins=20)
                plt.xlabel(numeric_cols[0])
                plt.ylabel('Frequency')
                plt.title(f'Distribution of {numeric_cols[0]}')
        
        else:
            # Default: create a simple chart with first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                df[numeric_cols[0]].value_counts().head(10).plot(kind='bar')
                plt.xlabel(numeric_cols[0])
                plt.ylabel('Count')
                plt.title(f'{numeric_cols[0]} Distribution')
        
        plt.tight_layout()
        
        # Save to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        chart_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
        
    except Exception as e:
        print(f"Error generating chart: {str(e)}")
        return None

@app.post("/analysis-gen")
async def download_csv(request: URLRequest):
    try:
        print(f"User prompt: {request.prompt}")
        
        # Download the CSV file
        async with httpx.AsyncClient() as client_http:
            response = await client_http.get(str(request.fileAddress))
            response.raise_for_status()
        
        # Parse CSV
        csv_content = response.text
        df = pd.read_csv(io.StringIO(csv_content))
        print(f"Loaded DataFrame with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Analyze with OpenAI
        analysis_text = analyze_dataframe_with_openai(df, request.prompt)
        
        # Generate chart if relevant
        chart_base64 = generate_chart_from_prompt(df, request.prompt, analysis_text)
        
        # Create data summary
        data_summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "numeric_columns": list(df.select_dtypes(include=['number']).columns),
            "categorical_columns": list(df.select_dtypes(include=['object']).columns)
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