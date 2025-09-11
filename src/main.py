from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import uvicorn
import httpx
import csv
import io


app = FastAPI()

class URLRequest(BaseModel):
    fileAddress: HttpUrl

@app.post("/analysis-gen")
async def download_csv(request: URLRequest):
    try:
        print("teste")
        async with httpx.AsyncClient() as client:
            response = await client.get(str(request.fileAddress))
            response.raise_for_status()
        
        # Check if it's a CSV file
        # content_type = response.headers.get('content-type', '')
        # if 'csv' not in content_type and not str(request.fileAddress).endswith('.csv'):
        #     raise HTTPException(status_code=400, detail="URL must point to a CSV file")
        
        # Count rows in CSV
        csv_content = response.text
        csv_reader = csv.reader(io.StringIO(csv_content))
        row_count = sum(1 for row in csv_reader)
        
        return {
            "url": str(request.fileAddress),
            "length": row_count,
            "message": "CSV file processed successfully"
        }
    
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Error downloading file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9090)