from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

@app.post("/analysis-gen")
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename.endswith(".xlsx"):
        return {"error": "Only .xlsx files are allowed"}
    
    # Read the file contents
    contents = await file.read()
    
    # Here you could save it or process with openpyxl/pandas
    with open(f"uploaded_{file.filename}", "wb") as f:
        f.write(contents)
    
    return {"filename": file.filename, "message": "File uploaded successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9090)
