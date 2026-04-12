# Analysis Gen Project

## Overview
This project is a FastAPI application designed to handle the uploading of Excel files. It provides an endpoint for users to upload `.xlsx` files, which are then validated and saved to the server.
This is part of the KPI project.

## Project Structure
```
analysis-gen
├── src
│   ├── main.py          # Main application code
│   ├── routes
│   │   └── __init__.py  # Route definitions (currently empty)
│   └── models
│       └── __init__.py  # Data models (currently empty)
├── requirements.txt      # Project dependencies
├── .gitignore            # Files and directories to ignore in Git
└── README.md             # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd analysis-gen
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command:
```
uvicorn src.main:app --host 127.0.0.1 --port 9090
```

You can then access the API at `http://127.0.0.1:9090/upload-excel/` to upload Excel files.

## Endpoints
- **POST /upload-excel/**: Upload an Excel file. Only `.xlsx` files are allowed. The uploaded file will be saved on the server.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.
