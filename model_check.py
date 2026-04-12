from openai import OpenAI
import os
import sys
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("OPENAI_API_KEY not found.")
    print("Create/verify analysis-gen/.env with: OPENAI_API_KEY=your_key_here")
    print("Then run this script again.")
    sys.exit(1)

client = OpenAI(api_key=api_key)
models = client.models.list()

for m in sorted([x.id for x in models.data]):
    print(m)