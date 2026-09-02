import os

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
app = FastAPI()


@app.get("/")
def root():
    return f"FastAPI service is running correctly - Version {os.getenv('APP_VERSION')}"