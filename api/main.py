import secrets
from typing import Annotated
import json

import pandas as pd

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from api.infer import infer_text

app = FastAPI()

security = HTTPBasic()


def get_current_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)]
):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = b"theydo"
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = b"netherlands"
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.post("/nlp/infer_text")
def infer_text_endpoint(text: str, username: Annotated[str, Depends(get_current_username)]):
    return {"text": text, "prediction": infer_text(text)}

@app.get("/nlp/model_info")
def model_info(username: Annotated[str, Depends(get_current_username)]):
    return {"name": "initial model", "version": "0.1", "model_bucket": "s3://mock-bucket"}
        