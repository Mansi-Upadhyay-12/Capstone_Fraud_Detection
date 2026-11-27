"""FastAPI entrypoint for Stage 6 Fraud Ring Detection & XAI Backend."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.api.routes import router
from app.backend.config import ALLOWED_CORS_ORIGINS
from app.backend.database.db import init_db
from app.backend.ml.frozen_model import get_model_bundle


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    get_model_bundle()
    yield


app = FastAPI(
    title="Temporal GNN Fraud Ring Detection & XAI API",
    description=(
        "FastAPI backend serving the frozen Stage 4H ExpA2_RankGauss182_BiDirGCN model, "
        "Stage 5C suspicious cluster discovery, and on-demand Stage 5B 3-zone Composite Midpoint IG."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)
