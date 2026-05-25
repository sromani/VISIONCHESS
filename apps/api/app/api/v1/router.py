"""API v1 router aggregation."""

from fastapi import APIRouter

from app.api.v1 import analyses, dataset_review, detect_board, detect_lc2fen, detect_pieces, health, scans, upload

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(upload.router)
api_router.include_router(detect_board.router)
api_router.include_router(detect_lc2fen.router)
api_router.include_router(detect_pieces.router)
api_router.include_router(scans.router)
api_router.include_router(analyses.router)
api_router.include_router(dataset_review.router)
