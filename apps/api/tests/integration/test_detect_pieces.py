"""Integration tests for detect-pieces endpoint."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


def _jpeg_bytes() -> bytes:
    board = np.zeros((400, 400, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            color = 200 if (r + c) % 2 == 0 else 60
            board[r * 50 : (r + 1) * 50, c * 50 : (c + 1) * 50] = color
    src = np.float32([[0, 0], [399, 0], [399, 399], [0, 399]])
    dst = np.float32([[50, 30], [350, 20], [330, 370], [30, 350]])
    warped = cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), (400, 400))
    ok, buf = cv2.imencode(".jpg", warped)
    assert ok
    return buf.tobytes()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_detect_pieces_success(client) -> None:
    response = await client.post(
        "/api/v1/detect-pieces",
        files={"file": ("board.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
    )
    if response.status_code == 422 and ("YOLO" in response.text or "ONNX" in response.text or "classifier" in response.text.lower()):
        pytest.skip("models not available")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "yolo_localize_classify"
    assert len(body["squares"]) == 64
    assert body["rectified_board_base64"]
    assert body["detector"]["localization_only"] is True
    assert body["classifier"]["model_type"] == "piece_classifier"
    assert "detections" in body
