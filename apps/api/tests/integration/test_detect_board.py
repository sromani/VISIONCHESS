"""Integration tests for detect-board endpoint."""

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
async def test_detect_board_success(client) -> None:
    response = await client.post(
        "/api/v1/detect-board",
        files={"file": ("board.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["output_width"] == 800
    assert body["output_height"] == 800
    assert len(body["corners"]) == 4
    assert body["warped_image_base64"]
    assert len(body["squares"]) == 64
    assert body["squares"][0]["name"] == "a8"
    assert body["squares"][-1]["name"] == "h1"
    assert body["debug_overlay_base64"]
    assert body["debug_montage_base64"]
    assert body["debug"] is not None
    assert body["debug"]["detected_lines_base64"]
    assert body["debug"]["rectified_board_base64"]
    assert body.get("fen")
    assert "classification" in body["metadata"]
    assert "homography" in body["metadata"]
    assert "split" in body["metadata"]
    assert body["metadata"].get("pipeline") == "geometry_first_scanner_v2"
    assert "observed_grid" in body["metadata"]


@pytest.mark.asyncio
async def test_get_square_crop(client) -> None:
    post = await client.post(
        "/api/v1/detect-board",
        files={"file": ("board.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
    )
    job_id = post.json()["job_id"]
    response = await client.get(f"/api/v1/detect-board/{job_id}/squares/e4.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_get_debug_overlay(client) -> None:
    post = await client.post(
        "/api/v1/detect-board",
        files={"file": ("board.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
    )
    job_id = post.json()["job_id"]
    response = await client.get(f"/api/v1/detect-board/{job_id}/debug/grid_overlay")
    assert response.status_code == 200
