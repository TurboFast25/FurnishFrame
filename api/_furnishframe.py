from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http import HTTPStatus


MODEL = os.environ.get("FURNISHFRAME_GEMINI_MODEL", "gemini-3.1-flash-image-preview")
ANALYSIS_MODEL = os.environ.get("FURNISHFRAME_ANALYSIS_MODEL", "gemini-2.5-flash")
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"


def require_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in the Vercel project environment variables.")
    return api_key


def read_json_body(raw_body: bytes) -> dict:
    if not raw_body:
        return {}
    return json.loads(raw_body)


def json_response(handler, status: HTTPStatus, body: dict) -> None:
    encoded = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def handle_api_error(handler, error: Exception) -> None:
    if isinstance(error, ValueError):
        json_response(handler, HTTPStatus.BAD_REQUEST, {"error": str(error)})
        return
    if isinstance(error, urllib.error.HTTPError):
        body = error.read().decode("utf-8", errors="replace")
        json_response(
            handler,
            HTTPStatus(error.code),
            {"error": f"Gemini API error {error.code}", "details": body},
        )
        return
    if isinstance(error, urllib.error.URLError):
        json_response(
            handler,
            HTTPStatus.BAD_GATEWAY,
            {"error": "Could not reach Gemini API", "details": str(error.reason)},
        )
        return
    raise error


def build_gemini_request(payload: dict) -> dict:
    room_image_data_url = payload.get("roomImageDataUrl")
    room_analysis = payload.get("roomAnalysis") or {}
    furniture = payload.get("furniture", [])
    user_prompt = str(payload.get("prompt", "")).strip()

    if not room_image_data_url:
        raise ValueError("roomImageDataUrl is required")

    mime_type, base64_data = parse_data_url(room_image_data_url)
    furniture_lines = [
        f"- {item['name']} at approximately ({round(item['x'])}%, {round(item['y'])}%) "
        f"with scale {item['scale']} and rotation {item['rotation']} degrees"
        for item in furniture
    ]
    mapping_lines = describe_room_mapping(room_analysis)

    prompt_parts = [
        "Use the provided room photo as the base image.",
        "Use the structured room analysis below as the primary scene map instead of relying only on the raw image.",
        *mapping_lines,
        "Add only the staged furniture items listed below.",
        "Do not redesign, replace, remove, or restyle any existing architecture, decor, furniture, windows, doors, art, rugs, or lighting already present in the room.",
        "Preserve the original camera position, room layout, perspective, materials, shadows, and all existing objects unless one of the staged items physically occludes a small portion of them.",
        "If a requested placement conflicts with a wall, doorway, window, or existing object, keep the item but shift it minimally to the nearest plausible floor position.",
        "Render a single photorealistic still image from the uploaded camera viewpoint.",
        "Stage the room photorealistically with the following furniture placements.",
        "\n".join(furniture_lines) if furniture_lines else "- No extra furniture placements were supplied.",
        f"Total staged items to add: {len(furniture)}.",
        "Viewpoint: keep the camera close to the uploaded photo viewpoint.",
        "Do not introduce any unrequested new furniture or decor.",
        "Preserve room geometry, perspective, and lighting unless explicitly changed.",
    ]
    if user_prompt:
        prompt_parts.append(f"Additional direction: {user_prompt}")

    return {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_data,
                        }
                    },
                    {"text": "\n".join(prompt_parts)},
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": "16:9",
                "imageSize": "2K",
            },
        },
    }


def build_analysis_request(payload: dict) -> dict:
    room_image_data_url = payload.get("roomImageDataUrl")
    if not room_image_data_url:
        raise ValueError("roomImageDataUrl is required")

    mime_type, base64_data = parse_data_url(room_image_data_url)
    prompt = "\n".join(
        [
            "Analyze this room photo and return a compact JSON room map for furniture staging.",
            "Return valid JSON only with these keys:",
            '{'
            '"summary": string,'
            '"roomType": string,'
            '"cameraView": string,'
            '"floorPolygon": [{"x": number, "y": number}],'
            '"wallZones": [{"name": string, "x": number, "y": number, "width": number, "height": number}],'
            '"avoidZones": [{"name": string, "x": number, "y": number, "width": number, "height": number}],'
            '"placementGuidance": [string],'
            '"lighting": string'
            '}',
            "Use percentages from 0 to 100 for every x, y, width, and height field.",
            "Keep floorPolygon to 3-6 points that approximate the visible walkable floor.",
            "Be concrete and concise.",
        ]
    )

    return {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_data,
                        }
                    },
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
        },
    }


def parse_data_url(data_url: str) -> tuple[str, str]:
    if not data_url.startswith("data:") or "," not in data_url:
        raise ValueError("roomImageDataUrl must be a valid data URL")

    header, encoded = data_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("roomImageDataUrl must contain base64 image data")

    mime_type = header[5:].split(";", 1)[0]
    if not mime_type.startswith("image/"):
        raise ValueError("roomImageDataUrl must contain an image")

    return mime_type, encoded


def call_gemini_api(request_body: dict, api_key: str, model: str) -> dict:
    url = f"{API_ROOT}/{model}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_generation_result(response: dict) -> dict:
    candidates = response.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini response did not include any candidates")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    image_data_url = ""
    text_parts = []

    for part in parts:
        inline_data = part.get("inlineData") or part.get("inline_data")
        if inline_data and inline_data.get("data"):
            mime_type = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"
            image_data_url = f"data:{mime_type};base64,{inline_data['data']}"
        elif part.get("text"):
            text_parts.append(part["text"])

    return {
        "imageDataUrl": image_data_url,
        "meta": {
            "model": MODEL,
            "text": "\n".join(text_parts).strip() or "No text response.",
        },
    }


def extract_analysis_result(response: dict) -> dict:
    candidates = response.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini analysis did not include any candidates")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text_payload = "\n".join(part["text"] for part in parts if part.get("text")).strip()
    if not text_payload:
        raise ValueError("Gemini analysis did not include a JSON response")

    cleaned_payload = strip_json_fence(text_payload)

    try:
        parsed = json.loads(cleaned_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse room analysis JSON: {exc}") from exc

    return {
        "summary": str(parsed.get("summary", "")).strip(),
        "roomType": str(parsed.get("roomType", "")).strip(),
        "cameraView": str(parsed.get("cameraView", "")).strip(),
        "floorPolygon": normalize_points(parsed.get("floorPolygon")),
        "wallZones": normalize_rects(parsed.get("wallZones")),
        "avoidZones": normalize_rects(parsed.get("avoidZones")),
        "placementGuidance": normalize_strings(parsed.get("placementGuidance")),
        "lighting": str(parsed.get("lighting", "")).strip(),
        "model": ANALYSIS_MODEL,
    }


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def describe_room_mapping(room_analysis: dict) -> list[str]:
    if not room_analysis:
        return ["- No structured room analysis was supplied."]

    floor_polygon = room_analysis.get("floorPolygon") or []
    wall_zones = room_analysis.get("wallZones") or []
    avoid_zones = room_analysis.get("avoidZones") or []
    guidance = room_analysis.get("placementGuidance") or []

    return [
        f"- Room summary: {room_analysis.get('summary') or 'Unknown room layout.'}",
        f"- Room type: {room_analysis.get('roomType') or 'unknown'}",
        f"- Camera view: {room_analysis.get('cameraView') or 'unknown'}",
        f"- Lighting: {room_analysis.get('lighting') or 'unspecified'}",
        f"- Floor polygon: {json.dumps(floor_polygon)}",
        f"- Wall zones: {json.dumps(wall_zones)}",
        f"- Avoid zones: {json.dumps(avoid_zones)}",
        f"- Placement guidance: {'; '.join(guidance) if guidance else 'none'}",
    ]


def normalize_points(value: object) -> list[dict[str, float]]:
    if not isinstance(value, list):
        return []

    points: list[dict[str, float]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if "x" not in item or "y" not in item:
            continue
        points.append({"x": clamp_percent(item["x"]), "y": clamp_percent(item["y"])})
    return points


def normalize_rects(value: object) -> list[dict[str, float | str]]:
    if not isinstance(value, list):
        return []

    rects: list[dict[str, float | str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        rects.append(
            {
                "name": str(item.get("name", "")).strip(),
                "x": clamp_percent(item.get("x", 0)),
                "y": clamp_percent(item.get("y", 0)),
                "width": clamp_percent(item.get("width", 0)),
                "height": clamp_percent(item.get("height", 0)),
            }
        )
    return rects


def normalize_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def clamp_percent(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, numeric))
