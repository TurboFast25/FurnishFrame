from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOST = "127.0.0.1"
PORT = 4173
MODEL = os.environ.get("ROOMVIS_GEMINI_MODEL", "gemini-3.1-flash-image-preview")
ANALYSIS_MODEL = os.environ.get("ROOMVIS_ANALYSIS_MODEL", "gemini-2.5-flash")
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
WEB_ROOT = Path(__file__).resolve().parent


class RoomVisHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_POST(self) -> None:
        if self.path not in {"/api/generate", "/api/analyze", "/api/generate-tour"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            self.respond_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Set GEMINI_API_KEY before starting server.py."},
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            if self.path == "/api/analyze":
                request_body = build_analysis_request(payload)
                gemini_response = call_gemini_api(request_body, api_key, ANALYSIS_MODEL)
                result = extract_analysis_result(gemini_response)
            elif self.path == "/api/generate-tour":
                result = generate_tour(payload, api_key)
            else:
                request_body = build_gemini_request(payload)
                gemini_response = call_gemini_api(request_body, api_key, MODEL)
                result = extract_generation_result(gemini_response)
        except ValueError as exc:
            self.respond_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self.respond_json(
                exc.code,
                {"error": f"Gemini API error {exc.code}", "details": body},
            )
            return
        except urllib.error.URLError as exc:
            self.respond_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": "Could not reach Gemini API", "details": str(exc.reason)},
            )
            return

        self.respond_json(HTTPStatus.OK, result)

    def respond_json(self, status: HTTPStatus, body: dict) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def build_gemini_request(payload: dict) -> dict:
    return build_gemini_request_for_view(payload, None)


def build_gemini_request_for_view(payload: dict, view: dict | None) -> dict:
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
    view_instruction = describe_view_instruction(view)

    prompt_parts = [
        "Use the provided room photo as the base image.",
        "Use the structured room analysis below as the primary scene map instead of relying only on the raw image.",
        *mapping_lines,
        "Add only the staged furniture items listed below.",
        "Do not redesign, replace, remove, or restyle any existing architecture, decor, furniture, windows, doors, art, rugs, or lighting already present in the room.",
        "Preserve the original camera position, room layout, perspective, materials, shadows, and all existing objects unless one of the staged items physically occludes a small portion of them.",
        "If a requested placement conflicts with a wall, doorway, window, or existing object, keep the item but shift it minimally to the nearest plausible floor position.",
        "Render the result as an immersive wide room view that keeps the full scene coherent for pan-and-zoom exploration.",
        "Stage the room photorealistically with the following furniture placements.",
        "\n".join(furniture_lines) if furniture_lines else "- No extra furniture placements were supplied.",
        f"Total staged items to add: {len(furniture)}.",
        view_instruction,
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


def generate_tour(payload: dict, api_key: str) -> dict:
    views = get_tour_views()
    generated_views = []

    for view in views:
        request_body = build_gemini_request_for_view(payload, view)
        gemini_response = call_gemini_api(request_body, api_key, MODEL)
        result = extract_generation_result(gemini_response)
        generated_views.append(
            {
                "id": view["id"],
                "label": view["label"],
                "hint": view["hint"],
                "imageDataUrl": result.get("imageDataUrl", ""),
                "meta": result.get("meta", {}),
            }
        )

    return {
        "views": generated_views,
        "defaultViewId": generated_views[0]["id"] if generated_views else "",
        "meta": {
            "model": MODEL,
            "viewCount": len(generated_views),
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
    with urllib.request.urlopen(request, timeout=90) as response:
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


def get_tour_views() -> list[dict[str, str]]:
    return [
        {
            "id": "front",
            "label": "Front",
            "hint": "Original anchored view",
            "instruction": (
                "Primary view: stay close to the uploaded camera position and show the full staged room head-on."
            ),
        },
        {
            "id": "left",
            "label": "Look Left",
            "hint": "Shift attention toward the left side of the room",
            "instruction": (
                "Secondary view: pivot the camera slightly left from the original viewpoint while preserving room continuity and item placement."
            ),
        },
        {
            "id": "right",
            "label": "Look Right",
            "hint": "Shift attention toward the right side of the room",
            "instruction": (
                "Secondary view: pivot the camera slightly right from the original viewpoint while preserving room continuity and item placement."
            ),
        },
        {
            "id": "closer",
            "label": "Move In",
            "hint": "Take a modest step into the room",
            "instruction": (
                "Secondary view: move the camera a small step forward into the room for a closer immersive look, without changing the room layout."
            ),
        },
    ]


def describe_view_instruction(view: dict | None) -> str:
    if not view:
        return "Viewpoint: keep the camera close to the uploaded photo viewpoint."
    return f"Viewpoint: {view['instruction']}"


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


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), RoomVisHandler)
    print(f"Serving RoomVis at http://{HOST}:{PORT}")
    server.serve_forever()
