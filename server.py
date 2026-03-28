from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOST = "127.0.0.1"
PORT = 4173
MODEL = os.environ.get("FURNISHFRAME_GEMINI_MODEL", "gemini-3.1-flash-image-preview")
ANALYSIS_MODEL = os.environ.get("FURNISHFRAME_ANALYSIS_MODEL", "gemini-2.5-flash")
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
WEB_ROOT = Path(__file__).resolve().parent


class FurnishFrameHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_POST(self) -> None:
        if self.path not in {"/api/generate", "/api/analyze", "/api/similar_products"}:
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
            elif self.path == "/api/similar_products":
                result = search_similar_products(payload, api_key)
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
    room_image_data_url = payload.get("roomImageDataUrl")
    room_analysis = payload.get("roomAnalysis") or {}
    room_finishes = payload.get("roomFinishes") or {}
    furniture = payload.get("furniture", [])
    user_prompt = str(payload.get("prompt", "")).strip()

    if not room_image_data_url:
        raise ValueError("roomImageDataUrl is required")

    mime_type, base64_data = parse_data_url(room_image_data_url)
    furniture_lines = []
    for item in furniture:
        product_url = str(item.get("productUrl", "")).strip()
        measurement_note = (
            " Use the linked Amazon listing as the authoritative source for exact product measurements "
            "to the best of your ability."
            if "amazon." in product_url.lower()
            else ""
        )
        source_note = f" Product link: {product_url}." if product_url else ""
        furniture_lines.append(
            f"- {item['name']} at approximately ({round(item['x'])}%, {round(item['y'])}%) "
            f"with scale {item['scale']} and rotation {item['rotation']} degrees."
            f"{source_note}{measurement_note}"
        )
    mapping_lines = describe_room_mapping(room_analysis)
    finish_lines = describe_room_finishes(room_finishes)

    prompt_parts = [
        "Use the provided room photo as the base image.",
        "Use the structured room analysis below as the primary scene map instead of relying only on the raw image.",
        *mapping_lines,
        *finish_lines,
        "Add only the staged furniture items listed below.",
        "Do not redesign, replace, remove, or restyle any existing architecture, decor, furniture, windows, doors, art, rugs, or lighting already present in the room, except for explicitly requested wall and floor finish changes.",
        "Preserve the original camera position, room layout, perspective, materials, shadows, and all existing objects unless one of the staged items physically occludes a small portion of them.",
        "If a requested furniture placement conflicts with a wall, doorway, window, or existing object, keep the item but shift it minimally to the nearest plausible floor position.",
        "Render a single photorealistic still image from the uploaded camera viewpoint.",
        "For any staged item with a product link, follow the real product proportions and dimensions as closely as possible.",
        "If the product link is an Amazon listing, use the exact measurements from that Amazon listing to the best of your ability.",
        "If exact measurements are unavailable, infer conservative real-world dimensions from the product image, title, and typical category proportions instead of inventing exaggerated sizes.",
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


def describe_room_finishes(room_finishes: dict) -> list[str]:
    wall_color = str(room_finishes.get("wallColor", "")).strip()
    wall_material = str(room_finishes.get("wallMaterial", "")).strip()
    floor_color = str(room_finishes.get("floorColor", "")).strip()
    floor_material = str(room_finishes.get("floorMaterial", "")).strip()

    lines = []
    if wall_color or wall_material:
        wall_parts = []
        if wall_color:
            wall_parts.append(f"color {wall_color}")
        if wall_material:
            wall_parts.append(f"material {wall_material}")
        lines.append(f"Update the visible walls to {' with '.join(wall_parts)}.")

    if floor_color or floor_material:
        floor_parts = []
        if floor_color:
            floor_parts.append(f"color {floor_color}")
        if floor_material:
            floor_parts.append(f"material {floor_material}")
        lines.append(f"Update the visible floor to {' with '.join(floor_parts)}.")

    if lines:
        lines.append("Keep the room geometry, floor plan, trim, and lighting consistent while applying these finish changes.")

    return lines


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


def search_similar_products(payload: dict, api_key: str | None = None) -> dict:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("items must include at least one staged item")

    prompt = str(payload.get("prompt", "")).strip()
    room_type = str((payload.get("roomAnalysis") or {}).get("roomType", "")).strip()
    generated_image_data_url = (
        payload.get("generatedImageDataUrl")
        or payload.get("imageDataUrl")
        or payload.get("resultImageDataUrl")
    )

    query_specs: dict[str, dict[str, object]] = {}
    if api_key and generated_image_data_url:
        try:
            request_body = build_similar_product_query_request(
                generated_image_data_url,
                items,
                prompt,
                room_type,
            )
            response = call_gemini_api(request_body, api_key, ANALYSIS_MODEL)
            query_specs = extract_similar_product_queries(response)
        except Exception:
            query_specs = {}

    searches = []

    for item in items[:6]:
        if not isinstance(item, dict):
            continue

        item_name = str(item.get("name", "")).strip()
        if not item_name:
            continue

        spec = query_specs.get(item_name.lower(), {})
        query = str(spec.get("searchQuery", "")).strip() or build_product_search_query(
            item_name,
            prompt,
            room_type,
        )
        results = fetch_duckduckgo_results(query)
        searches.append(
            {
                "itemName": item_name,
                "query": query,
                "results": results,
                "sourceProductUrl": str(item.get("productUrl", "")).strip(),
                "traits": spec.get("traits", []),
            }
        )

    return {"searches": searches}


def build_similar_product_query_request(
    image_data_url: str,
    items: list[dict],
    prompt: str,
    room_type: str,
) -> dict:
    mime_type, base64_data = parse_data_url(image_data_url)
    item_lines = [f"- {str(item.get('name', '')).strip()}" for item in items[:6] if item.get("name")]
    instructions = [
        "Analyze the provided generated interior image and produce highly specific shopping search queries.",
        "Focus only on the staged furniture items listed below.",
        *item_lines,
        f"Room type: {room_type or 'unknown'}",
        f"Style direction: {prompt or 'unspecified'}",
        "For each listed item, identify the visible design details from the image as precisely as possible.",
        "Use concise shopping-oriented descriptors like material, color, silhouette, arm style, leg style, finish, upholstery, era, and size impression.",
        "Return valid JSON only in this format:",
        '{"items":[{"itemName":string,"searchQuery":string,"traits":[string]}]}',
        "Make each searchQuery specific enough to find visually similar products, not generic category pages.",
    ]

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
                    {"text": "\n".join(instructions)},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
        },
    }


def extract_similar_product_queries(response: dict) -> dict[str, dict[str, object]]:
    candidates = response.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini similar-product query generation returned no candidates")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text_payload = "\n".join(part["text"] for part in parts if part.get("text")).strip()
    if not text_payload:
        raise ValueError("Gemini similar-product query generation returned no JSON payload")

    parsed = json.loads(strip_json_fence(text_payload))
    items = parsed.get("items")
    if not isinstance(items, list):
        return {}

    results = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_name = str(item.get("itemName", "")).strip()
        if not item_name:
            continue
        results[item_name.lower()] = {
            "searchQuery": str(item.get("searchQuery", "")).strip(),
            "traits": normalize_strings(item.get("traits")),
        }

    return results


def build_product_search_query(item_name: str, prompt: str, room_type: str) -> str:
    style_terms = re.findall(r"[A-Za-z][A-Za-z-]+", prompt.lower())
    trimmed_style = " ".join(style_terms[:4])
    parts = [item_name]
    if trimmed_style:
        parts.append(trimmed_style)
    if room_type:
        parts.append(room_type)
    parts.append("furniture")
    return " ".join(parts)


def fetch_duckduckgo_results(query: str) -> list[dict[str, str]]:
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        html_text = response.read().decode("utf-8", errors="replace")

    matches = re.findall(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    results = []
    for href, raw_title in matches:
        clean_url = normalize_duckduckgo_href(href)
        clean_title = clean_html_text(raw_title)
        if not clean_url or not clean_title:
            continue
        results.append(
            {
                "title": clean_title,
                "url": clean_url,
                "displayUrl": urllib.parse.urlparse(clean_url).netloc,
            }
        )
        if len(results) >= 5:
            break

    return results


def normalize_duckduckgo_href(href: str) -> str:
    parsed = urllib.parse.urlparse(html.unescape(href))
    if parsed.netloc and parsed.scheme in {"http", "https"}:
        return href

    query = urllib.parse.parse_qs(parsed.query)
    target = query.get("uddg", [""])[0]
    return urllib.parse.unquote(target)


def clean_html_text(value: str) -> str:
    no_tags = re.sub(r"<[^>]+>", "", value)
    return " ".join(html.unescape(no_tags).split())


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


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), FurnishFrameHandler)
    print(f"Serving FurnishFrame at http://{HOST}:{PORT}")
    server.serve_forever()
