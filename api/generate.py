from http.server import BaseHTTPRequestHandler

from api._furnishframe import (
    MODEL,
    build_gemini_request,
    call_gemini_api,
    extract_generation_result,
    handle_api_error,
    json_response,
    read_json_body,
    require_api_key,
)


class handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            api_key = require_api_key()
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = read_json_body(self.rfile.read(content_length))
            request_body = build_gemini_request(payload)
            gemini_response = call_gemini_api(request_body, api_key, MODEL)
            json_response(self, 200, extract_generation_result(gemini_response))
        except Exception as error:  # pragma: no cover - Vercel runtime path
            handle_api_error(self, error)
