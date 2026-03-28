from http.server import BaseHTTPRequestHandler

from api._furnishframe import (
    handle_api_error,
    json_response,
    read_json_body,
    require_api_key,
    search_similar_products,
)


class handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            api_key = require_api_key()
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = read_json_body(self.rfile.read(content_length))
            json_response(self, 200, search_similar_products(payload, api_key))
        except Exception as error:  # pragma: no cover - Vercel runtime path
            handle_api_error(self, error)
