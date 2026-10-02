from __future__ import annotations

import base64
import binascii
import json
import os
import re
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
from uuid import uuid4


def load_local_env() -> None:
    env_path = Path(__file__).with_name(".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name:
            os.environ.setdefault(name, value)


load_local_env()

HOST = "127.0.0.1"
PORT = 8001
MAX_REQUEST_BYTES = 32 * 1024 * 1024
ROBOFLOW_WORKFLOW_URL = os.environ.get(
    "ROBOFLOW_WORKFLOW_URL",
    "https://serverless.roboflow.com/"
    "akello-scovia/workflows/"
    "jaguzi-ear-tag-reader-1790754713845",
).strip()
UPLOAD_DIRECTORY = Path(
    os.environ.get("JAGUZA_UPLOAD_DIR", Path(__file__).with_name("uploads"))
)
IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/avif": ".avif",
}


def is_local_origin(origin: str | None) -> bool:
    if not origin:
        return True
    parsed = urlsplit(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


class UploadHandler(BaseHTTPRequestHandler):
    server_version = "JaguzaLocalUpload/1.0"

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        origin = self.headers.get("Origin")
        if origin and is_local_origin(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        origin = self.headers.get("Origin")
        if not is_local_origin(origin):
            self._send_json(403, {"error": "Origin is not allowed."})
            return
        self.send_response(204)
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json(404, {"error": "Not found."})
            return
        self._send_json(
            200,
            {
                "status": "ready",
                "model_configured": bool(os.environ.get("ROBOFLOW_API_KEY", "").strip()),
                "workflow_url": ROBOFLOW_WORKFLOW_URL,
            },
        )

    def do_POST(self) -> None:
        if self.path == "/api/read-ear-tag":
            self._read_ear_tag()
            return
        if self.path != "/api/uploads":
            self._send_json(404, {"error": "Not found."})
            return
        origin = self.headers.get("Origin")
        if not is_local_origin(origin):
            self._send_json(403, {"error": "Origin is not allowed."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length."})
            return
        if content_length <= 0:
            self._send_json(400, {"error": "Request body is empty."})
            return
        if content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "Upload exceeds the 32 MB limit."})
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data;"):
            self._send_json(415, {"error": "Expected multipart/form-data."})
            return

        raw_message = (
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode(
                "latin-1"
            )
            + self.rfile.read(content_length)
        )
        message = BytesParser(policy=policy.default).parsebytes(raw_message)
        if not message.is_multipart():
            self._send_json(400, {"error": "Invalid multipart body."})
            return

        fields: dict[str, str] = {}
        files: dict[str, tuple[str, bytes]] = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not isinstance(name, str):
                continue
            payload = part.get_payload(decode=True) or b""
            filename = part.get_filename()
            if filename is not None:
                files[name] = (part.get_content_type().lower(), payload)
            else:
                fields[name] = payload.decode("utf-8", errors="replace")

        image = files.get("image")
        if image is None or not image[1]:
            self._send_json(400, {"error": "The image field is required."})
            return
        image_extension = IMAGE_EXTENSIONS.get(image[0])
        if image_extension is None:
            self._send_json(415, {"error": "Unsupported image content type."})
            return
        try:
            metadata = json.loads(fields.get("ocr_json", "{}"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "ocr_json must contain valid JSON."})
            return
        if not isinstance(metadata, dict):
            self._send_json(400, {"error": "ocr_json must be a JSON object."})
            return

        requested_id = str(metadata.get("capture_id", ""))
        capture_id = re.sub(r"[^A-Za-z0-9_-]", "_", requested_id).strip("_")
        if not capture_id:
            capture_id = uuid4().hex
        capture_directory = UPLOAD_DIRECTORY / capture_id
        capture_directory.mkdir(parents=True, exist_ok=True)
        saved_files: list[str] = []

        image_name = f"image{image_extension}"
        (capture_directory / image_name).write_bytes(image[1])
        saved_files.append(image_name)

        for field_name, extension in (
            ("tag_crop", ".png"),
            ("annotated_image", ".png"),
        ):
            attachment = files.get(field_name)
            if attachment is not None and attachment[1]:
                filename = f"{field_name}{extension}"
                (capture_directory / filename).write_bytes(attachment[1])
                saved_files.append(filename)

        (capture_directory / "ocr.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        saved_files.append("ocr.json")
        if "ocr_error" in fields:
            (capture_directory / "ocr_error.txt").write_text(
                fields["ocr_error"], encoding="utf-8"
            )
            saved_files.append("ocr_error.txt")

        self._send_json(
            201,
            {
                "capture_id": capture_id,
                "saved_files": saved_files,
                "message": "Photo and evidence saved on this machine.",
            },
        )

    def _read_ear_tag(self) -> None:
        origin = self.headers.get("Origin")
        if not is_local_origin(origin):
            self._send_json(403, {"error": "Origin is not allowed."})
            return

        api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
        if not api_key:
            self._send_json(
                503,
                {
                    "error": (
                        "ROBOFLOW_API_KEY is not configured on the local server. "
                        "Set it in the server environment and restart the server."
                    )
                },
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length."})
            return
        if content_length <= 0:
            self._send_json(400, {"error": "Request body is empty."})
            return
        if content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "Image exceeds the 32 MB limit."})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON."})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "Request body must be a JSON object."})
            return
        image_base64 = payload.get("imageBase64")
        if not isinstance(image_base64, str) or not image_base64:
            self._send_json(400, {"error": "imageBase64 is required."})
            return
        try:
            base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError):
            self._send_json(400, {"error": "imageBase64 is not valid base64."})
            return

        workflow_body = json.dumps(
            {"inputs": {"image": {"type": "base64", "value": image_base64}}}
        ).encode("utf-8")
        request = Request(
            ROBOFLOW_WORKFLOW_URL,
            data=workflow_body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                workflow_result = json.loads(response.read())
        except HTTPError as error:
            detail = error.read(2048).decode("utf-8", errors="replace")
            self._send_json(
                error.code,
                {
                    "error": f"Roboflow workflow returned HTTP {error.code}.",
                    "detail": detail,
                    "workflow_url": ROBOFLOW_WORKFLOW_URL,
                },
            )
            return
        except (URLError, TimeoutError, OSError) as error:
            self._send_json(502, {"error": f"Could not reach Roboflow: {error}"})
            return
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(502, {"error": "Roboflow returned invalid JSON."})
            return

        normalized_result = normalize_workflow_result(workflow_result)
        self._send_json(200, normalized_result)


def normalize_workflow_result(result: object) -> object:
    if not isinstance(result, dict):
        return result
    outputs = result.get("outputs")
    if not isinstance(outputs, list):
        return result

    normalized_outputs: list[object] = []
    for output in outputs:
        if not isinstance(output, dict):
            normalized_outputs.append(output)
            continue
        normalized_output = dict(output)
        for name in ("tag_text", "tag_detections", "output_image"):
            value = normalized_output.get(name)
            if isinstance(value, dict) and "value" in value:
                value = value["value"]
            if name == "tag_detections" and isinstance(value, dict):
                value = value.get("predictions", value)
            normalized_output[name] = value
        normalized_outputs.append(normalized_output)

    return {**result, "outputs": normalized_outputs}


if __name__ == "__main__":
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), UploadHandler)
    print(f"Jaguza upload API listening on http://{HOST}:{PORT}")
    print(f"Uploads are saved to {UPLOAD_DIRECTORY.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()