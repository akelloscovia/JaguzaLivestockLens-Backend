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

from io import BytesIO

from PIL import Image


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
IMAGES_DIRECTORY = Path(
    os.environ.get("JAGUZA_IMAGES_DIR", Path(__file__).with_name("images"))
)
JAGUZA_FARM_API_URL = os.environ.get(
    "JAGUZA_FARM_API_URL",
    "https://backend.jaguzalivestockug.com/api/create_ear_tag_reading",
).strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_TAG_PROMPT = (
    "You are looking at a cropped photo of a cattle ear tag. "
    "tag_text should contain everything printed on the tag, including any name, "
    "word, and number, exactly as it appears (e.g. 'BLITZ 1527'). "
    "Some official tags (e.g. USDA/AIN tags) print two numbers: a long full official "
    "ID (often in groups, e.g. '840 003 127 152 810') and a shorter visual number "
    "that is usually the last several digits of the full ID (e.g. '52810'). "
    "tag_number should contain just the short visual number, digits only, as a "
    "string, or null if there is no number on the tag. "
    "tag_official_number should contain the full official ID if one is printed, "
    "digits only with no spaces (e.g. '840003127152810'), or null if there isn't a "
    "separate full official number (for example, if the tag only has one number, "
    "put it in tag_number and leave tag_official_number null). "
    "Identify the primary color of the tag material. If a farm name or identifier is "
    "visible on the tag, include it; otherwise use null for that field. "
    "Respond with ONLY compact JSON, no markdown, in exactly this shape: "
    '{"tag_text": string or null, "tag_number": string or null, '
    '"tag_official_number": string or null, "tag_color": string or null, '
    '"farm": string or null}'
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
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ready",
                    "model_configured": bool(
                        os.environ.get("ROBOFLOW_API_KEY", "").strip()
                    ),
                    "workflow_url": ROBOFLOW_WORKFLOW_URL,
                },
            )
            return
        if self.path.startswith("/images/"):
            self._serve_image(self.path[len("/images/"):])
            return
        self._send_json(404, {"error": "Not found."})

    def _serve_image(self, relative_path: str) -> None:
        relative_path = relative_path.split("?", 1)[0]
        candidate = (IMAGES_DIRECTORY / relative_path).resolve()
        images_root = IMAGES_DIRECTORY.resolve()
        if images_root not in candidate.parents or not candidate.is_file():
            self._send_json(404, {"error": "Image not found."})
            return
        content_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }.get(candidate.suffix.lower(), "application/octet-stream")
        body = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

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

        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(content_length)

        if content_type.lower().startswith("multipart/form-data;"):
            raw_message = (
                f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode(
                    "latin-1"
                )
                + body
            )
            message = BytesParser(policy=policy.default).parsebytes(raw_message)
            if not message.is_multipart():
                self._send_json(400, {"error": "Invalid multipart body."})
                return

            image_file_bytes: bytes | None = None
            capture_id = None
            farm_id = None
            animal_id = None
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name == "imageFile":
                    image_file_bytes = part.get_payload(decode=True) or b""
                elif name == "capture_id":
                    capture_id = (part.get_payload(decode=True) or b"").decode(
                        "utf-8", errors="replace"
                    )
                elif name == "farm_id":
                    farm_id = (part.get_payload(decode=True) or b"").decode(
                        "utf-8", errors="replace"
                    )
                elif name == "animal_id":
                    animal_id = (part.get_payload(decode=True) or b"").decode(
                        "utf-8", errors="replace"
                    )

            if not image_file_bytes:
                self._send_json(400, {"error": "The imageFile field is required."})
                return
            image_base64 = base64.b64encode(image_file_bytes).decode("ascii")
        else:
            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "Request body must be valid JSON."})
                return
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "Request body must be a JSON object."})
                return
            image_base64 = payload.get("imageBase64")
            if not isinstance(image_base64, str) or not image_base64:
                self._send_json(
                    400, {"error": "imageBase64 or imageFile is required."}
                )
                return
            try:
                base64.b64decode(image_base64, validate=True)
            except (binascii.Error, ValueError):
                self._send_json(400, {"error": "imageBase64 is not valid base64."})
                return
            capture_id = payload.get("capture_id")
            farm_id = payload.get("farm_id")
            animal_id = payload.get("animal_id")

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

        #PROCESSING CHANGES - START

        picture_original_b64 = image_base64
        picture_annotated_b64 = None
        first_detection: dict | None = None
        if isinstance(normalized_result, dict):
            outputs = normalized_result.get("outputs")
            if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
                picture_annotated_b64 = outputs[0].get("output_image")
                detections = outputs[0].get("tag_detections")
                if isinstance(detections, list) and detections and isinstance(
                    detections[0], dict
                ):
                    first_detection = detections[0]

        #PROCESSING CHANGES - END

        #PHOTO SYNC CHANGES - START

        image_id = uuid4().hex
        image_subdirectory = IMAGES_DIRECTORY / image_id
        image_subdirectory.mkdir(parents=True, exist_ok=True)

        picture_original_url = self._save_base64_image(
            picture_original_b64, image_subdirectory / "original.jpg"
        )
        picture_annotated_url = self._save_base64_image(
            picture_annotated_b64, image_subdirectory / "annotated.jpg"
        )
        picture_cropped_url = self._save_cropped_image(
            picture_original_b64, first_detection, image_subdirectory / "cropped.jpg"
        )

        #PHOTO SYNC CHANGES - END

        #ADD RESPONSE FILES CHANGES - START

        roboflow_reading = dict(normalized_result) if isinstance(normalized_result, dict) else {
            "result": normalized_result
        }
        outputs = roboflow_reading.get("outputs")
        if isinstance(outputs, list):
            roboflow_reading["outputs"] = [
                {k: v for k, v in output.items() if k != "output_image"}
                if isinstance(output, dict)
                else output
                for output in outputs
            ]

        pictures = {
            "original": picture_original_url,
            "annotated": picture_annotated_url,
            "cropped": picture_cropped_url,
        }

        #ADD RESPONSE FILES CHANGES - END

        #OPEN AI TAG READING CHANGES - START

        cropped_path = image_subdirectory / "cropped.jpg"
        if cropped_path.is_file():
            openai_reading = self._read_tag_with_openai(cropped_path.read_bytes())
        else:
            openai_reading = {
                "error": "No cropped tag image was available to analyze."
            }

        #OPEN AI TAG READING CHANGES - END

        #JAGUZA FARM FORWARDING CHANGES - START

        jaguza_farm_sync = self._forward_to_jaguza_farm(
            capture_id=capture_id,
            farm_id=farm_id,
            animal_id=animal_id,
            roboflow_reading=roboflow_reading,
            openai_reading=openai_reading,
            image_subdirectory=image_subdirectory,
        )

        #JAGUZA FARM FORWARDING CHANGES - END

        self._send_json(
            200,
            {
                "roboflow_reading": roboflow_reading,
                "openai_reading": openai_reading,
                "pictures": pictures,
                "jaguza_farm_sync": jaguza_farm_sync,
            },
        )

    def _forward_to_jaguza_farm(
        self,
        capture_id: object,
        farm_id: object,
        animal_id: object,
        roboflow_reading: dict,
        openai_reading: dict,
        image_subdirectory: Path,
    ) -> dict[str, object]:
        if not JAGUZA_FARM_API_URL:
            return {"error": "JAGUZA_FARM_API_URL is not configured."}

        fields: dict[str, str] = {
            "roboflow_reading": json.dumps(roboflow_reading),
            "openai_reading": json.dumps(openai_reading),
        }
        if capture_id:
            fields["capture_id"] = str(capture_id)
        if farm_id:
            fields["farm_id"] = str(farm_id)
        if animal_id:
            fields["animal_id"] = str(animal_id)

        files: dict[str, tuple[str, str, bytes]] = {}
        for field_name, filename in (
            ("picture_original", "original.jpg"),
            ("picture_annotated", "annotated.jpg"),
            ("picture_cropped", "cropped.jpg"),
        ):
            file_path = image_subdirectory / filename
            if file_path.is_file():
                files[field_name] = (filename, "image/jpeg", file_path.read_bytes())

        boundary = uuid4().hex
        body = BytesIO()
        for name, value in fields.items():
            body.write(f"--{boundary}\r\n".encode("utf-8"))
            body.write(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(
                    "utf-8"
                )
            )
            body.write(value.encode("utf-8"))
            body.write(b"\r\n")
        for name, (filename, content_type, file_bytes) in files.items():
            body.write(f"--{boundary}\r\n".encode("utf-8"))
            body.write(
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("utf-8")
            )
            body.write(file_bytes)
            body.write(b"\r\n")
        body.write(f"--{boundary}--\r\n".encode("utf-8"))

        request = Request(
            JAGUZA_FARM_API_URL,
            data=body.getvalue(),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except HTTPError as error:
            detail = error.read(2048).decode("utf-8", errors="replace")
            return {
                "error": f"Jaguza Farm backend returned HTTP {error.code}.",
                "detail": detail,
            }
        except (URLError, TimeoutError, OSError) as error:
            return {"error": f"Could not reach Jaguza Farm backend: {error}"}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"error": "Jaguza Farm backend returned invalid JSON."}

    def _read_tag_with_openai(self, image_bytes: bytes) -> dict[str, object]:
        if not OPENAI_API_KEY:
            return {"error": "OPENAI_API_KEY is not configured on the local server."}

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        body = json.dumps(
            {
                "model": OPENAI_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": OPENAI_TAG_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
                "temperature": 0,
            }
        ).encode("utf-8")
        request = Request(
            OPENAI_CHAT_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                result = json.loads(response.read())
        except HTTPError as error:
            detail = error.read(2048).decode("utf-8", errors="replace")
            return {
                "error": f"OpenAI returned HTTP {error.code}.",
                "detail": detail,
            }
        except (URLError, TimeoutError, OSError) as error:
            return {"error": f"Could not reach OpenAI: {error}"}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"error": "OpenAI returned invalid JSON."}

        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return {"error": "Unexpected OpenAI response shape."}

        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Could not parse OpenAI response as JSON.", "raw": content}
        if not isinstance(parsed, dict):
            return {"error": "OpenAI response JSON was not an object.", "raw": content}
        return parsed

    def _save_cropped_image(
        self, original_b64: object, detection: dict | None, destination: Path
    ) -> str | None:
        if not isinstance(original_b64, str) or not original_b64 or detection is None:
            return None
        try:
            center_x = float(detection["x"])
            center_y = float(detection["y"])
            width = float(detection["width"])
            height = float(detection["height"])
        except (KeyError, TypeError, ValueError):
            return None
        try:
            original_bytes = base64.b64decode(original_b64, validate=True)
            with Image.open(BytesIO(original_bytes)) as original:
                image_width, image_height = original.size
                left = max(0, int(center_x - width / 2))
                top = max(0, int(center_y - height / 2))
                right = min(image_width, int(center_x + width / 2))
                bottom = min(image_height, int(center_y + height / 2))
                if right <= left or bottom <= top:
                    return None
                cropped = original.convert("RGB").crop((left, top, right, bottom))
                destination.parent.mkdir(parents=True, exist_ok=True)
                cropped.save(destination, format="JPEG")
        except (binascii.Error, ValueError, OSError):
            return None
        host = self.headers.get("Host", f"{HOST}:{PORT}")
        relative_path = destination.relative_to(IMAGES_DIRECTORY).as_posix()
        return f"http://{host}/images/{relative_path}"

    def _save_base64_image(self, value: object, destination: Path) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            image_bytes = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            return None
        destination.write_bytes(image_bytes)
        host = self.headers.get("Host", f"{HOST}:{PORT}")
        relative_path = destination.relative_to(IMAGES_DIRECTORY).as_posix()
        return f"http://{host}/images/{relative_path}"


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
    IMAGES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), UploadHandler)
    print(f"Jaguza upload API listening on http://{HOST}:{PORT}")
    print(f"Uploads are saved to {UPLOAD_DIRECTORY.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()