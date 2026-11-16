# OCR client request fixtures

Request bodies captured byte for byte from the unmodified clients by a local recording server that replied a canned 200.

- `mistral-python-3.7.0-ocr-image.json`: `mistralai` 3.7.0 (Python 3.12, `MISTRAL_API_KEY` unset), `Mistral(server_url=URL).ocr.process(model="mistral-ocr-latest", document={"type": "image_url", "image_url": "data:image/png;base64,..."}, include_image_base64=True)` on a 48x24 RGB PNG; `POST /v1/ocr`, `Content-Type: application/json`, `Accept: application/json`, `user-agent: mistral-client-python/3.7.0`.
