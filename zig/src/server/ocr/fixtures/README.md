# OCR client request fixtures

Request bodies captured byte for byte from the unmodified clients by a local recording server that replied a canned 200.

- `mistral-python-3.7.0-ocr-image.json`: `mistralai` 3.7.0 (Python 3.12, `MISTRAL_API_KEY` unset), `Mistral(server_url=URL).ocr.process(model="mistral-ocr-latest", document={"type": "image_url", "image_url": "data:image/png;base64,..."}, include_image_base64=True)` on a 48x24 RGB PNG; `POST /v1/ocr`, `Content-Type: application/json`, `Accept: application/json`, `user-agent: mistral-client-python/3.7.0`.
- `lit-2514f8e-ocr.multipart`: LiteParse 2514f8e (`cargo build --release -p liteparse --bin lit`), `lit parse --ocr-server-url http://127.0.0.1:PORT/ocr --num-workers 1 --format json page.png` on the same PNG; `POST /ocr`, `Content-Type: multipart/form-data; boundary=58d2094fd455366b-9209308a2fdfd403-4cb870f78f225bb5-840e3d54fef15603`, `accept: */*`.
