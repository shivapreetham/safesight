FROM python:3.11-slim

WORKDIR /app

# Torch-free serving build: requirements.txt has no torch, so the image and
# its RSS fit free 512MB-RAM hosts (Render free tier). The ONNX fp32 backend
# is also measured ~2.5x faster than eager torch on CPU (docs/benchmarks.md).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ml/
COPY service/ service/

# PORT is provided by the platform (Render, HF Spaces); defaults to 7860.
ENV SAFESIGHT_DEVICE=cpu \
    SAFESIGHT_BACKEND=onnx \
    SAFESIGHT_PRESET=cascade \
    SAFESIGHT_SCORE_LOG_PATH=/tmp/scores.jsonl \
    SAFESIGHT_FEEDBACK_LOG_PATH=/tmp/feedback.jsonl \
    HF_HOME=/tmp/hf

EXPOSE 7860

CMD ["sh", "-c", "uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
