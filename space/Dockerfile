# The computation of views at ~100 m (Panache's D74). Its data are built into the image, so that a
# Space waking up from sleep starts with them: the public releases of AdrienJKTH/panache-archive.
FROM python:3.12-slim

# LightGBM's OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY panache panache

ENV PYTHONPATH=/app PANACHE_DATA_DIR=/data
RUN mkdir -p /data \
    && python -m panache.fetch_archive --compute \
    && chmod -R a+rwX /data

CMD ["uvicorn", "panache.compute_app:create_compute_app", "--factory", \
     "--host", "0.0.0.0", "--port", "7860"]
