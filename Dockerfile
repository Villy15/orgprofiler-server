# Use a slim Python image
FROM python:3.12-slim

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# ---- JVM/Python environment ----
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SCYJAVA_FETCH_JAVA=never
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Optional: choose core ImageJ (stable headless). Use sc.fiji:fiji:2.14.0 for full Fiji.
ENV IMAGEJ_COORD=net.imagej:imagej:2.14.0

WORKDIR /app

# ---- Python deps first (cache layer) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- (Optional) pre-warm the ImageJ Maven cache to speed up first run ----
# This pulls the ImageJ artifacts into the image during build.
RUN python - <<'PY'
import os, imagej, scyjava
os.environ['SCYJAVA_FETCH_JAVA']='never'
scyjava.config.add_option('-Djava.awt.headless=true')
scyjava.config.add_option('-Xmx512m')
imagej.init(os.getenv('IMAGEJ_COORD','net.imagej:imagej:2.14.0'), mode='headless')
PY

# ---- Your app code ----
COPY . .

# ---- Run server (Render provides $PORT) ----
CMD ["sh", "-c", "uvicorn main:api --host 0.0.0.0 --port ${PORT:-8080}"]
