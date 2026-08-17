FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ARG DEBIAN_SNAPSHOT=20250630T000000Z

RUN rm -f /etc/apt/sources.list.d/debian.sources \
    && printf 'deb http://snapshot.debian.org/archive/debian/%s bookworm main\n' "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list \
    && printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99snapshot \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        fonts-dejavu-core=2.37-6 \
        fonts-hosny-amiri=0.113-1 \
        tesseract-ocr=5.3.0-2 \
        tesseract-ocr-ara=1:4.1.0-2 \
        tesseract-ocr-eng=1:4.1.0-2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY integration ./integration
COPY requirements/integration.txt ./requirements/integration.txt
RUN python -m pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements/integration.txt \
    && python -m pip install --no-cache-dir --no-build-isolation -e .
