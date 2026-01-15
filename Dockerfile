FROM python:3.11-slim

ARG EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE CHANGELOG.md CITATION.cff /app/
COPY src /app/src

RUN pip install --upgrade pip \
    && if [ -z "$EXTRAS" ]; then \
        pip install .; \
    else \
        pip install ".[${EXTRAS}]"; \
    fi

RUN adduser --disabled-password --gecos "" appuser
USER appuser

ENTRYPOINT ["classiflow"]
CMD ["--help"]
