FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations

RUN pip install --upgrade pip && pip install .

EXPOSE 8000

CMD ["uvicorn", "oink_finai.main:app", "--host", "0.0.0.0", "--port", "8000"]

