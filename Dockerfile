ARG PYTHON_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
FROM ${PYTHON_IMAGE} AS runtime

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --gid "${APP_GID}" oink \
    && useradd --uid "${APP_UID}" --gid oink --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin oink

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations

RUN pip install --root-user-action=ignore --no-compile . && \
    python -c "import openai; import oink_finai"

EXPOSE 8000

USER oink:oink

CMD ["uvicorn", "oink_finai.main:app", "--host", "0.0.0.0", "--port", "8000"]
