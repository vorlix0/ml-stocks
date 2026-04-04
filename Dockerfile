# ---- build stage ----
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools
RUN pip install --upgrade pip

COPY pyproject.toml ./
COPY src/ ./src/
COPY config.py ./

# Install project in editable mode (without dev extras)
RUN pip install --no-cache-dir -e .

# ---- runtime stage ----
FROM python:3.12-slim AS runtime

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages and project from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Copy remaining project files
COPY cli.py tune_model.py train_model.py test_model.py process_data.py download_finance_data.py ./

# Create data/model/output directories
RUN mkdir -p data/raw data/processed models outputs/charts \
    && chown -R appuser:appuser /app

USER appuser

# Default: show CLI help
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
