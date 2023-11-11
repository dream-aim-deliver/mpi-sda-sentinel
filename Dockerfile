# Stage 1: Build stage
FROM python:3.10 AS builder

WORKDIR /scraper

COPY . /scraper

RUN python3 -m venv .venv
ENV PATH="/scraper/.venv/bin:$PATH"
RUN pip install --upgrade pip && pip install -r requirements.txt

# Stage 2: Run stage
FROM python:3.10-slim

ENV PORT=8000

WORKDIR /scraper

COPY --from=builder /scraper /scraper
COPY --from=builder /scraper/.venv /scraper/.venv

ENV PATH="/scraper/.venv/bin:$PATH"

EXPOSE ${PORT}
CMD ["python", "server.py"]

