FROM python:3.12-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUTF8=1

CMD ["python", "baseline_rag.py"]
