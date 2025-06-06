FROM python:3.10-slim

WORKDIR /app

# 👇 Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# 👇 Install Python dependencies before copying the rest
RUN pip install --no-cache-dir -r requirements.txt

# 👇 Now copy the rest of the app (this may change more often)
COPY . .

ENV PYTHONPATH=/app

CMD ["python", "scripts/train.py"]
