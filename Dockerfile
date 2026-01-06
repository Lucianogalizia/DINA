FROM python:3.11-slim

WORKDIR /app

# dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app
COPY app.py .

# Streamlit config
ENV PORT=8080
EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
