FROM python:3.12-slim

WORKDIR /app

# Once bagimliliklari kur (katman onbellegi icin ayri adim)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Streamlit varsayilan portu
EXPOSE 8501

# Dokploy/Traefik arkasinda calismak icin headless ve tum arayuzlere bagli
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
