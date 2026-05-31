FROM python:3.12-slim

WORKDIR /app

# Once bagimliliklari kur (katman onbellegi icin ayri adim)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Streamlit varsayilan portu
EXPOSE 8501

# Konteyner sagligi: Streamlit'in /_stcore/health uc noktasi kontrol edilir
# (slim imajda curl olmadigi icin python urllib kullanilir).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

# Dokploy/Traefik arkasinda calismak icin headless ve tum arayuzlere bagli
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
