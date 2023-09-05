# syntax=docker/dockerfile:1.2
FROM python:latest
# put you docker configuration here

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos de la aplicación al contenedor
COPY challenge/ ./challenge/
COPY requirements.txt .

# Instala las dependencias de producción
RUN pip install -r requirements.txt

# Expón el puerto en el que tu aplicación FastAPI se ejecuta (ajusta según sea necesario)
EXPOSE 8080

# Comando de inicio para ejecutar la aplicación
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
