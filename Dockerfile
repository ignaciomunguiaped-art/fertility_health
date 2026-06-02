# Usar imagen base de Python
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de requisitos
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación
COPY app.py .
COPY templates/ templates/
COPY modelos/ modelos/

# Exponer puerto
EXPOSE 8080

# Variable de entorno para Port
ENV PORT=8080
ENV MODEL_PATH=/app/modelos

# Comando para ejecutar la app con gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 --access-logfile - --error-logfile - app:app

