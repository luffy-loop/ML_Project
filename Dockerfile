FROM node:20-alpine AS web-build
WORKDIR /workspace/web
COPY web/package*.json ./
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
COPY --from=web-build /workspace/web/dist ./web/dist
EXPOSE 8000
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
