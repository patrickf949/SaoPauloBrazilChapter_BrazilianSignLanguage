# Base image
FROM python:3.10-slim

# Set working directory to /app
WORKDIR /app

# Copy the entire repo into the container (so model 2 levels up is included)
COPY . .

# Switch to backend directory for installing dependencies
WORKDIR /app/webapp/backend

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install python-dotenv  # for local .env support

# Hugging Face Spaces default host/port
ENV HOST=0.0.0.0
ENV PORT=7860

# Expose port
EXPOSE 7860

# Command to run FastAPI
# load_dotenv() will be called in your FastAPI app's startup so it works locally
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
