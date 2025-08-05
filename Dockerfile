# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Explicitly ensure comicvine_index folder is copied
COPY comicvine_index /app/comicvine_index

# Debug: list contents of FAISS folder during build
RUN ls -l /app/comicvine_index

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (Cloud Run default)
EXPOSE 8080

# Start Gunicorn server
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
