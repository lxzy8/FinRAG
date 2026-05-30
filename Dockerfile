# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    gromacs \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies and the uaamd package
RUN pip install --no-cache-dir -e .

# Pre-download the force field (optional but recommended for universal use)
RUN uaamd ff update charmm36

# Command to run on container start
ENTRYPOINT ["uaamd"]
CMD ["--help"]
