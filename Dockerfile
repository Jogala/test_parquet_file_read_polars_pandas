# Use Python 3.12.8 as the base image
FROM python:3.12.8

# Set the working directory inside the container
WORKDIR /app

# Copy the Poetry files
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install poetry

# Force Poetry to install dependencies globally (no virtualenv)
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-root

# Copy all source files into the container
COPY . .

# Run the script when the container starts
CMD ["sh", "-c", "python create_parquet_files.py && python test_read_parquet_file.py && python analysis_script.py"]
