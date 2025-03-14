# README
Create a parquet files using polars and pandas.

Then we read these created parquet files with polars (polars and pyarrow backend) and pandas (pyarrow backend).
I skipped the option to use fastparquet as a backend with pandas as it is very slow.

# Run locally:
I create a virtual environment with python 3.12.8, activate it and install dependencie with poetry:
```bash
poetry lock --no-update && poetry install --no-root
```

then create the parquet files running:
```bash
python create_parquet_files.py
```
which will write the parquet file into a folder `synthetic_parquet_files`.

Then read the files multiple times, checking for changes in the read values by running
```bash
python test_read_parquet_file.py
```

Subsequently analyse the data running
```bash
python analysis_script.py
```

# Build and Run in Docker
You need to be in to root directory of this project, as we mount the project folder as a volume in the docker container
```bash
docker build -t parquet_checker .
docker run -d -v $(pwd):/app parquet_checker
```
