# %%
import hashlib
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm


def hash_dataframe_polars(df, col_to_check):
    return hashlib.sha256(df.select(col_to_check).explode(col_to_check).write_csv(None).encode()).hexdigest()


def hash_dataframe_pandas(df, col_to_check):
    return hashlib.sha256(df[col_to_check].to_csv().encode()).hexdigest()


def find_precision_violations(
    paths_parquet_files: list[Path],
    name_test: str,
    rtol: float,
    atol: float,
    hash_fun: Callable[[Any], str],
    read_parquet_file: Callable[[Path], Any],
    num_reps_per_file: int,
    col_to_check: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    precision_violated = []
    errors_reading_files = []

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use tqdm to monitor file processing
    for file_index, path_file in enumerate(tqdm(paths_parquet_files, desc=f"Processing {name_test} Files")):
        previous_hash = None
        previous_df = None

        # Use tqdm to monitor repetitions per file
        for rep in tqdm(range(num_reps_per_file), desc=f"Reps for File {file_index}", leave=False):
            try:
                df = read_parquet_file(path_file)
            except Exception as e:
                errors_reading_files.append(
                    {"name_test": name_test, "file_index": file_index, "rep": rep, "error": str(e)}
                )
                continue

            hash = hash_fun(df)
            if hash != previous_hash and previous_hash is not None:
                tqdm.write(f"{name_test}, hashes differ for file {path_file}")

                for i in range(df.shape[0]):
                    data_1 = np.asarray(df[col_to_check][i])
                    data_2 = np.asarray(previous_df[col_to_check][i])  # type: ignore
                    mask = ~np.isclose(data_1, data_2, rtol=rtol, atol=atol)

                    if len(data_1) != len(data_2):
                        raise ValueError("Lengths differ")

                    if np.any(mask):
                        tqdm.write(f"Precision violation found for {sum(mask)} elements")

                        precision_violated.append(
                            {
                                "name_test": name_test,
                                "length_1": len(data_1),
                                "length_2": len(data_2),
                                "data_1": data_1[mask].tolist(),
                                "data_2": data_2[mask].tolist(),
                                "i": i,
                                "rep": rep,
                                "file_index": file_index,
                            }
                        )

            previous_hash = hash
            previous_df = df

    # Convert results to Polars DataFrame and save as Parquet
    if precision_violated:
        df_precision_violated = pl.DataFrame(precision_violated)
        df_precision_violated.write_parquet(output_dir / f"precision_violations_{name_test}.parquet")

    if errors_reading_files:
        df_errors = pl.DataFrame(errors_reading_files)
        df_errors.write_parquet(output_dir / f"errors_reading_files_{name_test}.parquet")

    return precision_violated, errors_reading_files


col_to_check = "value"
list_set_up = [
    {
        "name_test": "polars_rust",
        "read_parquet_file": lambda path_file: pl.read_parquet(
            path_file, columns=["date", col_to_check], use_pyarrow=False
        ),
        "hash_fun": lambda df: hash_dataframe_polars(df, col_to_check),
    },
    {
        "name_test": "pandas_pyarrow",
        "read_parquet_file": lambda path_file: pd.read_parquet(
            path_file, columns=["date", col_to_check], engine="pyarrow"
        ),
        "hash_fun": lambda df: hash_dataframe_pandas(df, col_to_check),
    },
    # {
    #     "name_test": "pandas_fastparquet",
    #     "read_parquet_file": lambda path_file: pd.read_parquet(
    #         path_file, columns=["date", col_to_check], engine="fastparquet"
    #     ),
    #     "hash_fun": lambda df: hash_dataframe_pandas(df, col_to_check),
    # },
    {
        "name_test": "polars_pyarrow",
        "read_parquet_file": lambda path_file: pl.read_parquet(
            path_file, columns=["date", col_to_check], use_pyarrow=True
        ),
        "hash_fun": lambda df: hash_dataframe_polars(df, col_to_check),
    },
]

rtol = 1e-7
atol = 1e-10
num_reps_per_file = 50
root_dir_results = Path("./results_check_parquet_file")
shutil.rmtree(root_dir_results, ignore_errors=True)

for name_df_lib in ["polars", "pandas"]:
    all_violations = []
    all_errors_reading_files = []
    print("##########################################################################################")
    print(f"loading parquet files created with {name_df_lib}:")
    print("##########################################################################################")
    output_dir_results = root_dir_results / name_df_lib
    dir_parquet_files = Path("synthetic_parquet_files") / name_df_lib
    paths_parquet_files = list(dir_parquet_files.glob("*.parquet"))

    for set_up in list_set_up:
        print("-------------------------------------------------------------")
        print(f"checking {set_up['name_test']}")
        print("-------------------------------------------------------------")

        violations, errors_reading_files = find_precision_violations(
            paths_parquet_files=paths_parquet_files,
            name_test=set_up["name_test"],
            rtol=rtol,
            atol=atol,
            hash_fun=set_up["hash_fun"],
            read_parquet_file=set_up["read_parquet_file"],
            num_reps_per_file=num_reps_per_file,
            col_to_check=col_to_check,
            output_dir=output_dir_results,
        )

        print(f"number of violation events {len(violations)}")
        print(f"number of errors reading files {len(errors_reading_files)}")

        all_violations += violations
        all_errors_reading_files += errors_reading_files

    violations_df = pd.DataFrame(all_violations)
    violations_df.to_csv(output_dir_results / "all_violations.csv")

    errors_reading_files_df = pd.DataFrame(all_errors_reading_files)
    errors_reading_files_df.to_csv(output_dir_results / "all_errors_reading_files.csv")
