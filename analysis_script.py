# %%
import ast
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

root_dir_results = Path("./results_check_parquet_file")

for name_df_lib in ["polars", "pandas"]:
    print("##########################################################################################")
    print(f"Analysis of parquet files created by {name_df_lib}:")
    print("##########################################################################################")

    output_dir_results = root_dir_results / name_df_lib
    errors_file = output_dir_results / "all_errors_reading_files.csv"
    violations_file = output_dir_results / "all_violations.csv"

    if not errors_file.exists():
        print(f"File {errors_file} does not exist")

    if not violations_file.exists():
        print(f"File {violations_file} does not exist")
        continue

    violations_df = pl.read_csv(violations_file)

    violations_df = violations_df.with_columns(
        pl.col("data_1").map_elements(lambda x: ast.literal_eval(x), return_dtype=pl.List(pl.Float64)),
        pl.col("data_2").map_elements(lambda x: ast.literal_eval(x), return_dtype=pl.List(pl.Float64)),
    )

    violations_df = violations_df.with_columns(
        pl.col("data_1").len().alias("len_data_1"),
        pl.col("data_2").len().alias("len_data_2"),
    )

    if violations_df.filter(pl.col("len_data_1") != pl.col("len_data_2")).height > 0:
        print("Lengths differ")

    violations_df = violations_df.drop("len_data_1").rename({"len_data_2": "num_violations"})

    tot_violations = (
        violations_df.group_by("name_test")
        .agg(pl.col("num_violations").sum().alias("total_violations"))
        .sort("total_violations")
    )

    print(tot_violations)

    # for name_test in violations_df["name_test"].unique():
    #     df = violations_df.filter(pl.col("name_test") == name_test)
    #     df = df.explode(["data_1", "data_2"])

    #     df = df.with_columns((pl.col("data_1") - pl.col("data_2")).alias("diff"))
    #     df = df.with_columns(pl.col("diff").abs().alias("abs_diff"))
    #     df = df.with_columns((pl.col("abs_diff") / pl.col("data_1").abs()).alias("abs_relative_diff"))

    #     cols = ["name_test", "data_1", "data_2", "abs_diff", "abs_relative_diff"]
    #     df = df.sort(by="abs_diff", descending=True)
    #     print("sorted by abs_diff")
    #     print("head")
    #     print(df.select(cols).head(5))
    #     # print("tail")
    #     # print(df.select(cols).tail(5))

    #     df = df.sort(by="abs_relative_diff", descending=True)
    #     print("sorted by abs_relative_diff")
    #     print("head")
    #     print(df.select(cols).head(5))
    #     # print("tail")
    #     # print(df.select(cols).tail(5))
