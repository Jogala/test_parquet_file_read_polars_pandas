# %%
import polars as pl

file_path = "synthetic_parquet_files/polars/data_000.parquet"  # point this at the appropriate data file


def df_generator(file_path):
    for _ in range(100):
        yield pl.read_parquet(file_path)


def pair_generator(file_path):
    gen = df_generator(file_path)
    first_df = next(gen)
    for new_df in gen:
        yield first_df, new_df


def is_numeric_dtype(dtype):
    """Check if a Polars dtype is numeric."""
    numeric_types = [
        pl.Float32,
        pl.Float64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ]
    return any(isinstance(dtype, t) for t in numeric_types)


def compare_dataframes(df1, df2, rtol=1e-5, atol=1e-8):
    """Compare two DataFrames with specified tolerances."""
    if df1.shape != df2.shape:
        return False

    for col in df1.columns:
        s1 = df1[col]
        s2 = df2[col]

        # For numeric columns, use tolerance
        if is_numeric_dtype(s1.dtype):
            # Check if values are close within tolerance
            diff = abs(s1 - s2)
            tol = atol + rtol * abs(s2)
            if not (diff <= tol).all():
                return False
        else:
            # For non-numeric columns, use exact comparison
            if not (s1 == s2).all():
                return False

    return True


# Use the pair generator to compare each pair using polars.testing.assert_frame_equal
for i, (df1, df2) in enumerate(pair_generator(file_path)):
    print(f"On comparison: {i}")
    # assert_frame_equal(df1, df2)

    # Simple way to check if DataFrames are equal
    are_equal = compare_dataframes(df1, df2)
    assert are_equal, f"DataFrames at index {i} are not equal"

print("All lazily generated DataFrames are identical!")
