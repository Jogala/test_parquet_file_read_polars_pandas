import polars as pl
import pytest
from polars.testing import assert_frame_equal

# File path to the parquet file
file_path = "synthetic_parquet_files/polars/data_000.parquet"


def df_generator(file_path):
    """Lazily yield 100 DataFrames by reading the parquet file."""
    for _ in range(100):
        yield pl.read_parquet(file_path)


def pair_generator(file_path):
    """
    Lazily yield each unique pair (old_df, new_df) from the generator.
    Only previously seen DataFrames are kept in memory.
    """
    gen = df_generator(file_path)
    old_df = next(gen)
    for new_df in gen:
        yield old_df, new_df
        old_df = new_df


def test_dataframes_are_identical():
    """Test that all dataframes generated from the parquet file are identical."""
    # Use the pair generator to compare each pair
    for i, (df1, df2) in enumerate(pair_generator(file_path)):
        print(f"On comparison: {i}")
        try:
            assert_frame_equal(df1, df2)
            print(f"Comparison {i} passed!")
        except AssertionError as e:
            print(f"Comparison {i} FAILED!")
            print(f"Error details: {e}")
            pytest.fail(f"DataFrames at comparison {i} are not equal. Error: {e}")

    print("All tested DataFrames are identical!")


# This allows the test to be run with `python -m pytest test_file.py -v`
if __name__ == "__main__":
    test_dataframes_are_identical()
