# %%
import numpy as np
import pandas as pd

file_path = "synthetic_parquet_files/polars/data_000.parquet"  # Update with actual file path

while True:

    def df_generator(file_path):
        """Lazily yield 100 pandas DataFrames by reading the parquet file."""
        for _ in range(100):
            yield pd.read_parquet(file_path)

    def pair_generator(file_path):
        """Lazily yield each unique pair (old_df, new_df) from the generator."""
        gen = df_generator(file_path)
        first_df = next(gen)
        for new_df in gen:
            yield first_df, new_df

    def compare_dataframes(df1, df2, rtol=1e-5, atol=1e-8):
        """Compare two pandas DataFrames, allowing for numeric tolerance."""
        if df1.shape != df2.shape:
            return False

        for col in df1.columns:
            s1 = df1[col]
            s2 = df2[col]

            if pd.api.types.is_numeric_dtype(s1.dtype):
                # Get boolean mask for differing values
                mask = ~np.isclose(s1.to_numpy(), s2.to_numpy(), rtol=rtol, atol=atol, equal_nan=True)

                # Print the differing values
                if mask.any():
                    print("Mismatched values:")
                    print(pd.DataFrame({"s1": s1[mask], "s2": s2[mask]}))

                return not mask.any()
            else:
                # Compare non-numeric columns with NaN-safe equality
                if not s1.equals(s2):
                    return False

        return True

    # Use the pair generator to compare each pair
    for i, (df1, df2) in enumerate(pair_generator(file_path)):
        print(f"On comparison: {i}")

        equal = compare_dataframes(df1, df2)

        assert equal, f"DataFrames at index {i} are not equal"

    print("All lazily generated DataFrames are identical!")
