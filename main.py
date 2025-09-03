import pyarrow.parquet as pq
import kagglehub as kh

# Replace with the actual path to your Parquet file
parquet_file_path = kh.dataset_download(
    handle="lichess/chess-evaluations", path="train-00000-of-00013.parquet"
)

# Open the Parquet file
pq_file = pq.ParquetFile(parquet_file_path)

print(f"File: {parquet_file_path}")
print(f"Total number of row groups: {pq_file.num_row_groups}")
print("-" * 30)

# Iterate through each row group and print its size
for i in range(pq_file.num_row_groups):
    row_group_meta = pq_file.metadata.row_group(i)
    print(f"Row group {i}: {row_group_meta.num_rows} rows")
