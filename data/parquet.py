import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Danh sách các file CSV và file Parquet tương ứng
files = [
    ("application_train.csv", "application_train.parquet"),
    ("application_test.csv", "application_test.parquet")
]

for csv_file_path, parquet_file_path in files:
    # Đọc file CSV
    df = pd.read_csv(csv_file_path)

    # Chuyển đổi DataFrame thành table của Arrow
    table = pa.Table.from_pandas(df)

    # Ghi table thành file Parquet
    pq.write_table(table, parquet_file_path)

    print(f"Chuyển đổi {csv_file_path} sang {parquet_file_path} thành công!")
