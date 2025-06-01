import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import numpy as np

chunksize = 20_000_000
final_parquet = "data/final_data.parquet"


pqwriter = None


def init_pqwriter(table):
    global pqwriter
    if pqwriter is None:
        try:
            os.remove(final_parquet)
        except:
            pass
        pqwriter = pq.ParquetWriter(final_parquet, table.schema)


def process_and_write_chunk(chunk):
    global pqwriter

    grouped = chunk.groupby("iteration_id")

    processed_data = []

    for iteration_id, group in grouped:
        if (
            iteration_id == chunk["iteration_id"].iloc[0]
            or iteration_id == chunk["iteration_id"].iloc[-1]
        ):
            continue

        target = group["label"].iloc[0]
        seq_len = len(group)
        numerical_features = np.array(group["time_diff"], dtype=float)
        categorical_features = np.array(group["domain"], dtype=object)

        processed_data.append(
            {
                "sequence_id": iteration_id,
                "target": target,
                "user_id": group["user_id"].iloc[0],
                "_seq_len": seq_len,
                "time_diff": np.array(numerical_features, dtype=np.float16),
                "timestamp": np.array(group["timestamp"]),
                "domain": categorical_features,
            }
        )

    processed_df = pd.DataFrame(processed_data)


    table = pa.Table.from_pandas(processed_df)

    init_pqwriter(table)

    pqwriter.write_table(table)

def main():
    global pqwriter

    for i, chunk in enumerate(
        pd.read_csv("data/prepared_data_new.csv", chunksize=chunksize)
    ):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])


        chunk_first_iteration_id = chunk["iteration_id"].iloc[0]
        chunk_last_iteration_id = chunk["iteration_id"].iloc[-1]
        chunk = chunk[chunk["iteration_id"] != chunk_first_iteration_id]
        chunk = chunk[chunk["iteration_id"] != chunk_last_iteration_id]

        process_and_write_chunk(chunk)
    if pqwriter:
        pqwriter.close()


if __name__ == "__main__":
    main()