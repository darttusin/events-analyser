import pandas as pd
import random
import multiprocessing as mp

import json
from datetime import datetime
from tqdm import tqdm

import numpy as np

import warnings

warnings.filterwarnings("ignore")
tqdm.pandas()

N_PROCESS = 30

OFFERS_NEW_PATH = "data/offer_events_new.csv"
EVENTS_PATH = "data/events.csv"
ITERATION_NEW_PATH = "data/iterations_new.json"

EVENTS_NEW_PATH = "data/prepared_data_new.csv"


print('Reading data...')
offers_df = pd.read_csv(OFFERS_NEW_PATH)
df = pd.read_csv(EVENTS_PATH)
df = df.rename(columns={"timestamp": "date"})


df = df[df["user"].isin(offers_df["user"].unique())]
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

with open(ITERATION_NEW_PATH, 'r') as f:
    iterations = json.load(f)


def process_batch(frame: list) -> list[str]:
    lines = []
    for user, row in tqdm(frame):
        iterations_user = iterations.get(user)
        if iterations_user is None:
            continue

        for iteration_id, val in iterations_user.items():
            date = datetime.fromisoformat(val["date"])
            cutted_row = row[row["date"] < date]

            if cutted_row.empty:
                continue

            cutted_row.sort_values("date", inplace=True, ignore_index=True)
            cutted_row["time_diff"] = (cutted_row["date"]).diff().dt.total_seconds()
            cutted_row.loc[0, "time_diff"] = 0


            label = 1 if val["label"] else 0
            
            for _row in cutted_row[["domain", "time_diff", "date"]].to_dict(
                orient="records"
            ):
                lines.append(
                    f"{val['user']},{iteration_id},{_row['domain']},{_row['time_diff']},{_row['date']},{label}\n"
                )
    return lines


with mp.Pool(N_PROCESS) as p:
    groups = [(user, group) for user, group in tqdm(df.groupby("user"), desc="Prepare")]
    chunk_size = (len(groups) // N_PROCESS) + 1
    chunks = [
        groups[i:i + chunk_size] 
        for i in range(0, len(groups), chunk_size)
    ]
    
    del groups
    del df
    del offers_df

    print('Starting processing...')
    all_lines = p.map(process_batch, chunks)

with open(EVENTS_NEW_PATH, "w") as file: 
    file.write("user_id,iteration_id,domain,time_diff,timestamp,label\n") 

    for lines in tqdm(all_lines):
        for line in lines: 
            file.write(line)