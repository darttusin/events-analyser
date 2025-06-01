import pandas as pd
import numpy as np

import multiprocessing as mp

import json
from tqdm import tqdm

N_PROCESS = 35
CLICKS_PATH = "../data/clicks.csv"
OFFERS_PATH = "../data/offers.csv"

print('Reading data...')
clicks = pd.read_csv(CLICKS_PATH)
offers = pd.read_csv(OFFERS_PATH)

good_iterations = clicks.iteration_id.unique()
iterations = {}

def process_frame(frame: pd.DataFrame) -> dict:
    frame_iterations = {}
    for _, row in tqdm(frame.iterrows(), total=len(frame)):
        iteration_id = row["iteration_id"]

        if row["user"] not in frame_iterations:
            frame_iterations[row["user"]] = {}

        frame_iterations[row["user"]][iteration_id] = {
            "label": iteration_id in good_iterations,
            "date": row["timestamp"],
            "user": row["user"],
        }
    
    return frame_iterations


with mp.Pool(N_PROCESS) as p:
    data = p.map(process_frame, np.array_split(offers, N_PROCESS))

    iterations = {}

    for part in data:
        for user in tqdm(part):
            if user not in iterations:
                iterations[user] = {}
            
            for iter_id, iter_data in part[user].items():
                iterations[user][iter_id] = iter_data
json.dump(
    iterations,
    open("data/iterations_new.json", "w"),
    indent=4,
)


print('Merging offers...')
offers_df = (
    offers.merge(clicks, on=["iteration_id", "user"], how="left")
    .drop_duplicates(subset=["user", "timestamp"])
    .drop(["iteration_id"], axis=1)
)

print('Make target...')
offers_df["domain"] = np.where(
    offers_df["click_date"].isna(), "OFFER_DECLINED", "OFFER_ACCEPTED"
)

print('Clearing rows...')
offers_df.drop("click_date", axis=1, inplace=True)
offers_df.rename(columns={"timestamp": "date"}, inplace=True)

print('Saving...')
offers_df.to_csv("prepare/data/offer_events_new.csv", index=False)

