from pathlib import Path

import pandas as pd

import cv2

DIR_LOCATION = "PROJDATASET"
ID_LIST = pd.read_csv(f"{DIR_LOCATION}/ostracod_IDs.csv")

def load_dataset():
    # Initialize Image DataFrame
    data_df = pd.DataFrame()

    # Load Image Data to DataFrame
    for _, row in ID_LIST.iterrows():
        print(f"Reading Index: {row['OrganismID']}")
        # Load the image into the row
        dir_path = list(Path(DIR_LOCATION).glob(f"{row['OrganismID']:03}*"))[0]
        file_paths = Path(dir_path).glob("*.png")
        for file in file_paths:
            row_data = [row["Species"], row["Sex"], row["Stage"], row["location"]]
            resized_img = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (1500, 1000)).flatten()
            row_data.extend(resized_img)
            data_df = pd.concat([data_df, pd.DataFrame(row_data).T])

    data_df.reset_index()
    data_df.drop("index", axis=1)
    data_df.dropna()
    data_df.to_csv("/dev/shm/bngo.csv", index=False)

load_dataset()