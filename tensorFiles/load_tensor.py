import numpy as np
import pandas as pd
import pathlib
import shutil
import os

DIR_LOCATION = "../../PROJDATASET"
ID_LIST = pd.read_csv(f"{DIR_LOCATION}/ostracod_IDs.csv")
NEW_DIR_LOCATION = "../../TENSOR_PROJDATASET"


def load_dataset():
    # Initialize Image DataFrame
    data_df = pd.DataFrame()

    # Load Image Data to DataFrame
    for _, row in ID_LIST.iterrows():
        print(f"Reading Index: {row['OrganismID']}")

        # Load the image into the row
        dir_path = list(pathlib.Path(DIR_LOCATION).glob(f"{row['OrganismID']:03}*"))[0]
        file_paths = pathlib.Path(dir_path).glob("*.tif")
        for file in file_paths:
            headers = np.array([row["Species"], row["Sex"], row["Stage"], row["location"], str(file)])
            header_df = pd.DataFrame(headers).T
            header_df[2] = header_df[2] if header_df[2].isna else header_df[2].astype(np.uint8)
            row_df = header_df
            data_df = pd.concat([data_df, row_df])
    data_df = data_df.reset_index()
    data_df = data_df.drop("index", axis=1)
    data_df = data_df.dropna()

    # Rename columns
    data_df.columns.values[0] = "species"
    data_df.columns.values[1] = "sex"
    data_df.columns.values[2] = "stage"
    data_df.columns.values[3] = "location"
    data_df.columns.values[4] = "filepath"

    return data_df


def image_dataset(data_df):
    return (data_df.iloc[:, [0, 4]])


data_df = load_dataset()
image_df = image_dataset(data_df)


for _, row in image_df.iterrows():
    source_path = row[1]
    destination_path = os.path.join(f"{NEW_DIR_LOCATION}/{row[0]}", os.path.basename(source_path))
    print(f"Writing to {destination_path}")
    shutil.copy2(source_path, destination_path)
print("Done")