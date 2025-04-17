from pathlib import Path

import cudf as cdf
import pandas as pd
import numpy as np

import cv2

DIR_LOCATION = "/dev/shm/PROJDATASET"
ID_LIST = pd.read_csv(f"{DIR_LOCATION}/ostracod_IDs.csv")

def load_dataset():
    # Initialize Image DataFrame
    data_df = pd.DataFrame()

    # Temp DataFrames for conversion
    header_df = pd.DataFrame()
    image_df = pd.DataFrame()

    # Load Image Data to DataFrame
    for _, row in ID_LIST.iterrows():
        print(f"Reading Index: {row['OrganismID']}")

        # Load the image into the row
        dir_path = list(Path(DIR_LOCATION).glob(f"{row['OrganismID']:03}*"))[0]
        file_paths = Path(dir_path).glob("*.tif")
        for file in file_paths:
            # Populate Headers
            headers = np.array([row["Species"], row["Sex"], str(row["Stage"]), row["location"]])
            header_df = pd.DataFrame(headers).T

            # Populate Image
            resized_img = np.array(cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (1500, 1000)).flatten()).astype(np.uint8)
            img_df = pd.DataFrame(resized_img).T

            # Append to full datadf
            row_df = pd.concat([header_df, img_df], axis=1)
            data_df = pd.concat([data_df, row_df])

    # Data Cleanup
    data_df = data_df.reset_index()
    data_df = data_df.drop("index", axis=1)
    
    # Rename columns
    data_df.columns.values[0] = "species"
    data_df.columns.values[1] = "sex"
    data_df.columns.values[2] = "stage"
    data_df.columns.values[3] = "location"

    data_df['stage'] = data_df['stage'].astype(np.float32)
    data_df = data_df.dropna()
    data_df['stage'] = data_df['stage'].astype(np.uint8)


    return data_df

