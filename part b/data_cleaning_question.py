from scipy.sparse import load_npz

import numpy as np
import csv
import os
from typing import Dict, List

def _load_question_meta_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {}
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data[int(row[0])] = [int(i) for i in row[1].strip('"').strip(']').strip("[").split(", ")]
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data
# /data/subject_meta.csv

def load_question_meta(root_dir="./data"):
    path = os.path.join(root_dir, "question_meta.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    data = _load_question_meta_csv(path)
    return data

def sort_by_question(data):
    # Sort the data by question_id
    # write into a new csv file called question_meta_sorted.csv
    sorted_items = sorted(data.items(), key=lambda item: item[1])
    sorted_dict = dict(sorted_items)
    final = remove_0_from_list(sorted_dict)
    return final

def remove_0_from_list(data: Dict[int, List[int]]):
    # Remove 0 from the data
    # write into a new csv file called question_meta_sorted_no_duplicates.csv
    return {k: [i for i in v if i != 0] for k, v in data.items()}

def write_to_csv(data, path = "./data/question_meta_sorted.csv"):
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in data.items():
            writer.writerow([key, value])

def main():
    data = load_question_meta()
    sorted_data = sort_by_question(data)
    write_to_csv(sorted_data)
    

if __name__ == "__main__":
    main()