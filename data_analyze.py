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

# count the number of total patterns of the question_meta
def count_total_patterns(question_meta: Dict[int, List[int]]):
    # count the number of total patterns of the question_meta
    existed_patterns = set()
    for value in question_meta.values():
        if tuple(value) not in existed_patterns:
            existed_patterns.add(tuple(value))
        
    return len(existed_patterns)


def main():
    question_meta = load_question_meta()
    print(count_total_patterns(question_meta))

if __name__ == "__main__":
    main()