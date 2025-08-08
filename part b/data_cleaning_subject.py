from scipy.sparse import load_npz
import numpy as np
import csv
import os
from typing import Dict, List

def _load_subject_meta_csv(path):
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
                data[int(row[0])] = row[1]
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data
# /data/subject_meta.csv

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
    path = os.path.join(root_dir, "question_meta_sorted.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    data = _load_question_meta_csv(path)
    return data

# I want to load the question_meta.csv file and write a function
# call sort_by_question to sort the data by subject_id

def load_subject_meta_csv(root_dir="./data"):
    path = os.path.join(root_dir, "subject_meta.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    data = _load_subject_meta_csv(path)
    return data

def subject_that_exist(question_meta: Dict[int, List[int]]):
    # Sort the data by question_id
    # write into a new csv file called question_meta_sorted.csv
    cleaned_data = set()
    for subject_list in question_meta.values():
        for question_id in subject_list:
            if question_id not in cleaned_data and question_id != 0:
                cleaned_data.add(question_id)
    return cleaned_data

def write_to_csv(path = "data/cleaned_subject_meta.csv"):
    path2 = "./data/clean_question_meta.csv"
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file)
        cleaned_data = subject_that_exist(load_question_meta())
        subject = load_subject_meta_csv()
        question = load_question_meta()
        for i, subject_id in enumerate(cleaned_data):
            writer.writerow([i, subject[subject_id]])
            for j in range(len(question.keys())):
                if subject_id in question[j]:
                    question[j] = [i if s == subject_id else s for s in question[j]]
        with open(path2, "w") as csv_file2:
            writer2 = csv.writer(csv_file2)
            for key, value in question.items():
                writer2.writerow([key, value])
                        

def main():
    write_to_csv()
    

if __name__ == "__main__":
    main()