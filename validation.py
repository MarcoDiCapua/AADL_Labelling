import os
import shutil
import pandas as pd
from utility import load_config, create_directory, copy_file

class Validation:
    def __init__(self, config_path="config.json"):
        self.config = load_config(config_path)
        self.ground_truth_file = self.config.get("ground_truth", "")
        self.output_folder = self.config.get("output_folder", "")
        
        # Paths for the generated labels files
        self.tfidf_labels_file = os.path.join(self.output_folder, "Algorithm\TF-IDF\TFIDF_Labels.csv")
        self.lda_labels_file = os.path.join(self.output_folder, "Algorithm\LDA\LDA_Labels.csv")

        # Create the validation folder and perform all setup
        self.validation_folder = os.path.join(self.output_folder, "Validation")

        # Create the validation folder if it does not exist and copy the ground truth file
        create_directory(self.validation_folder)

        if os.path.exists(self.ground_truth_file):
            copy_file(self.ground_truth_file, self.validation_folder)
            print(f"Ground truth file copied to {self.validation_folder}")
        else:
            print(f"Ground truth file not found at {self.ground_truth_file}")

    def validate_TFIDF_labels(self):
        # Load the TF-IDF labels files
        if os.path.exists(self.tfidf_labels_file):
            self.tfidf_df = pd.read_csv(self.tfidf_labels_file)
            print(f"TF-IDF labels loaded from {self.tfidf_labels_file}")
        else:
            print(f"TF-IDF labels file not found at {self.tfidf_labels_file}")
            self.tfidf_df = None

        # Load the ground truth file as a DataFrame
        if os.path.exists(self.ground_truth_file):
            self.ground_truth_df = pd.read_excel(self.ground_truth_file)
            print(f"Ground truth file loaded from {self.ground_truth_file}")
        else:
            print(f"Ground truth file not found at {self.ground_truth_file}")
            self.ground_truth_df = None

    def validate_LDA_labels(self):
        # Load the LDA labels file as a DataFrame
        if os.path.exists(self.lda_labels_file):
            self.lda_df = pd.read_csv(self.lda_labels_file)
            print(f"LDA labels loaded from {self.lda_labels_file}")
        else:
            print(f"LDA labels file not found at {self.lda_labels_file}")
            self.lda_df = None

        # Load the ground truth file as a DataFrame
        if os.path.exists(self.ground_truth_file):
            self.ground_truth_df = pd.read_excel(self.ground_truth_file)
            print(f"Ground truth file loaded from {self.ground_truth_file}")
        else:
            print(f"Ground truth file not found at {self.ground_truth_file}")
            self.ground_truth_df = None