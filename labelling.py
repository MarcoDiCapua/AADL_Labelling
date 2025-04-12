import os
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure that necessary NLTK data is available
nltk.download('punkt', download_dir='./nltk_data')
nltk.download('punkt_tab', download_dir='./nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')
nltk.download('wordnet', download_dir='./nltk_data')

class TextPreprocessing:
    def __init__(self, config_path="config.json"):
        # Load the configuration to get the file paths
        self.config = self.load_config(config_path)
        self.clusters_file = self.config.get("clusters", "input/average_sim/clusters.csv")
        self.suitable_models_data_file = "output/suitable_models_data.csv"
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def load_config(self, config_path):
        # Loads configuration from the config.json file
        import json
        with open(config_path, "r") as file:
            config = json.load(file)
        return config

    def preprocess(self):
        # Load the two CSV files
        clusters_df = pd.read_csv(self.clusters_file)
        suitable_models_df = pd.read_csv(self.suitable_models_data_file)

        # Preprocess the 'Model' column in clusters.csv
        clusters_df['Model'] = clusters_df['Model'].apply(self.preprocess_model_name)

        # Preprocess the 'Model', 'Component', 'Feature', and 'ConnectionInstance' columns in suitable_models_data.csv
        suitable_models_df['Model'] = suitable_models_df['Model'].apply(self.preprocess_model_name)
        suitable_models_df['Component'] = suitable_models_df['Component'].apply(self.preprocess_column)
        suitable_models_df['Feature'] = suitable_models_df['Feature'].apply(self.preprocess_column)
        suitable_models_df['ConnectionInstance'] = suitable_models_df['ConnectionInstance'].apply(self.preprocess_column)

        # Save the preprocessed DataFrames back to new CSV files for inspection
        preprocessed_clusters_file = os.path.join(self.config.get("output_folder", ""), 'preprocessed_clusters.csv')
        preprocessed_suitable_models_file = os.path.join(self.config.get("output_folder", ""), 'preprocessed_suitable_models_data.csv')

        clusters_df.to_csv(preprocessed_clusters_file, index=False)
        suitable_models_df.to_csv(preprocessed_suitable_models_file, index=False)

        print(f"Preprocessed clusters data saved to {preprocessed_clusters_file}")
        print(f"Preprocessed suitable models data saved to {preprocessed_suitable_models_file}")

    def preprocess_model_name(self, model_name):
        # Remove file extension
        model_name = model_name.replace('.aaxl2', '')
        
        # Normalize text to lowercase
        model_name = model_name.lower()

        # Tokenize, remove stopwords, and lemmatize
        tokens = self.tokenize_and_clean(model_name)
        
        return " ".join(tokens)

    def preprocess_column(self, column_data):
        # Check if the value is a string (handle NaN or non-string values)
        if isinstance(column_data, str):
            # Normalize text to lowercase
            column_data = column_data.lower()

            # Tokenize, remove stopwords, and lemmatize
            tokens = self.tokenize_and_clean(column_data)
            
            return " ".join(tokens)
        else:
            # If the value is not a string, return it as is (it could be NaN or numeric)
            return column_data

    def tokenize_and_clean(self, text):
        # Replace underscores and hyphens with spaces
        text = text.replace('_', ' ').replace('-', ' ')  # Convert underscores and hyphens to spaces
        
        # Tokenize the text into words
        tokens = word_tokenize(text)

        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]

        # Lemmatize the words
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        # Remove non-alphanumeric characters (except hyphens and underscores)
        tokens = [re.sub(r'[^a-zA-Z_-]', '', word) for word in tokens if word]

        return tokens
