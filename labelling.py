import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from utility import load_config, create_directory, file_exists, delete_file, list_files_in_directory, copy_file

nltk.download('punkt', download_dir='./nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')
nltk.download('wordnet', download_dir='./nltk_data')

class TextPreprocessing:
    def __init__(self, config_path="config.json"):
        self.config = load_config(config_path)
        self.clusters_file = self.config.get("clusters", "")
        self.suitable_models_cluster = os.path.join(self.config.get("output_folder", ""), "AADL/suitable_models_cluster.csv")
        self.suitable_models_data_file = os.path.join(self.config.get("output_folder", ""), "AADL/suitable_models_data.csv")
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Define the Preprocessing folder path
        self.preprocessing_folder = os.path.join(self.config.get("output_folder", ""), "Preprocessing")
        create_directory(self.preprocessing_folder)
        # Define the Top25Report folder path
        self.top25_report_folder = os.path.join(self.preprocessing_folder, "Top25Report")
        create_directory(self.top25_report_folder)

        # Specific words to remove
        self.words_to_remove = ["aadl", "aadlib", "impl"]

    def preprocess(self):
        clusters_df = pd.read_csv(self.clusters_file)
        suitable_models_df = pd.read_csv(self.suitable_models_data_file)
        preprocessed_clusters_file = os.path.join(self.preprocessing_folder, 'preprocessed_clusters.csv')
        preprocessed_suitable_models_file = os.path.join(self.preprocessing_folder, 'preprocessed_suitable_models_data.csv')

        # Preprocess the 'Model' column in clusters.csv
        clusters_df['Model'] = clusters_df['Model'].apply(self.preprocess_model_name)
        # Save CSV after tokenization of model names
        self.save_intermediate_csv(clusters_df, 'tokenized_clusters')

        # Preprocess the 'Model', 'Component', 'Feature', and 'ConnectionInstance' columns in suitable_models_data.csv
        suitable_models_df['Model'] = suitable_models_df['Model'].apply(self.preprocess_column)
        suitable_models_df['Component'] = suitable_models_df['Component'].apply(self.preprocess_column)
        suitable_models_df['Feature'] = suitable_models_df['Feature'].apply(self.preprocess_column)
        suitable_models_df['ConnectionInstance'] = suitable_models_df['ConnectionInstance'].apply(self.preprocess_column)
        self.save_intermediate_csv(suitable_models_df, 'tokenized_suitable_models_data')

        # Remove non-alphanumeric characters (except hyphens and underscores)
        clusters_df['Model'] = clusters_df['Model'].apply(self.remove_non_alphanumeric)
        self.save_intermediate_csv(clusters_df, 'non_alphanumeric_removed_clusters')

        suitable_models_df['Model'] = suitable_models_df['Model'].apply(self.remove_non_alphanumeric)
        suitable_models_df['Component'] = suitable_models_df['Component'].apply(self.remove_non_alphanumeric)
        suitable_models_df['Feature'] = suitable_models_df['Feature'].apply(self.remove_non_alphanumeric)
        suitable_models_df['ConnectionInstance'] = suitable_models_df['ConnectionInstance'].apply(self.remove_non_alphanumeric)
        self.save_intermediate_csv(suitable_models_df, 'non_alphanumeric_removed_suitable_models_data')

        # Remove stopwords (skip empty or NaN entries)
        clusters_df['Model'] = clusters_df['Model'].apply(self.remove_stopwords)
        self.save_intermediate_csv(clusters_df, 'stopwords_removed_clusters')
        
        suitable_models_df['Model'] = suitable_models_df['Model'].apply(self.remove_stopwords)
        suitable_models_df['Component'] = suitable_models_df['Component'].apply(self.remove_stopwords)
        suitable_models_df['Feature'] = suitable_models_df['Feature'].apply(self.remove_stopwords)
        suitable_models_df['ConnectionInstance'] = suitable_models_df['ConnectionInstance'].apply(self.remove_stopwords)
        self.save_intermediate_csv(suitable_models_df, 'stopwords_removed_suitable_models_data')

        # Lemmatize (skip empty or NaN entries)
        clusters_df['Model'] = clusters_df['Model'].apply(self.lemmatize)
        self.save_intermediate_csv(clusters_df, 'lemmatized_clusters')

        suitable_models_df['Model'] = suitable_models_df['Model'].apply(self.lemmatize)
        suitable_models_df['Component'] = suitable_models_df['Component'].apply(self.lemmatize)
        suitable_models_df['Feature'] = suitable_models_df['Feature'].apply(self.lemmatize)
        suitable_models_df['ConnectionInstance'] = suitable_models_df['ConnectionInstance'].apply(self.lemmatize)
        self.save_intermediate_csv(suitable_models_df, 'lemmatized_suitable_models_data')

        # Save the final preprocessed DataFrames back to new CSV files for inspection
        clusters_df.to_csv(preprocessed_clusters_file, index=False)
        suitable_models_df.to_csv(preprocessed_suitable_models_file, index=False)

        print(f"Preprocessed clusters data saved to {preprocessed_clusters_file}")
        print(f"Preprocessed suitable models data saved to {preprocessed_suitable_models_file}")
        
        # Generate the Top25Report
        self.generate_top25_report()

    def preprocess_model_name(self, model_name):
        if isinstance(model_name, str) and model_name.strip():
            model_name = model_name.replace('.aaxl2', '')       
            model_name = model_name.lower()
            model_name = re.sub(r'[_-]', ' ', model_name)
            for word in self.words_to_remove:
                model_name = re.sub(r'\b' + re.escape(word) + r'\b', '', model_name)
            tokens = word_tokenize(model_name)       
            return " ".join(tokens)
        return model_name

    def preprocess_column(self, column_data):
        if isinstance(column_data, str) and column_data.strip():
            column_data = column_data.lower()
            column_data = re.sub(r'[_-]', ' ', column_data)
            for word in self.words_to_remove:
                column_data = re.sub(r'\b' + re.escape(word) + r'\b', '', column_data)
            tokens = word_tokenize(column_data)  
            return " ".join(tokens)
        return column_data

    def remove_stopwords(self, text):
        if isinstance(text, str) and text.strip():
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in self.stop_words]
            return " ".join(tokens)
        return text

    def lemmatize(self, text):
        if isinstance(text, str) and text.strip():
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            return " ".join(tokens)
        return text

    def remove_non_alphanumeric(self, text):
        if isinstance(text, str) and text.strip():
            tokens = word_tokenize(text)
            tokens = [re.sub(r'[^a-zA-Z_-]', '', word) for word in tokens if word]
            return " ".join(tokens)
        return text

    def save_intermediate_csv(self, df, step_name):
        file_path = os.path.join(self.preprocessing_folder, f"{step_name}.csv")
        df.to_csv(file_path, index=False)
        print(f"Intermediate preprocessing results saved to {file_path}")

    def generate_top25_report(self):
        """Generates and saves the Top 25 words from all the preprocessed CSV files."""
        # List of all processed CSV files
        csv_files = [
            os.path.join(self.preprocessing_folder, 'tokenized_clusters.csv'),
            os.path.join(self.preprocessing_folder, 'tokenized_suitable_models_data.csv'),
            os.path.join(self.preprocessing_folder, 'non_alphanumeric_removed_clusters.csv'),
            os.path.join(self.preprocessing_folder, 'non_alphanumeric_removed_suitable_models_data.csv'),
            os.path.join(self.preprocessing_folder, 'stopwords_removed_clusters.csv'),
            os.path.join(self.preprocessing_folder, 'stopwords_removed_suitable_models_data.csv'),
            os.path.join(self.preprocessing_folder, 'lemmatized_clusters.csv'),
            os.path.join(self.preprocessing_folder, 'lemmatized_suitable_models_data.csv')
        ]
        
        # Process each CSV file and create a Top 25 words report
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                # Concatenate all text data from relevant columns
                text_data = " ".join(df["Model"].dropna())  # Get all model data and drop NaN
                # Tokenize the text
                tokens = word_tokenize(text_data.lower())
                # Remove stopwords
                filtered_tokens = [word for word in tokens if word not in self.stop_words and word.isalnum()]
                # Count the frequency of each word
                word_counts = Counter(filtered_tokens)
                # Get the top 25 words
                top25 = word_counts.most_common(25)

                # Save the top 25 words as a plot
                words, counts = zip(*top25)
                plt.figure(figsize=(10, 6))
                plt.barh(words, counts, color='skyblue')
                plt.xlabel('Frequency')
                plt.title(f"Top 25 Words in {os.path.basename(csv_file)}")
                
                # Save the plot to the Top25Report folder
                plot_file = os.path.join(self.top25_report_folder, f"top25_{os.path.basename(csv_file).replace('.csv', '.png')}")
                plt.tight_layout()
                plt.savefig(plot_file)
                plt.close()
                print(f"Top 25 words plot saved to {plot_file}")

