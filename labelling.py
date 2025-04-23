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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np
from sklearn.preprocessing import LabelEncoder

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

        # Define the Preprocessing folder path and Top25Report folder path
        self.preprocessing_folder = os.path.join(self.config.get("output_folder", ""), "Preprocessing")
        create_directory(self.preprocessing_folder)
        self.top25_report_folder = os.path.join(self.preprocessing_folder, "Top25Report")
        create_directory(self.top25_report_folder)

        # Specific words to remove
        self.words_to_remove = ["aadl", "aadlib", "aadlprojects",  "impl"] #forse anche "this", "main", "system", "subsystem" sono da rimuovere

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

        # Note: The stopwords removal is commented out for suitable models as it does not remove a lot of words
        # suitable_models_df['Model'] = suitable_models_df['Model'].apply(self.remove_stopwords)
        # suitable_models_df['Component'] = suitable_models_df['Component'].apply(self.remove_stopwords)
        # suitable_models_df['Feature'] = suitable_models_df['Feature'].apply(self.remove_stopwords)
        # suitable_models_df['ConnectionInstance'] = suitable_models_df['ConnectionInstance'].apply(self.remove_stopwords)
        # self.save_intermediate_csv(suitable_models_df, 'stopwords_removed_suitable_models_data')

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
        
        # Generate the Top25Report for all processed CSV files
        print("Generating Top 25 words report...")
        self.generate_top25_report()
        print("Top 25 words report generated successfully.")

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
            tokens = [re.sub(r'[^a-zA-Z_-]', '', word) if len(word) > 2 or len(word) == 1 else word for word in tokens if word]
            return " ".join(tokens)
        return text

    def save_intermediate_csv(self, df, step_name):
        file_path = os.path.join(self.preprocessing_folder, f"{step_name}.csv")
        df.to_csv(file_path, index=False)
        print(f"Intermediate preprocessing results saved to {file_path}")

    def generate_top25_report(self):
 
        csv_files = [
            os.path.join(self.preprocessing_folder, 'tokenized_clusters.csv'),
            os.path.join(self.preprocessing_folder, 'non_alphanumeric_removed_clusters.csv'),
            os.path.join(self.preprocessing_folder, 'stopwords_removed_clusters.csv'),
            os.path.join(self.preprocessing_folder, 'lemmatized_clusters.csv'),
        ]
        
        # Process each CSV file and create a Top 25 words report for model names
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                # Concatenate all text data from relevant columns
                text_data = " ".join(df["Model"].dropna())  # Get all model data and drop NaN
                tokens = word_tokenize(text_data)
                # Count the frequency of each word and get the top 25
                word_counts = Counter(tokens)
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

        csv_files_data = [
            os.path.join(self.preprocessing_folder, 'tokenized_suitable_models_data.csv'),
            os.path.join(self.preprocessing_folder, 'non_alphanumeric_removed_suitable_models_data.csv'),
            #os.path.join(self.preprocessing_folder, 'stopwords_removed_suitable_models_data.csv'),
            os.path.join(self.preprocessing_folder, 'lemmatized_suitable_models_data.csv')
        ]
    
        # Create subfolders for each column (Component, Feature, ConnectionInstance)
        component_folder = os.path.join(self.top25_report_folder, "Component")
        feature_folder = os.path.join(self.top25_report_folder, "Feature")
        connection_folder = os.path.join(self.top25_report_folder, "ConnectionInstance")

        create_directory(component_folder)
        create_directory(feature_folder)
        create_directory(connection_folder)

        # Process each CSV file and create a Top 25 words report for component, feature, and connection instance
        for csv_file in csv_files_data:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                
                # Create an empty list to store the results
                top25_data = []

                # For each relevant column (Component, Feature, ConnectionInstance), create a top 25 words report
                for column in ['Component', 'Feature', 'ConnectionInstance']:
                    text_data = " ".join(df[column].dropna())  # Get data for the specific column and drop NaN
                    tokens = word_tokenize(text_data)
                    word_counts = Counter(tokens)
                    top25 = word_counts.most_common(25)

                    # Save the top 25 words and their counts for this column
                    top25_data.append({
                        'Column': column,
                        'Top 25 Words': [word for word, _ in top25],
                        'Counts': [count for _, count in top25]
                    })
                    
                    # Plot the top 25 words for this column
                    words, counts = zip(*top25)
                    plt.figure(figsize=(10, 6))
                    plt.barh(words, counts, color='skyblue')
                    plt.xlabel('Frequency')
                    plt.title(f"Top 25 Words in {column} - {os.path.basename(csv_file)}")
                    
                    # Save the plot to the appropriate subfolder
                    if column == 'Component':
                        plot_file = os.path.join(component_folder, f"top25_{column}_{os.path.basename(csv_file).replace('.csv', '.png')}")
                    elif column == 'Feature':
                        plot_file = os.path.join(feature_folder, f"top25_{column}_{os.path.basename(csv_file).replace('.csv', '.png')}")
                    else:
                        plot_file = os.path.join(connection_folder, f"top25_{column}_{os.path.basename(csv_file).replace('.csv', '.png')}")

                    plt.tight_layout()
                    plt.savefig(plot_file)
                    plt.close()
                    print(f"Top 25 words plot for {column} saved to {plot_file}")

                # Create a CSV file to store the top 25 words for each column
                top25_report_file = os.path.join(self.top25_report_folder, f"top25_words_{os.path.basename(csv_file).replace('.csv', '.csv')}")
                report_data = []
                for data in top25_data:
                    for word, count in zip(data['Top 25 Words'], data['Counts']):
                        report_data.append([data['Column'], word, count])
                
                # Write the data to the CSV
                top25_report_df = pd.DataFrame(report_data, columns=['Column', 'Word', 'Count'])
                top25_report_df.to_csv(top25_report_file, index=False)
                print(f"Top 25 words report saved to {top25_report_file}")


class Labeling:
    def __init__(self, config_path="config.json"):
        self.config = load_config(config_path)
        self.clusters_file = self.config.get("clusters", "")
        self.suitable_models_data_file = os.path.join(self.config.get("output_folder", ""), "AADL/suitable_models_data.csv")
        self.preprocessing = os.path.join(self.config.get("output_folder", ""), "Preprocessing")
        self.num_clusters = 44  # Total number of clusters

        # Define the Algorithm, TF-IDF, LDA and Chi-Square folder path
        self.algorithm_folder = os.path.join(self.config.get("output_folder", ""), "Algorithm")
        create_directory(self.algorithm_folder)
        self.TFIDF_folder = os.path.join(self.algorithm_folder, "TF-IDF")
        create_directory(self.TFIDF_folder)
        self.LDA_folder = os.path.join(self.algorithm_folder, "LDA")
        create_directory(self.LDA_folder)
        self.ChiSquare_folder = os.path.join(self.algorithm_folder, "ChiSquare")
        create_directory(self.ChiSquare_folder)

    def apply_tfidf(self):
        # Read preprocessed clusters and suitable models data files
        clusters_df = pd.read_csv(os.path.join(self.preprocessing, 'preprocessed_clusters.csv'))
        suitable_models_df = pd.read_csv(os.path.join(self.preprocessing, 'preprocessed_suitable_models_data.csv'))

        # Apply TF-IDF on preprocessed_clusters.csv (all models names)
        print("Applying TF-IDF on preprocessed_clusters.csv...")
        tfidf_cluster = self.calculate_tfidf(clusters_df['Model'], clusters_df['Cluster'])
        self.save_top_tfidf(tfidf_cluster, "Clusters_Top_10_TFIDF.csv")

        # Apply TF-IDF on preprocessed_suitable_models_data.csv (only suitable models names)
        print("Applying TF-IDF on preprocessed_suitable_models_data.csv...")
        tfidf_suitable_models = self.calculate_tfidf(suitable_models_df['Model'], suitable_models_df['Cluster'])
        self.save_top_tfidf(tfidf_suitable_models, "Suitable_Models_Top_10_TFIDF.csv")

        # Combine Component, Feature, and ConnectionInstance columns into one
        print("Combining Component, Feature, and ConnectionInstance columns for TF-IDF...")
        suitable_models_df['Combined'] = suitable_models_df.apply(
            lambda row: ' '.join([str(row['Component']), str(row['Feature']), str(row['ConnectionInstance'])]).strip(),
            axis=1
        )
        empty_combined_rows = suitable_models_df[suitable_models_df['Combined'].str.strip() == '']
        if not empty_combined_rows.empty:
            print(f"ATTENZIONE: Le seguenti righe hanno la colonna 'Combined' vuota:")
            print(empty_combined_rows[['Model', 'Component', 'Feature', 'ConnectionInstance']])

        # Apply TF-IDF on the combined column
        tfidf_by_cluster = self.calculate_tfidf(suitable_models_df['Combined'], suitable_models_df['Cluster'])
        self.save_top_tfidf(tfidf_by_cluster, "Combined_Top_10_TFIDF.csv")

    def calculate_tfidf(self, text_data, clusters):
        """Calculate the TF-IDF for each cluster separately"""
        tfidf_by_cluster = {}

        # Apply TF-IDF within each cluster
        for cluster_id in range(1, self.num_clusters + 1):
            # Filter the data for the current cluster
            cluster_data = text_data[clusters == cluster_id]
            print(f"Cluster {cluster_id}: {cluster_data}")  # Debugging line to check data

            # Verifica se tutte le parole di una riga sono corte
            cluster_data = cluster_data.apply(self.filter_short_words)
            print(f"Filtered Cluster {cluster_id}: {cluster_data}")  # Debugging line to check data after filtering

            # Escludi righe con solo parole corte
            cluster_data = cluster_data[cluster_data.str.len() > 0]  # Rimuovi righe vuote
            if cluster_data.empty:
                print(f"Warning: Cluster {cluster_id} contains only short words. Skipping TF-IDF calculation for this cluster.")
                continue  # Salta il calcolo del TF-IDF per il cluster se non ci sono parole valide

            tfidf_matrix, feature_names = self.compute_tfidf(cluster_data)
            print(f"TF-IDF Matrix for Cluster {cluster_id}: {tfidf_matrix.shape}")  # Debugging line

            # Check if the matrix is empty or all zeros
            if tfidf_matrix.shape[0] == 0 or np.sum(tfidf_matrix.toarray()) == 0:
                print(f"Warning: TF-IDF matrix for Cluster {cluster_id} is empty or contains only zeros.")

            tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

            # Sort words by their TF-IDF scores and get the top 10
            word_scores = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
            sorted_word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
            top_words = sorted_word_scores[:10]  # Top 10 words by TF-IDF score

            tfidf_by_cluster[cluster_id] = top_words

        return tfidf_by_cluster

    def filter_short_words(self, text):
        """Remove words with length < 3"""
        if isinstance(text, str):
            words = text.split()
            words = [word for word in words if len(word) >= 4]  # Rimuove le parole brevi che non sono significative
            return ' '.join(words)
        return text

    def compute_tfidf(self, text_data):
        """Compute TF-IDF matrix for the given text data"""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(text_data)
        feature_names = vectorizer.get_feature_names_out()  # Get the feature names (words)
        return tfidf_matrix, feature_names

    def save_top_tfidf(self, top_tfidf, file_name):
        """Save the top 10 TF-IDF words and their scores to a CSV file"""
        output_file = os.path.join(self.TFIDF_folder, file_name)
        # Create a dataframe for easier saving with pandas
        data = []
        for cluster_id, top_words in top_tfidf.items():
            words = [word for word, _ in top_words]
            scores = [score for _, score in top_words]
            data.append([cluster_id, ", ".join(words), ", ".join(map(str, scores))])
        
        df = pd.DataFrame(data, columns=['Cluster', 'Top 10 Words (TF-IDF)', 'Scores'])
        df.to_csv(output_file, index=False)

        print(f"Top 10 TF-IDF words and scores saved to {output_file}")

    def visualize_top_words(self, top_tfidf, title):
        """Generate a bar plot for the top TF-IDF words"""
        for cluster_id in range(1, self.num_clusters + 1):
            fig, ax = plt.subplots(figsize=(8, 6))

            # Top Words
            words, scores = zip(*top_tfidf[cluster_id])

            ax.barh(words, scores, color='skyblue')
            ax.set_xlabel("TF-IDF Score")
            ax.set_title(f"{title} - Cluster {cluster_id}")

            plt.tight_layout()
            plt.savefig(os.path.join(self.TFIDF_folder, f"Cluster_{cluster_id}_{title}_Top_10.png"))
            plt.close()
            print(f"Top 10 TF-IDF words plot for Cluster {cluster_id} saved.")
