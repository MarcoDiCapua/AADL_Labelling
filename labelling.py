import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
from utility import load_config, create_directory, file_exists, delete_file, list_files_in_directory, copy_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
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
        self.words_to_remove = ["aadl", "aadlib", "aadlprojects",  "impl", "this", "main", "system", "subsystem"]

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
        #suitable_models_df['ConnectionInstance'] = suitable_models_df['ConnectionInstance'].apply(self.preprocess_column)
        self.save_intermediate_csv(suitable_models_df, 'tokenized_suitable_models_data')

        # Remove non-alphanumeric characters (except hyphens and underscores)
        clusters_df['Model'] = clusters_df['Model'].apply(self.remove_non_alphanumeric)
        self.save_intermediate_csv(clusters_df, 'non_alphanumeric_removed_clusters')

        suitable_models_df['Model'] = suitable_models_df['Model'].apply(self.remove_non_alphanumeric)
        suitable_models_df['Component'] = suitable_models_df['Component'].apply(self.remove_non_alphanumeric)
        suitable_models_df['Feature'] = suitable_models_df['Feature'].apply(self.remove_non_alphanumeric)
        #suitable_models_df['ConnectionInstance'] = suitable_models_df['ConnectionInstance'].apply(self.remove_non_alphanumeric)
        self.save_intermediate_csv(suitable_models_df, 'non_alphanumeric_removed_suitable_models_data')

        # Remove stopwords (skip empty or NaN entries)
        clusters_df['Model'] = clusters_df['Model'].apply(self.remove_stopwords)
        self.save_intermediate_csv(clusters_df, 'stopwords_removed_clusters')

        # Remove stopwords from suitable models data
        suitable_models_df['Model'] = suitable_models_df['Model'].apply(self.remove_stopwords)
        suitable_models_df['Component'] = suitable_models_df['Component'].apply(self.remove_stopwords)
        suitable_models_df['Feature'] = suitable_models_df['Feature'].apply(self.remove_stopwords)
        #suitable_models_df['ConnectionInstance'] = suitable_models_df['ConnectionInstance'].apply(self.remove_stopwords)
        self.save_intermediate_csv(suitable_models_df, 'stopwords_removed_suitable_models_data')

        # Lemmatize (skip empty or NaN entries)
        clusters_df['Model'] = clusters_df['Model'].apply(self.lemmatize)
        self.save_intermediate_csv(clusters_df, 'lemmatized_clusters')

        suitable_models_df['Model'] = suitable_models_df['Model'].apply(self.lemmatize)
        suitable_models_df['Component'] = suitable_models_df['Component'].apply(self.lemmatize)
        suitable_models_df['Feature'] = suitable_models_df['Feature'].apply(self.lemmatize)
        #suitable_models_df['ConnectionInstance'] = suitable_models_df['ConnectionInstance'].apply(self.lemmatize)
        self.save_intermediate_csv(suitable_models_df, 'lemmatized_suitable_models_data')

        # Remove words of length 1 and remove rows with no valid Model or Component
        clusters_df['Model'] = clusters_df['Model'].apply(self.remove_short_words)
        # Remove rows with no valid Model
        clusters_df = clusters_df[clusters_df['Model'].str.strip().ne('')]

        suitable_models_df['Model'] = suitable_models_df['Model'].apply(self.remove_short_words)
        suitable_models_df['Component'] = suitable_models_df['Component'].apply(self.remove_short_words)
        suitable_models_df['Feature'] = suitable_models_df['Feature'].apply(self.remove_short_words)
        #suitable_models_df['ConnectionInstance'] = suitable_models_df['ConnectionInstance'].apply(self.remove_short_words)
        suitable_models_df = suitable_models_df[suitable_models_df['Component'].str.strip().ne('') | suitable_models_df['Feature'].str.strip().ne('')]#| suitable_models_df['ConnectionInstance'].str.strip().ne('')

        
        # Save the final preprocessed DataFrames back to new CSV files
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

    def remove_short_words(self, text):
        if isinstance(text, str):
            words = text.split()
            words = [word for word in words if len(word) > 1]
            return ' '.join(words)
        return text

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
            os.path.join(self.preprocessing_folder, 'stopwords_removed_suitable_models_data.csv'),
            os.path.join(self.preprocessing_folder, 'lemmatized_suitable_models_data.csv')
        ]
    
        # Create subfolders for each column (Component, Feature, ConnectionInstance)
        component_folder = os.path.join(self.top25_report_folder, "Component")
        feature_folder = os.path.join(self.top25_report_folder, "Feature")
        #connection_folder = os.path.join(self.top25_report_folder, "ConnectionInstance")

        create_directory(component_folder)
        create_directory(feature_folder)
        #create_directory(connection_folder)

        # Process each CSV file and create a Top 25 words report for component, feature, and connection instance
        for csv_file in csv_files_data:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                
                # Create an empty list to store the results
                top25_data = []

                # For each relevant column (Component, Feature), create a top 25 words report
                for column in ['Component', 'Feature']: #, 'ConnectionInstance'
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
                    #else:
                    #    plot_file = os.path.join(connection_folder, f"top25_{column}_{os.path.basename(csv_file).replace('.csv', '.png')}")

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
        self.num_topics = 5  # Number of topics for LDA

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

        # Combine Component and Feature columns
        suitable_models_df['Combined'] = suitable_models_df.apply(
            lambda row: ' '.join([str(row['Component']), str(row['Feature'])]).strip(), # , str(row['ConnectionInstance'])
            axis=1
        )
        empty_combined_rows = suitable_models_df[suitable_models_df['Combined'].str.strip() == '']
        if not empty_combined_rows.empty:
            print(f"ATTENZIONE: Le seguenti righe hanno la colonna 'Combined' vuota:")
            print(empty_combined_rows[['Model', 'Component', 'Feature']]) # , 'ConnectionInstance'

        # Apply TF-IDF on preprocessed_clusters.csv (all models names)
        print("Applying TF-IDF on preprocessed_clusters.csv (Model names)...")
        tfidf_cluster = self.calculate_tfidf(clusters_df['Model'], clusters_df['Cluster'])
        self.save_top_tfidf(tfidf_cluster, "Clusters_Top_10_TFIDF.csv")

        # Apply TF-IDF on preprocessed_suitable_models_data.csv (only suitable models names)
        # print("Applying TF-IDF on preprocessed_suitable_models_data.csv...")
        # tfidf_suitable_models = self.calculate_tfidf(suitable_models_df['Model'], suitable_models_df['Cluster'])
        # self.save_top_tfidf(tfidf_suitable_models, "Suitable_Models_Top_10_TFIDF.csv")

        # Apply TF-IDF on the combined column
        print("Applying TF-IDF on preprocessed_suitable_models_data.csv (Component + Feature)...")
        tfidf_by_cluster = self.calculate_tfidf(suitable_models_df['Combined'], suitable_models_df['Cluster'])
        self.save_top_tfidf(tfidf_by_cluster, "Combined_Top_10_TFIDF.csv")

        print("Generating TF-IDF report and plots...")
        self.generate_total_tfidf_report()
        self.plot_label_distribution()
        self.generate_summary_report()
        self.generate_tfidf_labels()
        print("TF-IDF labels, reports and plots generated successfully.")

    def calculate_tfidf(self, text_data, clusters):
        #Calculate the TF-IDF for each cluster separately
        tfidf_by_cluster = {}
        unique_clusters = clusters.unique()

        # Apply TF-IDF within each cluster if it exists in the data
        for cluster_id in range(1, self.num_clusters + 1):
            if cluster_id not in unique_clusters:
                print(f"Cluster {cluster_id} is not present in the data. Skipping TF-IDF calculation.")
                continue
            # Filter the data for the current cluster
            cluster_data = text_data[clusters == cluster_id]
            #print(f"Cluster {cluster_id}: {cluster_data}")  # Debugging line to check data

            tfidf_matrix, feature_names = self.compute_tfidf(cluster_data)
            #print(f"TF-IDF Matrix for Cluster {cluster_id}: {tfidf_matrix.shape}")  # Debugging line

            # Check if the matrix is empty or all zeros
            if tfidf_matrix.shape[0] == 0 or np.sum(tfidf_matrix.toarray()) == 0:
                print(f"Warning: TF-IDF matrix for Cluster {cluster_id} is empty or contains only zeros.")

            tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

            # Sort words by their TF-IDF scores and get the top 10
            word_scores = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
            sorted_word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
            top_words = sorted_word_scores[:10]

            tfidf_by_cluster[cluster_id] = top_words

        return tfidf_by_cluster

    def compute_tfidf(self, text_data):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(text_data)
        feature_names = vectorizer.get_feature_names_out()  # Get the feature names (words)
        return tfidf_matrix, feature_names

    def save_top_tfidf(self, top_tfidf, file_name):
        output_file = os.path.join(self.TFIDF_folder, file_name)
        data = []
        for cluster_id, top_words in top_tfidf.items():
            words = [word for word, _ in top_words]
            scores = [score for _, score in top_words]
            data.append([cluster_id, ", ".join(words), ", ".join(map(str, scores))])
        
        df = pd.DataFrame(data, columns=['Cluster', 'Top 10 Words (TF-IDF)', 'Scores'])
        df.to_csv(output_file, index=False)

        print(f"Top 10 TF-IDF words and scores saved to {output_file}")

    def generate_total_tfidf_report(self):
        clusters_tfidf_df = pd.read_csv(os.path.join(self.TFIDF_folder, 'Clusters_Top_10_TFIDF.csv'))
        combined_tfidf_df = pd.read_csv(os.path.join(self.TFIDF_folder, 'Combined_Top_10_TFIDF.csv'))
        # Merge the two dataframes based on the cluster ID
        total_tfidf_df = pd.merge(clusters_tfidf_df, combined_tfidf_df, on='Cluster', how='left', suffixes=('_Clusters', '_Combined'))

        total_tfidf_df.rename(columns={
            'Top 10 Words (TF-IDF)_Clusters': 'Clusters Top 10 Words (TF-IDF)',
            'Scores_Clusters': 'Clusters Scores',
            'Top 10 Words (TF-IDF)_Combined': 'Combined Top 10 Words (TF-IDF)',
            'Scores_Combined': 'Combined Scores'
        }, inplace=True)

        total_tfidf_file = os.path.join(self.TFIDF_folder, 'Total_Top_10_TFIDF.csv')
        total_tfidf_df.to_csv(total_tfidf_file, index=False)
        print(f"Total_Top_10_TFIDF.csv generated and saved to {total_tfidf_file}")

    def plot_label_distribution(self):
        total_tfidf_df = pd.read_csv(os.path.join(self.TFIDF_folder, 'Total_Top_10_TFIDF.csv'))

        # Create a new DataFrame with counts for each category (Clusters Top 10 Words and Combined Top 10 Words)
        total_tfidf_df['Clusters Top 10 Words Count'] = total_tfidf_df['Clusters Top 10 Words (TF-IDF)'].apply(self.safe_split)
        total_tfidf_df['Combined Top 10 Words Count'] = total_tfidf_df['Combined Top 10 Words (TF-IDF)'].apply(self.safe_split)

        # Plot the stacked bar chart
        plt.figure(figsize=(10, 6))
        bars1 = plt.bar(total_tfidf_df['Cluster'], total_tfidf_df['Clusters Top 10 Words Count'], label='Clusters Top 10 Words', color='skyblue')
        bars2 = plt.bar(total_tfidf_df['Cluster'], total_tfidf_df['Combined Top 10 Words Count'], label='Combined Top 10 Words', color='lightcoral', bottom=total_tfidf_df['Clusters Top 10 Words Count'])
        # Add the actual values on top of the bars
        for bar in bars1:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, str(int(yval)), ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            yval = bar.get_height() + bar.get_y()
            if bar.get_height() > 0:  # Only add the label if the height is greater than 0
                plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, str(int(bar.get_height())), ha='center', va='bottom', fontsize=10)

        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower'))
        plt.xticks(total_tfidf_df['Cluster'], total_tfidf_df['Cluster'], rotation=90)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Labels')
        plt.title('Label Distribution by Clusters')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.TFIDF_folder, 'label_distribution_stacked.png'))
        plt.close()
        print(f"Label distribution stacked bar chart saved to {os.path.join(self.TFIDF_folder, 'label_distribution_stacked.png')}")

    def safe_split(self, x):
        #Safe split function to handle empty or invalid values
        if isinstance(x, str) and x.strip():
            return len(x.split(','))
        return 0

    def generate_summary_report(self):
        clusters_tfidf_df = pd.read_csv(os.path.join(self.TFIDF_folder, 'Clusters_Top_10_TFIDF.csv'))
        combined_tfidf_df = pd.read_csv(os.path.join(self.TFIDF_folder, 'Combined_Top_10_TFIDF.csv'))
        summary_data = []

        for df, label in [(clusters_tfidf_df, 'Clusters'), (combined_tfidf_df, 'Combined')]:
            for cluster_id in df['Cluster']:
                words = df.loc[df['Cluster'] == cluster_id, 'Top 10 Words (TF-IDF)'].values[0].split(',')
                scores = np.array(list(map(float, df.loc[df['Cluster'] == cluster_id, 'Scores'].values[0].split(','))))
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                summary_data.append([cluster_id, label, len(words), avg_score, std_score])

        # Save summary data to a CSV file
        summary_df = pd.DataFrame(summary_data, columns=['Cluster', 'Label', 'Words Count', 'Avg TF-IDF', 'Std TF-IDF'])
        summary_file = os.path.join(self.TFIDF_folder, 'TFIDF_Summary_Report.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary report saved to {summary_file}")

    def generate_tfidf_labels(self):
        total_tfidf_df = pd.read_csv(os.path.join(self.TFIDF_folder, 'Total_Top_10_TFIDF.csv'))

        # Initialize an empty list to store the new rows for TFIDF_Labels.csv
        tfidf_labels_data = []

        # Iterate over each cluster to process the words and scores
        for idx, row in total_tfidf_df.iterrows():
            cluster = row['Cluster']
            clusters_words = row['Clusters Top 10 Words (TF-IDF)'].split(',')
            clusters_scores = list(map(float, row['Clusters Scores'].split(',')))

            # Check if 'Combined' columns are empty, if so skip processing for Combined
            if pd.isna(row['Combined Top 10 Words (TF-IDF)']) or pd.isna(row['Combined Scores']):
                combined_words = []
                combined_scores = []
            else:
                combined_words = row['Combined Top 10 Words (TF-IDF)'].split(',') 
                combined_scores = list(map(float, row['Combined Scores'].split(','))) 

            # Process for Clusters Top 5 Words (TF-IDF)
            clusters_top5 = self.get_top_words_with_filter(clusters_words, clusters_scores)

            # Process for Combined Top 5 Words (TF-IDF)
            combined_top5 = self.get_top_words_with_filter(combined_words, combined_scores)

            # Append the processed data to the list
            tfidf_labels_data.append([
                cluster, 
                ",".join(clusters_top5[0]), 
                ", ".join(map(str, clusters_top5[1])),
                ",".join(combined_top5[0]), 
                ", ".join(map(str, combined_top5[1]))
            ])

        # Create DataFrame and save to CSV
        tfidf_labels_df = pd.DataFrame(tfidf_labels_data, columns=['Cluster', 'Clusters Top 5 Words (TF-IDF)', 'Clusters Scores', 'Combined Top 5 Words (TF-IDF)', 'Combined Scores'])
        output_file = os.path.join(self.TFIDF_folder, 'TFIDF_Labels.csv')
        tfidf_labels_df.to_csv(output_file, index=False)

        print(f"TFIDF_Labels.csv generated and saved to {output_file}")

    def get_top_words_with_filter(self, words, scores):
        #Filters the top words with score > 1, and if none meet the criteria, takes the highest score word
        filtered_words = []
        filtered_scores = []

        for word, score in zip(words, scores):
            if score > 1:
                filtered_words.append(word)
                filtered_scores.append(score)

        # If no words have score > 1, take the first word and its score
        if not filtered_words and len(words) > 0 and len(scores) > 0:
            filtered_words = [words[0]]
            filtered_scores = [scores[0]]

        if len(filtered_words) < 5:
            return filtered_words, filtered_scores

        return filtered_words[:5], filtered_scores[:5]

    def apply_lda(self):
        # Read preprocessed clusters and suitable models data files
        clusters_df = pd.read_csv(os.path.join(self.preprocessing, 'preprocessed_clusters.csv'))
        suitable_models_df = pd.read_csv(os.path.join(self.preprocessing, 'preprocessed_suitable_models_data.csv'))

        # Apply LDA on preprocessed_clusters.csv (Model names)
        print("Applying LDA on preprocessed_clusters.csv (Model names)...")
        lda_cluster = self.calculate_lda(clusters_df['Model'], clusters_df['Cluster'])
        self.save_top_lda(lda_cluster, "Clusters_Top_10_LDA.csv")

        # Apply LDA on the combined column (Component + Feature) in preprocessed_suitable_models_data.csv
        print("Applying LDA on preprocessed_suitable_models_data.csv (Component + Feature)...")
        suitable_models_df['Combined'] = suitable_models_df.apply(
            lambda row: ' '.join([str(row['Component']), str(row['Feature'])]).strip(), 
            axis=1
        )
        lda_suitable_models = self.calculate_lda(suitable_models_df['Combined'], suitable_models_df['Cluster'])
        self.save_top_lda(lda_suitable_models, "Combined_Top_10_LDA.csv")

    def calculate_lda(self, text_data, clusters):
        lda_by_cluster = {}

        if clusters is not None:
            # Apply LDA within each cluster
            unique_clusters = clusters.unique()

            for cluster_id in range(1, self.num_clusters + 1):
                if cluster_id not in unique_clusters:
                    print(f"Cluster {cluster_id} is not present in the data. Skipping LDA calculation.")
                    continue
                # Filter the data for the current cluster
                cluster_data = text_data[clusters == cluster_id]
                print(f"Cluster {cluster_id}: {len(cluster_data)} models")  # Debugging line to check data

                if len(cluster_data) > 0:
                    # Vectorize the data
                    vectorizer = CountVectorizer()
                    X = vectorizer.fit_transform(cluster_data)

                    # Apply LDA
                    lda = LDA(n_components=self.num_topics, random_state=42)
                    lda.fit(X)

                    # Get the top words for each topic
                    top_words = self.get_top_lda_words(lda, vectorizer.get_feature_names_out(), 10)

                    lda_by_cluster[cluster_id] = top_words
        else:
            # Apply LDA on the entire dataset (no clustering, just all the data)
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(text_data)

            lda = LDA(n_components=self.num_topics, random_state=42)
            lda.fit(X)

            # Get the top words for each topic
            top_words = self.get_top_lda_words(lda, vectorizer.get_feature_names_out(), 10)
            #DA RIVEDERE
            lda_by_cluster['Global'] = top_words

        return lda_by_cluster

    def get_top_lda_words(self, lda, feature_names, n_top_words):
        """Get top words for each topic in LDA"""
        top_words = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-n_top_words:][::-1]
            top_words.append([feature_names[i] for i in top_indices])
        return top_words

    def save_top_lda(self, lda_results, file_name):
        """Save the top 10 LDA words and their topics to a CSV file"""
        output_file = os.path.join(self.LDA_folder, file_name)
        data = []
        for cluster_id, top_words in lda_results.items():
            words = [" ".join(words) for words in top_words]  # Combine top words for each topic
            data.append([cluster_id, ", ".join(words)])

        df = pd.DataFrame(data, columns=['Cluster', 'Top 10 Words (LDA)'])
        df.to_csv(output_file, index=False)

        print(f"Top 10 LDA words and topics saved to {output_file}")