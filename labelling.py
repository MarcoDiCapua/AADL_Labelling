import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
from utility import load_config, create_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer


nltk.download('punkt', download_dir='./nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')
nltk.download('wordnet', download_dir='./nltk_data')

class TextPreprocessing:
    def __init__(self, config_path="config.json"):
        self.config = load_config(config_path)
        self.clusters_file = self.config.get("clusters", "")
        self.ground_truth = self.config.get("ground_truth", "")
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
            model_name = re.sub(r'[,_-]', ' ', model_name)
            for word in self.words_to_remove:
                model_name = re.sub(r'\b' + re.escape(word) + r'\b', '', model_name)
            tokens = word_tokenize(model_name)       
            return " ".join(tokens)
        return model_name

    def preprocess_column(self, column_data):
        if isinstance(column_data, str) and column_data.strip():
            column_data = column_data.lower()
            column_data = re.sub(r'[,_-]', ' ', column_data)
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
            os.path.join(self.preprocessing_folder, 'preprocessed_clusters.csv'),
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
            os.path.join(self.preprocessing_folder, 'lemmatized_suitable_models_data.csv'),
            os.path.join(self.preprocessing_folder, 'preprocessed_suitable_models_data.csv')
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


class Labelling:
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

        # Apply TF-IDF on the combined column
        print("Applying TF-IDF on preprocessed_suitable_models_data.csv (Component + Feature)...")
        tfidf_by_cluster = self.calculate_tfidf(suitable_models_df['Combined'], suitable_models_df['Cluster'])
        self.save_top_tfidf(tfidf_by_cluster, "Combined_Top_10_TFIDF.csv")

        print("Generating TF-IDF report and plots...")
        self.generate_total_tfidf_report()
        self.generate_summary_report_tfidf()
        self.generate_tfidf_labels()
        self.plot_tfidf_scores()
        self.plot_label_distribution_tfidf()
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
            # Filter the models data belonging to the current cluster
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
            scores = [round(score, 3) for _, score in top_words]
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

    def generate_summary_report_tfidf(self):
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
            clusters_top5 = self.get_top_words_tfidf(clusters_words, clusters_scores)

            # Process for Combined Top 5 Words (TF-IDF)
            combined_top5 = self.get_top_words_tfidf(combined_words, combined_scores)

            # Append the processed data to the list
            tfidf_labels_data.append([
                cluster, 
                ",".join(clusters_top5[0]), 
                ", ".join(map(str, clusters_top5[1])),
                ",".join(combined_top5[0]), 
                ", ".join(map(str, combined_top5[1]))
            ])

        # Create DataFrame and save to CSV
        tfidf_labels_df = pd.DataFrame(tfidf_labels_data, columns=['Cluster', 'Clusters Top Words (TF-IDF)', 'Clusters Scores', 'Combined Top Words (TF-IDF)', 'Combined Scores'])
        output_file = os.path.join(self.TFIDF_folder, 'TFIDF_Labels.csv')
        tfidf_labels_df.to_csv(output_file, index=False)

        print(f"TFIDF_Labels.csv generated and saved to {output_file}")

    def get_top_words_tfidf(self, words, scores):
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
    
    def plot_tfidf_scores(self):
        # Read the TFIDF Labels data files for Clusters and Combined
        clusters_tfidf_df = pd.read_csv(os.path.join(self.TFIDF_folder, 'Clusters_Top_10_TFIDF.csv'))
        combined_tfidf_df = pd.read_csv(os.path.join(self.TFIDF_folder, 'Combined_Top_10_TFIDF.csv'))

        # Select the first 10 clusters for plotting
        selected_clusters = clusters_tfidf_df['Cluster'].head(10)  # Only use the first 10 clusters
        
        # Create a color map for distinguishing clusters
        colors = cm.get_cmap('tab10', len(selected_clusters))  # Use 'tab10' colormap, which has 10 distinct colors
        
        # Create a plot for Clusters_Top_10_TFIDF.csv
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for idx, cluster_id in enumerate(selected_clusters):
            # Get the relevant rows for the current cluster from Clusters_Top_10_TFIDF.csv
            cluster_data = clusters_tfidf_df[clusters_tfidf_df['Cluster'] == cluster_id]

            # Process the 'Clusters Top 10 Words' and 'Clusters Scores'
            cluster_words = cluster_data['Top 10 Words (TF-IDF)'].values[0].split(',')
            cluster_scores = list(map(float, cluster_data['Scores'].values[0].split(',')))

            # Plot for 'Clusters Top 10 Words'
            ax.plot(range(1, len(cluster_scores) + 1), cluster_scores, label=f"Cluster {cluster_id} (Clusters)", marker='o', color=colors(idx))

        # Set labels and title for the Clusters plot
        ax.set_xlabel('Words')
        ax.set_ylabel('Scores')
        ax.set_title('TF-IDF Scores for Top 10 Words by Cluster (Clusters_Top_10_TFIDF)')
        ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plot as PNG for Clusters
        output_file_clusters = os.path.join(self.TFIDF_folder, 'TFIDF_Scores_for_Clusters_Top_10_Words.png')
        plt.savefig(output_file_clusters)
        plt.close()
        print(f"TF-IDF scores plot for Clusters saved to {output_file_clusters}")

        # Create a plot for Combined_Top_10_TFIDF.csv
        fig, ax = plt.subplots(figsize=(12, 8))

        for idx, cluster_id in enumerate(selected_clusters):
            # Get the relevant rows for the current cluster from Combined_Top_10_TFIDF.csv
            combined_data = combined_tfidf_df[combined_tfidf_df['Cluster'] == cluster_id]

            # Process the 'Combined Top 10 Words' and 'Combined Scores'
            if pd.notna(combined_data['Top 10 Words (TF-IDF)'].values[0]):
                combined_words = combined_data['Top 10 Words (TF-IDF)'].values[0].split(',')
                combined_scores = list(map(float, combined_data['Scores'].values[0].split(',')))

                # Plot for 'Combined Top 10 Words'
                ax.plot(range(1, len(combined_scores) + 1), combined_scores, label=f"Cluster {cluster_id} (Combined)", marker='o', color=colors(idx))

        # Set labels and title for the Combined plot
        ax.set_xlabel('Words')
        ax.set_ylabel('Scores')
        ax.set_title('TF-IDF Scores for Top 10 Words by Cluster (Combined_Top_10_TFIDF)')
        ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plot as PNG for Combined
        output_file_combined = os.path.join(self.TFIDF_folder, 'TFIDF_Scores_for_Combined_Top_10_Words.png')
        plt.savefig(output_file_combined)
        plt.close()
        print(f"TF-IDF scores plot for Combined saved to {output_file_combined}")
        
    def plot_label_distribution_tfidf(self):
        # Read the TFIDF Labels data file
        tfidf_labels_df = pd.read_csv(os.path.join(self.TFIDF_folder, 'TFIDF_Labels.csv'))

        # Count the number of words in 'Clusters Top 5 Words (TF-IDF)' and 'Combined Top 5 Words (TF-IDF)'
        tfidf_labels_df['Clusters Top Words Count'] = tfidf_labels_df['Clusters Top Words (TF-IDF)'].apply(self.safe_split)
        tfidf_labels_df['Combined Top Words Count'] = tfidf_labels_df['Combined Top Words (TF-IDF)'].apply(self.safe_split)

        # Create a side-by-side bar chart
        plt.figure(figsize=(12, 6))

        # Set the bar width and the X-axis positions for each group of bars
        bar_width = 0.35
        index = np.arange(len(tfidf_labels_df['Cluster']))

        # Plot the 'Clusters Top Words' and 'Combined Top Words' as separate bars
        plt.bar(index, tfidf_labels_df['Clusters Top Words Count'], bar_width, label='Clusters Top Words', color='skyblue')
        plt.bar(index + bar_width, tfidf_labels_df['Combined Top Words Count'], bar_width, label='Combined Top Words', color='lightcoral')

        # Add the actual values on top of the bars
        for i, v in enumerate(tfidf_labels_df['Clusters Top Words Count']):
            plt.text(i, v + 0.05, str(v), ha='center', va='bottom', fontsize=10)

        for i, v in enumerate(tfidf_labels_df['Combined Top Words Count']):
            plt.text(i + bar_width, v + 0.05, str(v), ha='center', va='bottom', fontsize=10)

        # Set the labels and title
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Number of Labels', fontsize=12)
        plt.title('Label Distribution by Clusters (TF-IDF)', fontsize=14)

        # Set the x-axis ticks to show the cluster numbers (1 to 44)
        plt.xticks(index + bar_width / 2, tfidf_labels_df['Cluster'], rotation=90)

        # Add a legend
        plt.legend()

        # Save the plot as an image
        output_file = os.path.join(self.TFIDF_folder, 'Label_Distribution_TFIDF_Bars_Affiliated.png')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        print(f"Label distribution bar chart saved to {output_file}")

    def safe_split(self, x):
        #Safe split function to handle empty or invalid values
        if isinstance(x, str) and x.strip():
            return len(x.split(','))
        return 0

    def apply_lda(self):
        # Read preprocessed clusters and suitable models data files
        clusters_df = pd.read_csv(os.path.join(self.preprocessing, 'preprocessed_clusters.csv'))
        suitable_models_df = pd.read_csv(os.path.join(self.preprocessing, 'preprocessed_suitable_models_data.csv'))

        # Combine Component and Feature columns
        suitable_models_df['Combined'] = suitable_models_df.apply(
            lambda row: ' '.join([str(row['Component']), str(row['Feature'])]).strip(),
            axis=1
        )
        empty_combined_rows = suitable_models_df[suitable_models_df['Combined'].str.strip() == '']
        if not empty_combined_rows.empty:
            print(f"ATTENZIONE: Le seguenti righe hanno la colonna 'Combined' vuota:")
            print(empty_combined_rows[['Model', 'Component', 'Feature']])

        # Apply LDA on column (Model) in preprocessed_clusters.csv
        print("Applying LDA on preprocessed_clusters.csv (Model names)...")
        lda_cluster = self.calculate_lda(clusters_df['Model'], clusters_df['Cluster'])
        self.save_top_lda(lda_cluster, "Clusters_Top_LDA.csv")
        self.export_perplexity_values("Clusters_Perplexity.csv")

        # Apply LDA on the combined column (Component + Feature) in preprocessed_suitable_models_data.csv
        print("Applying LDA on preprocessed_suitable_models_data.csv (Component + Feature)...")
        lda_suitable_models = self.calculate_lda(suitable_models_df['Combined'], suitable_models_df['Cluster'])
        self.save_top_lda(lda_suitable_models, "Combined_Top_LDA.csv")
        self.export_perplexity_values("Combined_Perplexity.csv")
        
        # Generate the LDA_Labels.csv with combined results from Clusters_Top_LDA and Combined_Top_LDA
        print("Generating LDA_Labels.csv...")
        self.generate_lda_labels()

        # Generate the plots and reports
        print("Generating LDA reports and plots...")
        self.generate_stacked_bar_chart_lda()
        self.plot_perplexity("Clusters_Perplexity")
        self.plot_perplexity("Combined_Perplexity")

    def calculate_lda(self, text_data, clusters):
        lda_by_cluster = {}
        perplexity_data = {}

        if clusters is not None:
            # Apply LDA within each cluster
            unique_clusters = clusters.unique()

            for cluster_id in range(1, self.num_clusters + 1):
                if cluster_id not in unique_clusters:
                    print(f"Cluster {cluster_id} is not present in the data. Skipping LDA calculation.")
                    continue
                # Filter the models data belonging to the current cluster
                cluster_data = text_data[clusters == cluster_id]
                print(f"Cluster {cluster_id}: {len(cluster_data)} models")

                if len(cluster_data) > 0:
                    # Vectorize the data
                    print(f"Applying LDA on Cluster {cluster_id}...")
                    vectorizer = CountVectorizer()
                    X = vectorizer.fit_transform(cluster_data)
                    print(f"Vectorization complete for Cluster {cluster_id}. Shape: {X.shape}")

                    # Find optimal number of topics using perplexity
                    best_num_topics = self.find_optimal_num_topics(X, cluster_data)

                    # Apply LDA
                    lda = LDA(n_components=best_num_topics, random_state=42)
                    lda.fit(X)

                    # Get the top words for each topic
                    top_words = self.get_top_lda_words(lda, vectorizer.get_feature_names_out())

                    lda_by_cluster[cluster_id] = top_words

                    # Store perplexity for each cluster and number of topics
                    perplexity_data[cluster_id] = self.get_perplexity_scores(X)

        self.perplexity_data = perplexity_data  # Store the perplexity data for later export
        return lda_by_cluster

    def find_optimal_num_topics(self, X, cluster_data):
        perplexity_scores = []
        num_topics_range = range(2, 11)  # Test from 2 to 10 topics

        # Calculate perplexity for different numbers of topics
        for num_topics in num_topics_range:
            lda_model = LDA(n_components=num_topics, random_state=42)
            lda_model.fit(X)
            print(f"Fitting LDA model with {num_topics} topics...")
            # Calculate perplexity score
            perplexity_score = self.calculate_perplexity_score(lda_model, cluster_data)
            print(f"Perplexity score for {num_topics} topics: {perplexity_score}")
            perplexity_scores.append((num_topics, perplexity_score))

        # Choose the number of topics with the lowest perplexity
        best_num_topics = min(perplexity_scores, key=lambda x: x[1])[0]
        print(f"Optimal number of topics for this cluster: {best_num_topics}")
        return best_num_topics

    def calculate_perplexity_score(self, lda_model, text_data):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(text_data)
        perplexity = lda_model.perplexity(X)
        print(f"Perplexity for the LDA model: {perplexity}")
        return perplexity

    def get_perplexity_scores(self, X):
        perplexity_scores = {}
        for num_topics in range(2, 11):  # Check perplexity for topics from 2 to 10
            lda_model = LDA(n_components=num_topics, random_state=42)
            lda_model.fit(X)
            perplexity = lda_model.perplexity(X)
            perplexity_scores[num_topics] = perplexity
        return perplexity_scores

    def get_top_lda_words(self, lda, feature_names, min_words=5, threshold=2.0): 
        #min_words identify the minimum amount of words to be selected for each topic
        #threshold is the minimum probability for a word to be selected
        # Get the top words for each topic in the LDA model
        top_words = []
        
        # Get top words for each LDA component
        for topic_idx, topic in enumerate(lda.components_):
            # Sort words by their probability (weight)
            top_indices = topic.argsort()[-min_words:][::-1]
            print(f"\nTopic {topic_idx}:")
            print("Top word indices and probabilities:", list(zip(top_indices, topic[top_indices])))

            # Filter words based on the threshold probability
            topic_words = []
            for idx in top_indices:
                if topic[idx] >= threshold:
                    topic_words.append(feature_names[idx])
            print(f"Words selected above threshold {threshold}: {topic_words}")

            # If we have fewer words than the min_words, add additional words based on the remaining probabilities
            if len(topic_words) < min_words:
                print(f"Not enough words above threshold. Adding words to reach {min_words}.")
                additional_words = [feature_names[i] for i in topic.argsort()[:-min_words-1:-1] if feature_names[i] not in topic_words]
                topic_words.extend(additional_words)
                print(f"Additional words added: {additional_words}")
            
            # Ensure we don't exceed the max_words
            topic_words = topic_words[:min_words]
            print(f"Final words for topic {topic_idx}: {topic_words}")
            
            top_words.append(topic_words)
        
        return top_words

    def save_top_lda(self, lda_results, file_name):
        # Save the top 10 LDA words and their topics to a CSV file
        output_file = os.path.join(self.LDA_folder, file_name)
        data = []
        for cluster_id, top_words in lda_results.items():
            words = [" ".join(words) for words in top_words]  # Combine top words for each topic
            data.append([cluster_id, ", ".join(words)])

        df = pd.DataFrame(data, columns=['Cluster', 'Top Topics (LDA)'])
        df.to_csv(output_file, index=False)
        print(f"Top LDA words and topics saved to {output_file}")

    def export_perplexity_values(self, file_name):
        # Create a dataframe to store the perplexity values for each cluster and topic number
        perplexity_list = []

        # Create rows for each cluster
        for cluster_id in range(1, self.num_clusters + 1):
            if cluster_id in self.perplexity_data:
                row = {'Cluster': cluster_id}
                for num_topics, perplexity in self.perplexity_data[cluster_id].items():
                    row[f'Topics {num_topics}'] = perplexity
                perplexity_list.append(row)

        # Convert the list to a DataFrame and save it to a CSV file
        perplexity_df = pd.DataFrame(perplexity_list)
        output_file = os.path.join(self.LDA_folder, file_name)
        perplexity_df.to_csv(output_file, index=False)
        print(f"Perplexity values saved to {output_file}")

    def generate_lda_labels(self):
        clusters_df = pd.read_csv(os.path.join(self.LDA_folder, 'Clusters_Top_LDA.csv'))
        combined_df = pd.read_csv(os.path.join(self.LDA_folder, 'Combined_Top_LDA.csv'))

        # Merge the two dataframes on the 'Cluster' column
        merged_df = pd.merge(clusters_df[['Cluster', 'Top Topics (LDA)']], 
                             combined_df[['Cluster', 'Top Topics (LDA)']], 
                             on='Cluster', 
                             suffixes=('_Clusters', '_Combined'), 
                             how='left')

        # Save the merged dataframe to LDA_Labels.csv
        lda_labels_file = os.path.join(self.LDA_folder, 'LDA_Labels.csv')
        merged_df.to_csv(lda_labels_file, index=False)

        print(f"LDA_Labels.csv has been saved to {lda_labels_file}")

    def generate_stacked_bar_chart_lda(self):
        clusters_df = pd.read_csv(os.path.join(self.LDA_folder, 'Clusters_Top_LDA.csv'))
        combined_df = pd.read_csv(os.path.join(self.LDA_folder, 'Combined_Top_LDA.csv'))

        # Merge the two dataframes on the 'Cluster' column
        merged_df = pd.merge(clusters_df[['Cluster', 'Top Topics (LDA)']], 
                            combined_df[['Cluster', 'Top Topics (LDA)']], 
                            on='Cluster', suffixes=('_Clusters', '_Combined'), how='left')

        # Count the number of topics for each cluster in both 'Clusters' and 'Combined'
        merged_df['Topics_Clusters'] = merged_df['Top Topics (LDA)_Clusters'].apply(lambda x: len(x.split(',')))       
        # If a cluster doesn't have topics in Combined_Top_LDA, set Topics_Combined to 0
        merged_df['Topics_Combined'] = merged_df['Top Topics (LDA)_Combined'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)

        # Create a stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.35 
        x_pos = merged_df['Cluster'] 
        ax.bar(x_pos - bar_width/2, merged_df['Topics_Clusters'], label='Clusters Topics', color='skyblue', width=bar_width)
        ax.bar(x_pos + bar_width/2, merged_df['Topics_Combined'], label='Combined Topics', color='lightcoral', width=bar_width)
        for i in range(len(merged_df)):
            ax.text(x_pos[i] - bar_width/2, merged_df['Topics_Clusters'][i] + 0.1, 
                    str(merged_df['Topics_Clusters'][i]), ha='center', va='bottom', fontsize=10)
            ax.text(x_pos[i] + bar_width/2, merged_df['Topics_Combined'][i] + 0.1, 
                    str(merged_df['Topics_Combined'][i]), ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Topics')
        ax.set_title('Topics Distribution by Cluster')
        ax.set_xticks(range(1, 45))
        ax.set_xticklabels(range(1, 45))
        ax.legend()

        output_file = os.path.join(self.LDA_folder, "Topics_Distribution_by_Cluster.png")
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Stacked bar chart saved to {output_file}")

    def plot_perplexity(self, file_name):
        # Read the Perplexity data for the given file
        clusters_perplexity_df = pd.read_csv(os.path.join(self.LDA_folder, file_name + '.csv'))

        # Limit to the first 10 clusters for a clearer example
        selected_clusters = clusters_perplexity_df['Cluster'].head(10)
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot data for each cluster
        for cluster_id in selected_clusters:
            cluster_data = clusters_perplexity_df[clusters_perplexity_df['Cluster'] == cluster_id]
            ax.plot(cluster_data.columns[1:], cluster_data.iloc[0, 1:], label=f"Cluster {cluster_id}", marker='o')  # Add markers

        # Add labels and title
        ax.set_xlabel("Number of Topics")
        ax.set_ylabel("Perplexity")
        ax.set_title(f"Perplexity vs Number of Topics ({file_name})")  # Add the file name in the title

        # Set up legend
        ax.legend(title="Cluster", loc='upper left', bbox_to_anchor=(1, 1))

        # Adjust layout
        plt.tight_layout()

        # Save the plot as a PNG file
        output_file = os.path.join(self.LDA_folder, file_name + "_10_Clusters.png")
        plt.savefig(output_file)
        print(f"Perplexity plot saved to {output_file}")