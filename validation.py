import os
import shutil
import pandas as pd
from utility import load_config, create_directory, copy_file
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer  # Importing CountVectorizer
import nltk
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import seaborn as sns

# Download WordNet data
nltk.download('wordnet', download_dir='./nltk_data')

class Validation:
    def __init__(self, config_path="config.json"):
        # Load configuration and initialize paths
        self.config = load_config(config_path)
        self.ground_truth_file = self.config.get("ground_truth", "")
        self.output_folder = self.config.get("output_folder", "")
        
        # Paths for the generated labels files
        self.tfidf_labels_file = os.path.join(self.output_folder, "Algorithm/TF-IDF/TFIDF_Labels.csv")
        self.lda_labels_file = os.path.join(self.output_folder, "Algorithm/LDA/LDA_Labels.csv")

        # Create the validation folder if it does not exist and copy the ground truth file
        self.validation_folder = os.path.join(self.output_folder, "Validation")
        create_directory(self.validation_folder)

        if os.path.exists(self.ground_truth_file):
            copy_file(self.ground_truth_file, self.validation_folder)
            print(f"Ground truth file copied to {self.validation_folder}")
        else:
            print(f"Ground truth file not found at {self.ground_truth_file}")

        # Load the ground truth labels
        self.ground_truth_df = pd.read_excel(self.ground_truth_file) if os.path.exists(self.ground_truth_file) else None

        # Load the pre-trained Word2Vec model (Google News embeddings)
        self.word2vec_model = KeyedVectors.load_word2vec_format('C:/Users/mdica/Desktop/TESI/GoogleNews-vectors-negative300.bin', binary=True)

    def validate_TFIDF_labels(self):
        # Load the TF-IDF labels
        if os.path.exists(self.tfidf_labels_file):
            self.tfidf_df = pd.read_csv(self.tfidf_labels_file)
            print(f"TF-IDF labels loaded from {self.tfidf_labels_file}")
        else:
            print(f"TF-IDF labels file not found at {self.tfidf_labels_file}")
            self.tfidf_df = None
        
        # Merge "Clusters Top Words" and "Combined Top Words" for TF-IDF
        if self.tfidf_df is not None:
            # Handle NaN or empty Combined Top Words column
            self.tfidf_df["Combined_Top_Words"] = self.tfidf_df["Clusters Top Words (TF-IDF)"] + " " + self.tfidf_df["Combined Top Words (TF-IDF)"].fillna('')
            
            # Handle empty Combined_Top_Words: replace empty strings with Cluster Top Words
            self.tfidf_df["Combined_Top_Words"] = self.tfidf_df.apply(
                lambda row: row["Combined_Top_Words"] if row["Combined_Top_Words"] != '' else row["Clusters Top Words (TF-IDF)"], axis=1
            )
            print("TF-IDF labels merged successfully.")
            # Print the resulting dataframe to check its contents
            # print("Merged TF-IDF labels:")
            # print(self.tfidf_df["Combined_Top_Words"].head())  # Show the first few rows of the dataframe to inspect the merged result

        self.validate_labels(self.tfidf_df, "TFIDF")
        # Generate plots for TF-IDF
        print("Generating plots for TF-IDF...")
        self.plot_cosine_similarity('TF-IDF')
        self.plot_precision_recall_f1_boxplot('TF-IDF')
        self.plot_precision_recall_bar('TF-IDF')

    def validate_LDA_labels(self):
        # Load the LDA labels
        if os.path.exists(self.lda_labels_file):
            self.lda_df = pd.read_csv(self.lda_labels_file)
            print(f"LDA labels loaded from {self.lda_labels_file}")
        else:
            print(f"LDA labels file not found at {self.lda_labels_file}")
            self.lda_df = None
        
        # Merge "Top Topics (LDA)_Clusters" and "Top Topics (LDA)_Combined" for LDA
        if self.lda_df is not None:
            # Handle NaN or empty Combined Top Words column
            self.lda_df["Combined_Top_Words"] = self.lda_df["Top Topics (LDA)_Clusters"] + " " + self.lda_df["Top Topics (LDA)_Combined"].fillna('')
            
            # Handle empty Combined_Top_Words: replace empty strings with Top Topics (LDA)_Clusters
            self.lda_df["Combined_Top_Words"] = self.lda_df.apply(
                lambda row: row["Combined_Top_Words"] if row["Combined_Top_Words"] != '' else row["Top Topics (LDA)_Clusters"], axis=1
            )
            print("LDA labels merged successfully.")
            
        self.validate_labels(self.lda_df, "LDA")
        # Generate plots for LDA
        print("Generating plots for LDA...")
        self.plot_cosine_similarity('LDA')
        self.plot_precision_recall_f1_boxplot('LDA')
        self.plot_precision_recall_bar('LDA')

    def validate_labels(self, labels_df, method):
        # Initialize a list to collect results for each cluster
        results = []

        # Compare generated labels with ground truth using common validation techniques
        if self.ground_truth_df is not None and labels_df is not None:
            # Create a vocabulary based on all labels from both ground truth and generated labels
            all_labels = list(self.ground_truth_df['Label']) + list(labels_df['Combined_Top_Words'])
            vectorizer = CountVectorizer()
            vectorizer.fit(all_labels)  # Fit on both ground truth and generated labels
            print("Vocabulary fitted on both ground truth and generated labels.")

            for cluster_id in range(1, 45):
                # Get the labels from the ground truth and the generated labels for the current cluster
                gt_labels = self.ground_truth_df.loc[self.ground_truth_df['Cluster'] == cluster_id, 'Label'].values[0]
                predicted_labels = labels_df.loc[labels_df['Cluster'] == cluster_id, 'Combined_Top_Words'].values[0]

                if method == "TFIDF":
                    # Join words by space instead of comma
                    predicted_labels = ' '.join(predicted_labels.split(', ')).strip()  # Join words with spaces
                
                # Debugging: Check if labels are NaN or empty
                if pd.isna(gt_labels) or gt_labels == '':
                    print(f"Warning: Ground truth labels for Cluster {cluster_id} are NaN or empty.")
                if pd.isna(predicted_labels) or predicted_labels == '':
                    print(f"Warning: Predicted labels for Cluster {cluster_id} are NaN or empty.")
                
                # Print the labels before vectorization
                print(f"\nGround truth labels for Cluster {cluster_id}: {gt_labels}")
                print(f"Predicted labels for Cluster {cluster_id}: {predicted_labels}")
                
                # Convert the ground truth and predicted labels to vectors using Word2Vec and WordNet
                gt_vector = self.text_to_vector(gt_labels)  # Convert to Word2Vec and WordNet vector
                pred_vector = self.text_to_vector(predicted_labels)  # Convert to Word2Vec and WordNet vector

                # Print the resulting vectors
                # print(f"Vector for ground truth labels (Cluster {cluster_id}): {gt_vector}")
                # print(f"Vector for predicted labels (Cluster {cluster_id}): {pred_vector}")
                
                # Calculate cosine similarity between the ground truth and predicted labels
                cosine_sim = self.calculate_cosine_similarity(gt_vector, pred_vector)  # Pass the vectors directly
                cosine_sim = round(cosine_sim, 3)
                print(f"Cosine Similarity for Cluster {cluster_id}: {cosine_sim}")

                # Implement precision, recall, F1
                precision = self.calculate_precision(gt_labels, predicted_labels)
                recall = self.calculate_recall(gt_labels, predicted_labels)
                f1 = self.calculate_f1(precision, recall)

                precision = round(precision, 3)
                recall = round(recall, 3)
                f1 = round(f1, 3)

                # Collect the results for each cluster
                results.append([cluster_id, cosine_sim, precision, recall, f1])

        # Save results to a CSV file
        results_df = pd.DataFrame(results, columns=["Cluster", "Cosine Similarity", "Precision", "Recall", "F1"])
        
        # Save to different files based on method (TFIDF or LDA)
        if method == "TFIDF":
            output_file = os.path.join(self.validation_folder, "TF-IDF_Validation_Results.csv")
        elif method == "LDA":
            output_file = os.path.join(self.validation_folder, "LDA_Validation_Results.csv")
        
        results_df.to_csv(output_file, index=False)
        print(f"Validation results saved to {output_file}")

    def text_to_vector(self, text):
        # Convert text into a vector by counting word occurrences, using Word2Vec and WordNet for normalization
        words = text.split(" ")
        
        # Normalize words using WordNet (lemmatization) and Word2Vec (embedding)
        normalized_words = [self.normalize_word(word) for word in words]
        
        # Use Word2Vec to get the vector for each word and average them
        word_vectors = []
        for word in normalized_words:
            try:
                word_vectors.append(self.word2vec_model[word])  # Word2Vec embedding
            except KeyError:
                word_vectors.append(np.zeros(self.word2vec_model.vector_size))  # If the word is not in Word2Vec, use a zero vector

        # Average the word vectors to create a single vector representation of the text
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.word2vec_model.vector_size)  # Return zero vector if no words found in Word2Vec

    def normalize_word(self, word):
        # Normalize the word using WordNet (synonym mapping) and Word2Vec (word vector matching)
        lemma = self.get_wordnet_lemma(word)
        return lemma if lemma else word

    def get_wordnet_lemma(self, word):
        # Try to get the WordNet lemma for a word
        synsets = wn.synsets(word)
        if synsets:
            lemma = synsets[0].lemmas()[0].name()  # Get the first lemma
            return lemma
        return word  # If no lemma is found, return the original word

    def calculate_cosine_similarity(self, gt_vector, pred_vector):
        # Compute the cosine similarity
        return cosine_similarity([gt_vector], [pred_vector])[0][0] 

    def calculate_precision(self, gt_labels, predicted_labels):
        # Convert labels to sets
        gt_set = set(gt_labels.split())
        pred_set = set(predicted_labels.split())
        # Calculate precision
        intersection = gt_set.intersection(pred_set)
        return len(intersection) / len(pred_set) if len(pred_set) > 0 else 0

    def calculate_recall(self, gt_labels, predicted_labels):
        # Convert labels to sets
        gt_set = set(gt_labels.split())
        pred_set = set(predicted_labels.split())
        # Calculate recall
        intersection = gt_set.intersection(pred_set)
        return len(intersection) / len(gt_set) if len(gt_set) > 0 else 0

    def calculate_f1(self, precision, recall):
        # Calculate F1 score
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    # Plot Cosine Similarity as Bar plot
    def plot_cosine_similarity(self, method):
        file_path = os.path.join(self.validation_folder, f"{method}_Validation_Results.csv")
        df = pd.read_csv(file_path)

    # Plot the cosine similarity
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Cluster", y="Cosine Similarity", data=df, color='skyblue')
        
        # Adding labels to the bars (Cosine Similarity values on top of the bars)
        for index, value in enumerate(df['Cosine Similarity']):
            plt.text(index, value + 0.02, round(value, 3), ha='center', va='bottom', fontsize=6)
        
        # Setting labels and title
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Cosine Similarity', fontsize=12)
        plt.title(f'Cosine Similarity for Clusters ({method})', fontsize=14)
        
        # Adjust the x-axis to show cluster numbers (1 to 44)
        plt.xticks(ticks=range(44), labels=range(1, 45), rotation=90)
        
        # Display the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.validation_folder, f"{method}_Cosine_Similarity_Plot.png"))
        print(f"Cosine Similarity plot saved to {os.path.join(self.validation_folder, f'{method}_Cosine_Similarity_Plot.png')}")
        plt.close()

    # Plot Precision, Recall, F1 as Boxplot
    def plot_precision_recall_f1_boxplot(self, method):
        file_path = os.path.join(self.validation_folder, f"{method}_Validation_Results.csv")
        df = pd.read_csv(file_path)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[['Precision', 'Recall', 'F1']])
        plt.title(f"{method} - Boxplot of Precision, Recall, F1")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(os.path.join(self.validation_folder, f'{method}_Precision_Recall_F1_Boxplot.png'))  # Save to file
        plt.close()

    def plot_precision_recall_bar(self, method):

        # Load the results from the corresponding CSV file (TFIDF or LDA)
        file_path = os.path.join(self.validation_folder, f"{method}_Validation_Results.csv")
        df = pd.read_csv(file_path)

        # Prepare the data for the side-by-side bar chart
        clusters = df["Cluster"]
        precision = df["Precision"]
        recall = df["Recall"]

        # Create a side-by-side bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set the bar width
        bar_width = 0.35

        # Define the positions of the bars on the X-axis
        index = np.arange(len(clusters))

        # Plot Precision and Recall bars side by side with specified colors
        ax.bar(index, precision, bar_width, label='Precision', color='skyblue')
        ax.bar(index + bar_width, recall, bar_width, label='Recall', color='lightcoral')

        # Set labels and title
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{method} - Precision and Recall for each Cluster', fontsize=14)
        
        # Adjust the x-axis to show cluster numbers (1 to 44)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(clusters, rotation=90)

        # Display a legend
        ax.legend()

        # Adjust layout for better presentation
        plt.tight_layout()
        
        # Save the plot as PNG
        output_file = os.path.join(self.validation_folder, f"{method}_Precision_Recall_Bar.png")
        plt.savefig(output_file)
        plt.close()

        print(f"Side-by-side bar chart saved to {output_file}")