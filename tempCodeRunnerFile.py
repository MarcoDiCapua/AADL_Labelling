def plot_precision_recall_f1_boxplot_combined(self):
        tfidf_file = os.path.join(self.validation_folder, "TF-IDF_Validation_Results.csv")
        lda_file = os.path.join(self.validation_folder, "LDA_Validation_Results.csv")
        tfidf_df = pd.read_csv(tfidf_file)
        lda_df = pd.read_csv(lda_file)
        
        tfidf_df['Tecnica'] = 'TF-IDF'
        lda_df['Tecnica'] = 'LDA'
        combined_df = pd.concat([tfidf_df[['Precision', 'Recall', 'F1', 'Tecnica']], lda_df[['Precision', 'Recall', 'F1', 'Tecnica']]])
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Tecnica", y="value", hue="variable", 
                    data=pd.melt(combined_df, id_vars=["Tecnica"], value_vars=["Precision", "Recall", "F1"]),
                    palette="Set2")
        plt.title("Punteggi di Precision, Recall, e F1 - TF-IDF vs LDA", fontsize=16)
        plt.xlabel("Tecnica", fontsize=12)
        plt.ylabel("Punteggio", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.validation_folder, 'Precision_Recall_F1_Boxplot_Combined.png'))
        print(f"Boxplot for Precision, Recall, and F1 (TF-IDF vs LDA) saved to {os.path.join(self.validation_folder, 'Precision_Recall_F1_Boxplot_Combined.png')}")
        plt.close()
