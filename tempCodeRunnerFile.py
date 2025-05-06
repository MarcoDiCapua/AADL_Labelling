    def generate_stacked_bar_chart(self):
        # Read the CSV files containing the topics for each cluster
        clusters_df = pd.read_csv(os.path.join(self.LDA_folder, 'Clusters_Top_LDA.csv'))
        combined_df = pd.read_csv(os.path.join(self.LDA_folder, 'Combined_Top_LDA.csv'))

        # Merge the two dataframes on the 'Cluster' column
        # Ensure every cluster from clusters_df is included, even if it's missing in combined_df
        merged_df = pd.merge(clusters_df[['Cluster', 'Top Topics (LDA)']], 
                            combined_df[['Cluster', 'Top Topics (LDA)']], 
                            on='Cluster', suffixes=('_Clusters', '_Combined'), how='left')

        # Count the number of topics for each cluster in both 'Clusters' and 'Combined'
        merged_df['Topics_Clusters'] = merged_df['Top Topics (LDA)_Clusters'].apply(lambda x: len(x.split(',')))
        
        # If a cluster doesn't have topics in Combined_Top_LDA, set Topics_Combined to 0
        merged_df['Topics_Combined'] = merged_df['Top Topics (LDA)_Combined'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)

        # Create a stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.35  # Set the width of the bars

        # Position of the bars
        x_pos = merged_df['Cluster'] 

        # Create the bars for 'Topics_Clusters' and 'Topics_Combined'
        ax.bar(x_pos - bar_width/2, merged_df['Topics_Clusters'], label='Clusters Top Topics', color='skyblue', width=bar_width)
        ax.bar(x_pos + bar_width/2, merged_df['Topics_Combined'], label='Combined Top Topics', color='lightcoral', width=bar_width)

        # Annotate each bar with the number of topics
        for i in range(len(merged_df)):
            ax.text(x_pos[i] - bar_width/2, merged_df['Topics_Clusters'][i] + 0.1, 
                    str(merged_df['Topics_Clusters'][i]), ha='center', va='bottom', fontsize=10)
            ax.text(x_pos[i] + bar_width/2, merged_df['Topics_Combined'][i] + 0.1, 
                    str(merged_df['Topics_Combined'][i]), ha='center', va='bottom', fontsize=10)

        # Add labels and title
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Topics')
        ax.set_title('Topics Distribution by Cluster')
        
        # Set the X-axis ticks to range from 1 to 44 (since we have 44 clusters)
        ax.set_xticks(range(1, 45))  # Cluster numbers from 1 to 44
        ax.set_xticklabels(range(1, 45))

        ax.legend()

        # Save the plot as a PNG file in the same folder as other outputs
        output_file = os.path.join(self.LDA_folder, "topics_distribution_by_cluster.png")
        plt.tight_layout()
        plt.savefig(output_file)  # Save the plot
        print(f"Stacked bar chart saved to {output_file}")

        # Show the plot
        plt.show()