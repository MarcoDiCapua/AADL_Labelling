import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lxml import etree
from utility import load_config, create_directory, delete_file, list_files_in_directory, copy_file

class AADLManager:
    def __init__(self, config_path="config.json"):
        self.config = load_config(config_path)
        self.xmi_folder = self.config.get("xmi_folder", "")
        self.suitable_models_folder = self.config.get("xmi_suitable_models", "")
        self.output_folder = self.config.get("output_folder", "")
        self.clusters_file = self.config.get("clusters", "")

    def scan_aadl_files(self):
        # Ensure suitable models folder exists and delete old files
        create_directory(self.suitable_models_folder) 
        for existing_file in os.listdir(self.suitable_models_folder):
            file_path = os.path.join(self.suitable_models_folder, existing_file)
            delete_file(file_path)
            
        csv_file_path = os.path.join(self.output_folder, "AADL/suitable_models_data.csv")
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
            print(f"Existing file {csv_file_path} deleted.")

        # List all .AAXL2 files in the xmi_folder
        aadl_files = list_files_in_directory(self.xmi_folder)
        if not aadl_files:
            raise FileNotFoundError(f"No files found in {self.xmi_folder}")
        # Filter for .AAXL2 files
        aadl_files = [f for f in aadl_files if f.endswith('.aaxl2')]
        if not aadl_files:
            raise FileNotFoundError(f"No .AAXL2 files found in {self.xmi_folder}")
        
        print(f"Found {len(aadl_files)} AADL files to scan.")

        # Initialize lists to store suitable and not suitable files
        suitable_files = []
        not_suitable_files = []  

        print("Scanning for suitable AADL models...")
        for aadl_file in aadl_files:
            aadl_file_path = os.path.join(self.xmi_folder, aadl_file)

            # Parse the AADL file
            print(f"Scanning file: {aadl_file_path}")
            try:
                tree = etree.parse(aadl_file_path)
                root = tree.getroot()
            except Exception as e:
                print(f"Error parsing file {aadl_file}: {e}")
                continue

            # Check if the file contains at least one component and one feature
            if self.is_suitable_aadl_model(root):
                suitable_files.append(aadl_file)
                # Copy the file to the suitable models folder
                destination_file = os.path.join(self.suitable_models_folder, aadl_file)
                copy_file(aadl_file_path, destination_file)
                print(f"File {aadl_file} is suitable and copied to {self.suitable_models_folder}")    
            
                # Generate the CSV data for this model
                self.generate_model_csv(aadl_file, root)
            else:
                not_suitable_files.append(aadl_file)
                print(f"File {aadl_file} is not suitable.")
        
        # Generate suitable_models_cluster.csv and add cluster information
        self.generate_suitable_models_cluster_csv(suitable_files)
        self.add_cluster_to_suitable_models_data()

        print(f"Total suitable files: {len(suitable_files)}")
        print(f"Total not suitable files: {len(not_suitable_files)}")
        print("Scanning complete.")
        return suitable_files, aadl_files, not_suitable_files

    def is_suitable_aadl_model(self, root):
        # Remove namespaces from the XML tree by renaming tags.
        for elem in root.getiterator():
            elem.tag = self.remove_namespace_from_tag(elem.tag)
        
        # Find components and features
        components = root.findall('.//componentInstance')
        features = root.findall('.//featureInstance')
        
        if components and features:
            return True
        return False

    def remove_namespace_from_tag(self, tag):
        if '}' in tag:
            return tag.split('}', 1)[1]  # Split and return the part after '}'
        return tag
    
    def generate_model_csv(self, model_name, root):
        model_name_without_extension = model_name.split('.')[0]
        components = root.findall('.//componentInstance')
        features = root.findall('.//featureInstance')
        connection_instances = root.findall('.//connectionInstance')

        # CSV file path for suitable models data
        csv_file_path = os.path.join(self.output_folder, "AADL/suitable_models_data.csv")

        # Check if the CSV file already exists to append data, otherwise create it
        file_exists = os.path.exists(csv_file_path)

        with open(csv_file_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            if not file_exists:
                header = ['Model', 'Cluster', 'Component', 'Feature', 'ConnectionInstance']
                csv_writer.writerow(header)

            # Extract and concatenate the information for the components, features, and connection instances
            component_names = [component.get('name', 'Unnamed component') for component in components]
            feature_names = [feature.get('name', 'Unnamed feature') for feature in features]
            connection_names = [connection.get('name', 'Unnamed connectionInstance') for connection in connection_instances]

            # Concatenate all information as a string, separated by commas
            components_str = ', '.join(component_names)
            features_str = ', '.join(feature_names)
            connection_str = ', '.join(connection_names)

            # Write the data for the model into the CSV (one row per model)
            csv_writer.writerow([model_name_without_extension, '', components_str, features_str, connection_str])

        print(f"Data for model {model_name} appended to {csv_file_path}")

    def generate_suitable_models_cluster_csv(self, suitable_files):        
        # Read the clusters from clusters.csv
        cluster_mapping = {}
        with open(self.clusters_file, mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                cluster_mapping[row[0]] = row[1]  # Map model name to cluster number

        # Debugging: Print out the cluster mapping to verify its contents
        print("Cluster mapping:")
        for model, cluster in cluster_mapping.items():
            print(f"Model: {model}, Cluster: {cluster}")
        
        suitable_models_with_clusters = []

        # Debugging: Print out the list of suitable files
        print("\nSuitable files being processed:")
        for model in suitable_files:
            # Strip the .aaxl2 extension from the model name before comparing
            model_name_without_extension = model.split('.')[0]

            print(f"Processing model: {model_name_without_extension}")

            # Look for the cluster for the current model
            cluster = cluster_mapping.get(model_name_without_extension)
            
            # Debugging: Print the result of the cluster lookup
            if cluster:
                print(f"Model {model_name_without_extension} found in cluster {cluster}")
                suitable_models_with_clusters.append([model, cluster])
            else:
                print(f"Model {model_name_without_extension} not found in cluster mapping.")
        
        if suitable_models_with_clusters:
            # Output path for suitable_models_cluster.csv
            suitable_models_cluster_file = os.path.join(self.output_folder, 'AADL/suitable_models_cluster.csv')

            with open(suitable_models_cluster_file, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(['Model', 'Cluster'])  # Write header
                csv_writer.writerows(suitable_models_with_clusters)

            print(f"Suitable models with clusters have been written to {suitable_models_cluster_file}")
        else:
            print("No suitable models with clusters to write.")

    def add_cluster_to_suitable_models_data(self):
        """Adds the Cluster column to the suitable_models_data.csv based on clusters.csv"""
        suitable_models_data_file = os.path.join(self.output_folder, "AADL/suitable_models_data.csv")
        cluster_mapping = {}

        # Read the cluster mappings from clusters.csv
        with open(self.clusters_file, mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                cluster_mapping[row[0]] = row[1]  # Model to Cluster mapping

        # Read the suitable models data
        suitable_models_df = pd.read_csv(suitable_models_data_file)

        # Add the Cluster column based on the model name
        suitable_models_df['Cluster'] = suitable_models_df['Model'].apply(lambda x: cluster_mapping.get(x, 'Unknown'))

        # Save the updated suitable_models_data.csv
        suitable_models_df.to_csv(suitable_models_data_file, index=False)
        print(f"Updated suitable_models_data.csv with Cluster column.")

    def process_aadl_files(self, suitable_files, aadl_analysis, component_counter, 
                            feature_counter, connection_instance_counter, 
                            mode_instance_counter, flow_specification_counter):
        # Process each of the suitable AADL files
        for aadl_file in suitable_files:
            aadl_file_path = os.path.join(self.xmi_folder, aadl_file)
            try:
                tree = etree.parse(aadl_file_path)
                root = tree.getroot()
            except Exception as e:
                print(f"Error parsing file {aadl_file}: {e}")
                continue

            # Extract relevant tags and update counters
            components = root.findall('.//componentInstance')
            features = root.findall('.//featureInstance')
            connection_instances = root.findall('.//connectionInstance')
            mode_instances = root.findall('.//modeInstance')
            flow_specifications = root.findall('.//flowSpecification')

            # Process each tag using AADLAnalysis
            aadl_analysis.process_tag(components, component_counter, 'name', 'Unnamed component')
            aadl_analysis.process_tag(features, feature_counter, 'name', 'Unnamed feature')
            aadl_analysis.process_tag(connection_instances, connection_instance_counter, 'name', 'Unnamed connectionInstance')
            aadl_analysis.process_tag(mode_instances, mode_instance_counter, 'name', 'Unnamed modeInstance')
            aadl_analysis.process_tag(flow_specifications, flow_specification_counter, 'name', 'Unnamed flowSpecification')

class AADLAnalysis:
        
    def __init__(self, suitable_files, aadl_files, config_path="config.json"):
        self.suitable_files = suitable_files
        self.aadl_files = aadl_files
        self.config = load_config(config_path)
        self.suitable_models_folder = self.config.get("xmi_suitable_models", "")
        self.output_folder = os.path.join(self.config.get("output_folder", ""), "AADL")

    # Helper method to process tags and update counters.
    def process_tag(self, elements, counter, attribute, default_name):
        for element in elements:
            tag_value = element.get(attribute, default_name)
            counter[tag_value] += 1

    # Analyze AADL files and generate report.
    def generate_report(self, component_counter, feature_counter, connection_instance_counter,
                        mode_instance_counter, flow_specification_counter):
        output_file_path = os.path.join(self.output_folder, "aadl_scan_results.txt")

        with open(output_file_path, "w") as f:
            f.write(f"Number of AADL files scanned: {len(self.aadl_files)}\n")
            f.write(f"Number of suitable AADL files: {len(self.suitable_files)}\n")
            f.write(f"Number of not suitable AADL files: {len(self.aadl_files) - len(self.suitable_files)}\n")

            f.write(f"\nTotal components found: {len(component_counter)}\n")
            f.write(f"Total features found: {len(feature_counter)}\n")
            f.write(f"Total connection instances found: {len(connection_instance_counter)}\n")
            f.write(f"Total mode instances found: {len(mode_instance_counter)}\n")
            f.write(f"Total flow specifications found: {len(flow_specification_counter)}\n")

            f.write("\nTop 50 Components:\n")
            for component, count in component_counter.most_common(50):
                f.write(f"{component}: {count}\n")

            f.write("\nBottom 50 Components:\n")
            for component, count in component_counter.most_common()[-50:]:
                f.write(f"{component}: {count}\n")

            f.write("\nTop 50 Features:\n")
            for feature, count in feature_counter.most_common(50):
                f.write(f"{feature}: {count}\n")

            f.write("\nBottom 50 Features:\n")
            for feature, count in feature_counter.most_common()[-50:]:
                f.write(f"{feature}: {count}\n")

            f.write("\nTop 50 Connection Instances:\n")
            for connection_instance, count in connection_instance_counter.most_common(50):
                f.write(f"{connection_instance}: {count}\n")

            f.write("\nBottom 50 Connection Instances:\n")
            for connection_instance, count in connection_instance_counter.most_common()[-50:]:
                f.write(f"{connection_instance}: {count}\n")

            f.write("\nTop 50 Mode Instances:\n")
            for mode_instance, count in mode_instance_counter.most_common(50):
                f.write(f"{mode_instance}: {count}\n")

            f.write("\nBottom 50 Mode Instances:\n")
            for mode_instance, count in mode_instance_counter.most_common()[-50:]:
                f.write(f"{mode_instance}: {count}\n")

            f.write("\nTop 50 Flow Specifications:\n")
            for flow_specification, count in flow_specification_counter.most_common(50):
                f.write(f"{flow_specification}: {count}\n")

            f.write("\nBottom 50 Flow Specifications:\n")
            for flow_specification, count in flow_specification_counter.most_common()[-50:]:
                f.write(f"{flow_specification}: {count}\n")

        print(f"Results written to {output_file_path}")
        print("Analysis complete.")

    def plot_total_models(self):
        total_aadl_models = len(self.aadl_files)
        total_suitable_models = len(self.suitable_files)
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(['Total AADL Models', 'Suitable AADL Models'], [total_aadl_models, total_suitable_models], color='skyblue')
        plt.title('Total vs Suitable AADL Models')
        plt.ylabel('Count')
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())), 
                     ha='center', va='bottom', fontsize=12)
        plt.savefig(os.path.join(self.output_folder, 'total_vs_suitable_models.png'))
        plt.close()

    def plot_total_counts(self, component_counter, feature_counter, connection_instance_counter, mode_instance_counter, flow_specification_counter):
        counts = {
            'Components': len(component_counter),
            'Features': len(feature_counter),
            'Connections': len(connection_instance_counter),
            'Modes': len(mode_instance_counter),
            'Flow Specifications': len(flow_specification_counter)
        }

        plt.figure(figsize=(8, 6))
        bars = plt.bar(counts.keys(), counts.values(), color='skyblue')
        plt.title('Total Count of Components, Features, Connections, Modes and Flow Specifications')
        plt.ylabel('Count')
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())), 
                     ha='center', va='bottom', fontsize=12)
        plt.savefig(os.path.join(self.output_folder, 'total_counts.png'))
        plt.close()

    def plot_cluster_distribution(self):
        clusters_df = pd.read_csv(self.config.get("clusters", ""))
        cluster_counts = clusters_df['Cluster'].value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        bars = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, color='skyblue')
        plt.title('Cluster Distribution of AADL Models')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Models')
        for bar in bars.patches:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())), 
                     ha='center', va='bottom', fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'cluster_distribution.png'))
        plt.close()

    def plot_suitable_cluster_distribution(self):
        suitable_models_cluster_df = pd.read_csv(os.path.join(self.output_folder, 'suitable_models_cluster.csv'))

        # Count suitable models per cluster
        suitable_cluster_counts = suitable_models_cluster_df['Cluster'].value_counts().sort_index()
        all_clusters = set(range(1, 45))
        # Add missing clusters with 0 suitable models
        for cluster in all_clusters:
            if cluster not in suitable_cluster_counts:
                suitable_cluster_counts[cluster] = 0
        suitable_cluster_counts = suitable_cluster_counts.sort_index()
        plt.figure(figsize=(10, 6))
        bars = sns.barplot(x=suitable_cluster_counts.index, y=suitable_cluster_counts.values, color='skyblue')
        # Highlight clusters with 0 suitable models
        for cluster, count in suitable_cluster_counts.items():
            if count == 0:
                plt.axvline(x=cluster - 1, color='red', linestyle='--')
        for bar in bars.patches:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())), 
                    ha='center', va='bottom', fontsize=12)
        plt.xticks(list(suitable_cluster_counts.index), rotation=90)
        plt.title('Cluster Distribution of Suitable AADL Models')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Suitable Models')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'suitable_cluster_distribution.png'))
        plt.close()

    def plot_top_instances(self, component_counter, feature_counter, connection_instance_counter):
        top_25_components = component_counter.most_common(25)
        top_25_features = feature_counter.most_common(25)
        top_25_connections = connection_instance_counter.most_common(25)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=[comp[1] for comp in top_25_components], y=[comp[0] for comp in top_25_components], color='skyblue')
        plt.title('Top 25 Components')
        plt.xlabel('Count')
        plt.ylabel('Component')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'top_25_components.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=[comp[1] for comp in top_25_features], y=[comp[0] for comp in top_25_features], color='skyblue')
        plt.title('Top 25 Features')
        plt.xlabel('Count')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'top_25_features.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=[comp[1] for comp in top_25_connections], y=[comp[0] for comp in top_25_connections], color='skyblue')
        plt.title('Top 25 Connections')
        plt.xlabel('Count')
        plt.ylabel('Connection')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'top_25_connections.png'))
        plt.close()


    def plot_total_vs_suitable_models_pie(self):
        total_aadl_models = len(self.aadl_files)
        total_suitable_models = len(self.suitable_files)
        labels = ['Suitable\nModels', 'Not suitable\nMoldels']
        sizes = [total_suitable_models, total_aadl_models - total_suitable_models]
        colors = ['skyblue', 'lightcoral']

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title(f'Total vs Suitable AADL Models \nTotal Models: {total_aadl_models} \nSuitable Models: {total_suitable_models} ')
        plt.axis('equal')
        plt.savefig(os.path.join(self.output_folder, 'total_vs_suitable_models_pie.png'))
        plt.close()


    def plot_cluster_distribution_stacked(self):
        clusters_df = pd.read_csv(self.config.get("clusters", ""))
        suitable_models_cluster_df = pd.read_csv(os.path.join(self.output_folder, 'suitable_models_cluster.csv'))
        
        # Get counts for total models in each cluster
        total_cluster_counts = clusters_df['Cluster'].value_counts().sort_index()
        # Get counts for suitable models in each cluster
        suitable_cluster_counts = suitable_models_cluster_df['Cluster'].value_counts().sort_index()
        all_clusters = set(range(1, 45))
        # Add missing clusters with 0 values
        for cluster in all_clusters:
            if cluster not in total_cluster_counts:
                total_cluster_counts[cluster] = 0
            if cluster not in suitable_cluster_counts:
                suitable_cluster_counts[cluster] = 0
        total_cluster_counts = total_cluster_counts.sort_index()
        suitable_cluster_counts = suitable_cluster_counts.sort_index()
        non_suitable_cluster_counts = total_cluster_counts - suitable_cluster_counts

        plt.figure(figsize=(10, 6))
        plt.bar(total_cluster_counts.index, non_suitable_cluster_counts, label='Non-Suitable', color='lightcoral')
        plt.bar(total_cluster_counts.index, suitable_cluster_counts, bottom=non_suitable_cluster_counts, label='Suitable', color='skyblue')
        plt.title('Cluster Distribution of AADL Models (Stacked)')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Models')
        plt.legend()
        plt.xticks(range(1, 45), rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'cluster_distribution_stacked.png'))
        plt.close()
