import os
import csv
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
            else:
                not_suitable_files.append(aadl_file)
                print(f"File {aadl_file} is not suitable.")
        
        # Generate suitable_models_cluster.csv
        self.generate_suitable_models_cluster_csv(suitable_files)

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
            suitable_models_cluster_file = os.path.join(self.output_folder, 'suitable_models_cluster.csv')

            with open(suitable_models_cluster_file, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(['Model', 'Cluster'])  # Write header
                csv_writer.writerows(suitable_models_with_clusters)

            print(f"Suitable models with clusters have been written to {suitable_models_cluster_file}")
        else:
            print("No suitable models with clusters to write.")


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
