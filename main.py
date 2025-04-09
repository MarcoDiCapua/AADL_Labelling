from AADL_manager import AADLManager, AADLAnalysis
from collections import Counter
import os
from lxml import etree
from utility import load_config, create_directory, delete_file, list_files_in_directory, copy_file, get_current_timestamp

def main():
    try:
        # Initialize the AADLManager with the path to the configuration file
        aadl_manager = AADLManager(config_path="config.json")
        
        # Scan AADL files and get the list of suitable files
        suitable_files, aadl_files, not_suitable_files = aadl_manager.scan_aadl_files()
        
        # Initialize the AADLAnalysis class to generate the report
        aadl_analysis = AADLAnalysis(suitable_files, aadl_files, config_path="config.json")

        # Initialize counters for the tags
        component_counter = Counter()
        feature_counter = Counter()
        connection_instance_counter = Counter()
        mode_instance_counter = Counter()
        flow_specification_counter = Counter()

        # Process each of the suitable AADL files
        for aadl_file in suitable_files:
            aadl_file_path = os.path.join(aadl_manager.xmi_folder, aadl_file)
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

        # Generate the .txt report
        aadl_analysis.generate_report(component_counter, feature_counter, connection_instance_counter,
                                      mode_instance_counter, flow_specification_counter)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Entry point
if __name__ == "__main__":
    main()
