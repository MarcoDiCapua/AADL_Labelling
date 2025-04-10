from AADL_manager import AADLManager, AADLAnalysis
from collections import Counter
from utility import get_current_timestamp

def main():
    try:
        starting_time = get_current_timestamp()
        print("Starting elaboration time: ", starting_time)

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

        # Process the AADL files and update counters
        aadl_manager.process_aadl_files(suitable_files, aadl_analysis, component_counter, 
                                        feature_counter, connection_instance_counter, 
                                        mode_instance_counter, flow_specification_counter)

        # Generate the .txt report
        aadl_analysis.generate_report(component_counter, feature_counter, connection_instance_counter,
                                      mode_instance_counter, flow_specification_counter)
        
        print("Finished elaboration time: ", get_current_timestamp())
        print("Total time: ", get_current_timestamp() - starting_time)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Entry point
if __name__ == "__main__":
    main()
