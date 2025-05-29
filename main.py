from AADL_manager import AADLManager, AADLAnalysis
from collections import Counter
from utility import get_current_timestamp
from labelling import TextPreprocessing, Labeling
from validation import Validation
import sys

# Redirect standard output to a log file
# log_file_path = "output/log.txt"
# log_file = open(log_file_path, 'w')
# sys.stdout = log_file  


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
        
        # Generate plots
        # print("Generating plots...")
        # aadl_analysis.plot_total_models()
        # aadl_analysis.plot_total_vs_suitable_models_pie()
        # aadl_analysis.plot_cluster_distribution_stacked()
        # aadl_analysis.plot_cluster_distribution()
        # aadl_analysis.plot_suitable_cluster_distribution()
        # aadl_analysis.plot_top_instances(component_counter, feature_counter, connection_instance_counter)
        # aadl_analysis.plot_total_counts(component_counter, feature_counter, connection_instance_counter, mode_instance_counter, flow_specification_counter)
        print("Plots generated successfully.")
        
        # Preprocess the data
        print("Preprocessing data...")
        # text_preprocessor = TextPreprocessing(config_path="config.json")
        # text_preprocessor.preprocess()
        print("Data preprocessing completed.")
        
        # Apply TF-IDF and generate labels for clusters using the Labeling class
        # print("Generating labels using TF-IDF...")
        # labeling = Labeling(config_path="config.json")
        # labeling.apply_tfidf()
        # print("Generating labels using LDA...")
        # labeling.apply_lda()
        # print("Labels generation completed.")

        # Validation process
        print("Running validation...")
        validation = Validation(config_path="config.json")
        validation.validate_TFIDF_labels()
        validation.validate_LDA_labels()

        
        print("Finished elaboration time: ", get_current_timestamp())

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Restore stdout to the terminal after the program ends
# sys.stdout = sys.__stdout__
# log_file.close()

# Entry point
if __name__ == "__main__":
    main()
