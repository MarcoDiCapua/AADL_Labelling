# AADL Labelling

This project is designed to process AADL (Architecture Analysis and Design Language) files, identify suitable models, and generate detailed analysis reports based on the scanned files. The tool extracts key information from the AADL models, such as components, features, connections, modes, and flow specifications, and outputs a `.txt` report summarizing the findings. Thoose key information are used to identify labels which are going to be associated to clusters that cointains the AADL models. Labels are generated trhough various approaches such as TF-IDF, Chi-Square, LDA and GPT-based approaches.

## Overview

The project is divided into two main components:
1. **AADLManager** - Responsible for scanning and identifying suitable AADL models.
2. **AADLAnalysis** - Responsible for analyzing the suitable models and generating a `.txt` report with relevant details.

### Key Features:
- Scans AADL `.aaxl2` files to determine if they meet the criteria for being suitable.
- Extracts key information from the suitable models, such as components, features, connection instances, mode instances, and flow specifications.
- Generates a `.txt` report containing the analysis results, including:
  - Total counts of each tag found (e.g., components, features, connections, modes, flows).
  - Top 50 and bottom 50 items for each category (components, features, etc.).

## Structure

### `AADLManager` Class
- **Purpose**: Handles scanning of the AADL files and identifies whether each file is suitable.
- **Key Methods**:
  - `scan_aadl_files()`: Scans the directory for AADL files and checks if they are suitable based on the presence of components and features. It returns lists of suitable and not suitable files.
  - `is_suitable_aadl_model()`: Checks if a model contains at least one component and one feature, marking it as suitable.

### `AADLAnalysis` Class
- **Purpose**: Analyzes the suitable AADL files and generates a detailed `.txt` report with the counts of tags and the top/bottom 50 items for each category.
- **Key Methods**:
  - `process_tag()`: Processes and counts occurrences of each tag (e.g., component, feature, connection) in the AADL files.
  - `generate_report()`: Generates a `.txt` report that includes the counts of all relevant tags, as well as the top and bottom 50 items for each category.

### Workflow

1. **Scan the AADL Files**: The `AADLManager` class scans a directory for `.aaxl2` files. It checks each file to determine if it contains at least one component and one feature. Suitable files are copied to a dedicated folder.
2. **Analyze the Suitable Files**: The `AADLAnalysis` class processes the suitable files and extracts relevant information. It counts the occurrences of key tags (such as components, features, etc.) and generates a report.
3. **Generate the Report**: The `.txt` report is generated and saved in the output folder, containing a detailed summary of the analysis.
4. **Generate the Suitable Models and Clusters CSV**: The `AADLManager` also generates a `suitable_models_cluster.csv` file, which maps the suitable AADL models to their respective clusters. This CSV is saved in the output folder.

## Setup and Usage

### Prerequisites
- Python 3.x
- Required Python libraries: `lxml`, `collections`, `os`, `json`, and any other dependencies listed in `requirements.txt`.

### Installation
1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/MarcoDiCapua/AADL_Labelling.git
    cd AADL_Labelling
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration
Before running the script, ensure that your `config.json` is properly configured. The configuration file contains paths to the necessary folders, including:
- `xmi_folder`: Path to the folder containing the AADL `.aaxl2` files.
- `xmi_suitable_models`: Path to the folder where suitable models will be stored.
- `output_folder`: Path where the `.txt` report will be saved.

Example `config.json`:
```json
{
    "xmi_folder": "input/xmi",
    "xmi_suitable_models": "output/AADL/xmi_suitable_models",
    "output_folder": "output",
    "models_feature_extraction": "basic/full",
    "validate_csv": "validate/cluster_test_set_specific.csv"
}
```

### Running the Code
Run the `main.py` script to start the scanning and analysis process:

```bash
python main.py
```

After the script completes, the results will be saved in the specified `output_folder` as a `.txt` file containing the analysis.

### Example Output
The generated `.txt` report will contain information like:

- The total number of files scanned.
- The number of suitable and not suitable files.
- The counts of each tag (e.g., components, features).
- The top 50 and bottom 50 items for each tag.


