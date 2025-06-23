# Computational Drug Discovery: Bioactivity Prediction for a Coronavirus Target

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20RDKit%20%7C%20Seaborn-orange.svg)
![Environment](https://img.shields.io/badge/Environment-Google%20Colab-F9AB00.svg?logo=googlecolab)

This project is a walkthrough of a computational drug discovery workflow, from data acquisition to exploratory data analysis (EDA). The goal is to identify and analyze chemical compounds with potential bioactivity against a specific protein target of the Coronavirus, sourced from the ChEMBL database.

## Project Overview

The project follows these key steps:
1.  **Data Acquisition:** Programmatically download bioactivity data (IC50 values) for a selected Coronavirus protein target from the ChEMBL database.
2.  **Data Curation:** Clean and preprocess the data. This includes filtering for valid entries and classifying compounds as **active**, **inactive**, or **intermediate** based on their IC50 values.
3.  **Feature Engineering:** Calculate Lipinski's Rule of Five molecular descriptors (e.g., Molecular Weight, LogP) using the RDKit library. These descriptors help in evaluating the "drug-likeness" of a compound.
4.  **Exploratory Data Analysis (EDA):** Convert IC50 values to their logarithmic equivalent, pIC50, for more uniform data distribution. Visualize the relationship between chemical properties and bioactivity to gain insights into the dataset.

## Dataset

The bioactivity data was retrieved from the [ChEMBL Database](https://www.ebi.ac.uk/chembl/), a large, open-access database of bioactive drug-like small molecules.

-   **Target:** A 'SINGLE PROTEIN' associated with 'coronavirus'.
-   **Bioactivity Measure:** IC50 (Half-maximal inhibitory concentration), which indicates the concentration of a drug required to inhibit a biological process by 50%. A lower IC50 value signifies higher potency.

The raw data is processed and saved in two stages:
-   `bioactivity_data.csv`: The initial, raw data downloaded from ChEMBL.
-   `bioactivity_preprocessed_data.csv`: The curated dataset with bioactivity classes assigned.

## Key Visualizations

A central part of the analysis is understanding the relationship between **IC50** and its logarithmic counterpart, **pIC50** (`-log10(IC50 in M)`). The pIC50 scale is often preferred in QSAR modeling as it is more linearly correlated with the free energy of binding.

The plot below visualizes this relationship, with data points colored by their assigned bioactivity class.

### Relationship between IC50 and pIC50



This plot clearly demonstrates:
-   The **inverse relationship** between IC50 and pIC50.
-   The use of a **log scale** on the x-axis to effectively visualize data spanning several orders of magnitude.
-   The distinct clusters of **active**, **intermediate**, and **inactive** compounds, as defined by the IC50 thresholds:
    -   **Active:** IC50 <= 1,000 nM (pIC50 >= 6)
    -   **Inactive:** IC50 >= 10,000 nM (pIC50 <= 5)
    -   **Intermediate:** 1,000 nM < IC50 < 10,000 nM (5 < pIC50 < 6)

## Getting Started

This project is designed to be run in a Google Colab environment, which simplifies dependency management.

### Prerequisites
- A Google Account to use Google Colab and Google Drive.

### How to Run
1.  **Clone the Repository (Optional):**
2.  **Open in Google Colab:**
    -   Download a copy of the ipynb file.
    -   Go to [Google Colab](https://colab.research.google.com/).
    -   Click `File` -> `Upload notebook...` and select the `bioinformatics_1.ipynb` file from the cloned repository.
3.  **Run the Notebook:**
    -   Execute the cells in order from top to bottom.
    -   You will be prompted to mount your Google Drive. Authorize the connection to allow the notebook to save files (`.csv`) to your Drive.
    -   The notebook will automatically handle the installation of all necessary libraries, including `chembl_webresource_client` and `rdkit` (via Miniconda).
*(Note: The `/data` directory is created and populated in your Google Drive during the notebook's execution.)*

## Future Work
The exploratory analysis performed here lays the groundwork for the next logical step: building a predictive machine learning model.
-   **Develop a QSAR Model:** Build a Quantitative Structure-Activity Relationship (QSAR) model to predict the pIC50 value (and thus, the bioactivity) of a molecule based on its chemical structure.(**currently being explored**)
-   **Feature Selection:** Use more advanced molecular descriptors and apply feature selection techniques to improve model performance.
-   **Model Evaluation:** Train and evaluate various regression models (e.g., Random Forest, Gradient Boosting) and compare their predictive power.

## Acknowledgments
This project was inspired by and follows the methodology outlined in the "Computational Drug Discovery" tutorial series by the [Data Professor](http://youtube.com/dataprofessor) (Dr. Chanin Nantasenamat).
