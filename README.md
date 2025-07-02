# Computational Drug Discovery: Bioactivity Prediction for a Coronavirus Target

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20RDKit%20%7C%20Scikit--learn%20%7C%20Seaborn-orange.svg)
![Environment](https://img.shields.io/badge/Environment-Google%20Colab-F9AB00.svg?logo=googlecolab)

This project demonstrates a comprehensive computational drug discovery workflow, from data acquisition and preprocessing to advanced Quantitative Structure-Activity Relationship (QSAR) model development, hyperparameter optimization, and interpretability analysis. The primary goal is to predict the bioactivity (pIC50) of chemical compounds against a specific coronavirus protein target: **SARS coronavirus main protease (CHEMBL3927)**.

## Project Overview

This project implements a complete QSAR modeling pipeline, encompassing the following key stages:

1.  **Data Acquisition:** Programmatically download bioactivity data (IC50 values) for the selected Coronavirus protein target from the ChEMBL database.
2.  **Data Preprocessing & Curation:** Clean and preprocess the raw bioactivity data. This involves handling missing values, standardizing activity units, classifying compounds (active/inactive/intermediate), and transforming IC50 values to pIC50 for robust modeling.
3.  **Molecular Feature Engineering:** Generate diverse numerical representations of chemical structures using the RDKit library, including:
    *   **Lipinski's Rule of Five Descriptors:** Fundamental drug-likeness properties (MW, LogP, NumHDonors, NumHAcceptors).
    *   **Comprehensive RDKit Descriptors:** A wide range of physicochemical and structural descriptors.
    *   **Morgan Fingerprints:** 2048-bit circular fingerprints for capturing molecular substructures.
4.  **Exploratory Data Analysis (EDA):** Visualize key relationships within the dataset, including the distribution of bioactivity (pIC50), correlations between molecular properties, and statistical differences between active and inactive compounds.
5.  **QSAR Model Development (RandomForestRegressor):** Build a predictive model using the RandomForestRegressor algorithm to establish a baseline performance.
6.  **Hyperparameter Optimization:** Systematically improve model performance through advanced tuning techniques:
    *   **GridSearchCV:** Exhaustive search over a predefined hyperparameter grid.
    *   **RandomizedSearchCV:** Efficient stochastic sampling from hyperparameter distributions.
7.  **Model Evaluation & Interpretation:** Rigorously assess model performance using R-squared (R²) and Root Mean Squared Error (RMSE) on unseen data, compare different models, and analyze feature importances to gain chemical insights into the structure-activity relationships.

## Dataset

The bioactivity data is sourced from the [ChEMBL Database](https://www.ebi.ac.uk/chembl/), a freely accessible chemical database of compounds with drug-like properties and their biological activities.

*   **Target Organism:** Severe acute respiratory syndrome-related coronavirus
*   **Target Name:** SARS coronavirus 3C-like proteinase (CHEMBL3927) - a key enzyme for viral replication.
*   **Bioactivity Measure:** IC50 (Half-maximal inhibitory concentration), which quantifies the concentration of a compound required to inhibit 50% of a specific biological process. Lower IC50 values indicate higher potency.
*   **Activity Transformation:** IC50 values are converted to pIC50 values (`-log10(IC50 in M)`). This transformation provides a more linear scale for regression analysis.

The dataset is processed and stored in the following files within the repository:

*   `bioactivity_data.csv`: The initial, raw bioactivity data downloaded directly from ChEMBL.
*   `bioactivity_preprocessed_data.csv`: The curated dataset after handling missing values, calculating pIC50, and assigning bioactivity classes.

## Key Visualizations & EDA Insights

Exploratory Data Analysis is crucial for understanding the dataset's characteristics and guiding model development.

### Relationship between IC50 and pIC50

The plot below visualizes the inverse logarithmic relationship between IC50 and pIC50, with data points colored by their assigned bioactivity class.

![IC50 vs pIC50 Plot](https://github.com/Rishiatweb/Bioinformatics_CDD/blob/main/plot_ic50_vs_pic50.png)
This visualization effectively demonstrates:
*   The **logarithmic transformation** effectively compresses a wide range of IC50 values into a more manageable pIC50 scale.
*   The distinct clustering of **active** (pIC50 >= 6), **intermediate** (5 < pIC50 < 6), and **inactive** (pIC50 <= 5) compounds, as defined by the IC50 thresholds (`1,000 nM` and `10,000 nM`).

### Statistical Insights from Lipinski's Descriptors (Mann-Whitney U Test)

The Mann-Whitney U test was applied to compare the distributions of Lipinski's descriptors between 'active' and 'inactive' compounds. This non-parametric test helps determine if there are significant differences between these groups.

*   **pIC50:** **p-value = 4.76e-20** (< 0.05). **Interpretation:** A highly significant difference in pIC50 values exists between active and inactive compounds, validating the chosen activity thresholds.
*   **Molecular Weight (MW):** **p-value = 0.009** (< 0.05). **Interpretation:** Active and inactive compounds exhibit statistically different molecular weight distributions.
*   **LogP:** **p-value = 0.063** (> 0.05). **Interpretation:** No statistically significant difference in LogP (lipophilicity) distribution was observed between active and inactive compounds, suggesting it might not be a primary standalone differentiator.
*   **NumHDonors:** **p-value = 0.001** (< 0.05). **Interpretation:** Significant differences in the number of hydrogen bond donors were found between the groups.
*   **NumHAcceptors:** **p-value = 0.012** (< 0.05). **Interpretation:** Significant differences in the number of hydrogen bond acceptors were found between the groups.

These statistical findings suggest that MW, NumHDonors, and NumHAcceptors are important molecular properties distinguishing active from inactive compounds in this dataset.

## QSAR Model Development & Evaluation

A RandomForestRegressor was chosen for its robustness and ability to capture non-linear relationships in complex biological data.

### Baseline Model Performance

A RandomForestRegressor with default hyperparameters was trained as a benchmark:

*   **R-squared (R²):** 0.517
*   **Root Mean Squared Error (RMSE):** 0.541

### Hyperparameter Optimization

To improve upon the baseline, hyperparameter tuning was conducted using both exhaustive and stochastic search strategies.

#### GridSearchCV Results

*   **Best Hyperparameters:** `{'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}`
*   **Best Cross-Validation R-squared:** 0.6065
*   **Test Set Performance:**
    *   **R-squared (R²):** 0.533
    *   **RMSE:** 0.532

#### RandomizedSearchCV Results

*   **Best Hyperparameters:** `{'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': None}`
*   **Best Cross-Validation R-squared:** 0.6016
*   **Test Set Performance:**
    *   **R-squared (R²):** **0.546**
    *   **RMSE:** **0.524**

### Overall Model Performance Comparison (Test Set)

| Model                             | R-squared (R²) | RMSE  |
| :-------------------------------- | :------------- | :---- |
| Original Fingerprint Model        | 0.517          | 0.541 |
| GridSearchCV Fingerprint Model    | 0.533          | 0.532 |
| RandomizedSearchCV Fingerprint Model | **0.546**    | **0.524** |
| Descriptor-based Model            | 0.471          | 0.566 |

**Conclusion:** The **RandomizedSearchCV-tuned model** achieved the highest R-squared and lowest RMSE, making it the best-performing model in this study. Hyperparameter tuning clearly improved performance over the baseline.

### Feature Importance Analysis

Understanding which molecular features contribute most to activity is crucial for rational drug design.

#### Top 20 Most Important Features (Morgan Fingerprint Bits)

These are the most influential fingerprint bits from the best-performing **RandomizedSearchCV model**. While directly interpreting the exact substructure for each bit requires specialized RDKit functions, their high importance indicates that the presence or absence of these specific structural patterns is highly predictive.

```
morgan_1729    0.103123
morgan_1086    0.094664
morgan_675     0.051307
morgan_1197    0.043211
morgan_41      0.039579
morgan_866     0.037984
morgan_1871    0.036055
morgan_1603    0.020981
morgan_1088    0.020539
morgan_862     0.020167
morgan_1720    0.019389
morgan_1535    0.019083
morgan_116     0.018281
morgan_1573    0.017744
morgan_142     0.015062
morgan_1602    0.012581
morgan_1964    0.012467
morgan_980     0.010728
morgan_1087    0.008389
morgan_110     0.008294
```

![Fingerprint Feature Importances Plot](https://github.com/Rishiatweb/Bioinformatics_CDD/blob/main/Important%20Features.png)

#### Top 20 Most Important Features (RDKit Descriptors)

This analysis used a model trained on comprehensive RDKit descriptors for better interpretability, though it had slightly lower predictive performance (R² = 0.471).

```
FpDensityMorgan3       0.088133
AvgIpc                 0.074304
SMR_VSA5               0.038775
fr_pyridine            0.035761
Chi4v                  0.034197
Chi3v                  0.032673
fr_sulfide             0.030792
FpDensityMorgan1       0.029787
SMR_VSA6               0.029527
VSA_EState6            0.019258
MaxAbsPartialCharge    0.018511
MinPartialCharge       0.017989
Chi2v                  0.016498
Chi0                   0.015871
EState_VSA10           0.015124
TPSA                   0.013683
EState_VSA8            0.013387
SMR_VSA7               0.012771
VSA_EState2            0.012595
MolLogP                0.012055
```

![Descriptor Feature Importances Plot](https://github.com/Rishiatweb/Bioinformatics_CDD/blob/main/descriptors.png)

*   **Key Interpretive Insights:**
    *   **Molecular Density (FpDensityMorganX):** These descriptors, consistently at the top, highlight that the density of atoms and bonds within a molecule (how "packed" it is) is a crucial factor for its activity against Mpro.
    *   **Information Content (AvgIpc):** Suggests that the structural complexity or diversity of atomic environments within a molecule influences its inhibitory potency.
    *   **Specific Functional Groups (fr_pyridine, fr_sulfide):** The high importance of these fragment counts indicates that the presence of pyridine rings and sulfide linkages might be beneficial for binding to the main protease.
    *   **Molecular Shape and Connectivity (ChiX):** Chi indices are topological descriptors related to molecular branching and shape. Their prominence suggests that the overall molecular architecture and how atoms are connected are important for activity.
    *   **Polarity & Surface Area (MolLogP, TPSA):** While LogP was not a standalone differentiator in initial statistical tests, its importance in the regression model implies it contributes in combination with other features. TPSA (Topological Polar Surface Area) is critical for interactions and permeability.

## Getting Started

This project is designed for execution in a Google Colab environment, which simplifies dependency management and provides access to computational resources.

### Prerequisites

*   A Google Account to use Google Colab and Google Drive.
*   Basic understanding of Python and machine learning concepts.

### How to Run

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/[YourGitHubUsername]/[YourRepositoryName].git
    cd [YourRepositoryName]
    ```
2.  **Open in Google Colab:**
    *   Go to [Google Colab](https://colab.research.google.com/).
    *   Click `File` -> `Upload notebook...` and select the `bioinformatics_1ext_all_possible_testings.ipynb` file from your cloned repository.
3.  **Execute the Notebook:**
    *   Run all cells in order from top to bottom (`Runtime` -> `Run all`).
    *   You will be prompted to mount your Google Drive. Authorize the connection to allow the notebook to save and load files (`.csv`, `.pdf`) to your Drive.
    *   The notebook will automatically handle the installation of all necessary Python libraries, including `chembl_webresource_client` and `rdkit` (via Miniconda installation within Colab).
    *   Be aware that `conda install` and `pip install rdkit` steps might take several minutes to complete as they set up the RDKit environment.
    *   The notebook will output detailed performance metrics and generate various plots (saved as `.pdf` files and also displayed in the notebook).

*(Note: The `/content/drive/MyDrive/Bioactivity/data1` directory is created and populated in your Google Drive during the notebook's execution to store intermediate data files.)*

## Future Work

This project lays a solid foundation for further research and development:

*   **Expand Dataset:** Incorporate more bioactivity data for a wider range of coronavirus targets or compounds with diverse scaffolds to improve model generalizability.
*   **Explore Other ML Algorithms:** Investigate the performance of other advanced regression algorithms such as XGBoost, LightGBM, Support Vector Regressors (SVR), or even deep learning models (e.g., Graph Neural Networks for molecules) for potential accuracy gains.
*   **Advanced Feature Selection:** Apply more sophisticated feature selection or dimensionality reduction techniques to refine the feature set, potentially leading to simpler yet equally predictive models.
*   **Molecular Docking & Dynamics:** Integrate computational chemistry methods like molecular docking and molecular dynamics simulations to provide a physical basis for the observed QSARs, exploring how compounds interact with the protein target at an atomic level.
*   **Virtual Screening & Experimental Validation:** Utilize the developed QSAR models for large-scale virtual screening of chemical libraries to identify novel potential inhibitors, followed by experimental validation (e.g., in vitro assays) of the top-predicted compounds.
*   **Multi-Objective Optimization:** Develop models that can predict multiple desirable properties simultaneously (e.g., activity, ADMET properties like toxicity, solubility, permeability) to guide the design of truly effective and safe drug candidates.

## Acknowledgments

This project was inspired by and follows a methodology similar to the "Computational Drug Discovery" tutorial series by the [Data Professor](http://youtube.com/dataprofessor) (Dr. Chanin Nantasenamat), a valuable resource for applied machine learning in cheminformatics.
