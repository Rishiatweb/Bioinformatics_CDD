# AI Agent for QSAR-Driven Drug Discovery: An Active Learning Simulation

This repository contains a Jupyter Notebook (`ChemBL_agent_simulation.ipynb`) that implements and simulates an AI agent for accelerating molecular bioactivity prediction. The project demonstrates how active learning strategies can be applied to Quantitative Structure-Activity Relationship (QSAR) modeling to more efficiently guide the process of drug discovery.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
<a href="https://colab.research.google.com/github/Rishiatweb/Bioinformatics_CDD/blob/main/ChemBL_agent/ChemBL_agent_simulation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

## Table of Contents
1.  [Abstract](#abstract)
2.  [Scientific Background](#scientific-background)
    - [Quantitative Structure-Activity Relationship (QSAR)](#quantitative-structure-activity-relationship-qsar)
    - [Active Learning (AL)](#active-learning-al)
3.  [Project Workflow and Code Description](#project-workflow-and-code-description)
    - [Step 1: Environment Setup](#step-1-environment-setup)
    - [Step 2: Data Acquisition and Preprocessing](#step-2-data-acquisition-and-preprocessing)
    - [Step 3: Feature Engineering with Morgan Fingerprints](#step-3-feature-engineering-with-morgan-fingerprints)
    - [Step 4: Baseline Model Training](#step-4-baseline-model-training)
    - [Step 5: The AI Agent Simulation (Active Learning Loop)](#step-5-the-ai-agent-simulation-active-learning-loop)
    - [Step 6: Interactive Prediction Interface](#step-6-interactive-prediction-interface)
4.  [Results and Discussion](#results-and-discussion)
5.  [How to Run the Simulation](#how-to-run-the-simulation)
6.  [Potential Extensions and Future Work](#potential-extensions-and-future-work)
7.  [License](#license)
8.  [Citation](#citation)

## Abstract

The process of drug discovery is notoriously long and expensive, largely due to the vast chemical space that must be explored. This project demonstrates a computational approach to make this exploration more efficient. We simulate an AI agent that uses **active learning** to intelligently guide the development of a **Quantitative Structure-Activity Relationship (QSAR)** model. The agent starts with a small set of known compounds and iteratively predicts the bioactivity of a large pool of "undiscovered" molecules. Based on its predictions, it selects the most promising candidates for "experimental testing" (a simulation of revealing their true activity). This new information is then used to retrain and improve its internal model. The simulation shows that this intelligent, goal-directed strategy leads to a more rapid improvement in model performance compared to passive learning, thereby demonstrating a more resource-efficient path to discovering potent lead compounds.

## Scientific Background

### Quantitative Structure-Activity Relationship (QSAR)
QSAR modeling is a cornerstone of computational chemistry and cheminformatics. It is based on the principle that the biological activity of a chemical compound is directly related to its molecular structure. By creating a mathematical model that links structural features (descriptors) to observed activity, QSAR allows for the prediction of properties for novel, untested molecules. In this project, we build a QSAR model using a Random Forest Regressor to predict the `pIC50` value (the negative log of the half-maximal inhibitory concentration) of compounds targeting the SARS coronavirus 3C-like proteinase.

### Active Learning (AL)
Active Learning is a subfield of machine learning where the learning algorithm is permitted to choose the data from which it learns. In a typical supervised learning scenario (passive learning), the model is trained on a randomly selected, pre-labeled dataset. In contrast, an active learning agent starts with a small amount of labeled data and a large pool of unlabeled data. It iteratively queries an "oracle" (such as a human expert or a real-world experiment) to provide labels for the data points that it deems most informative. The goal is to achieve high model accuracy with fewer labeled training instances, which is critical in domains like drug discovery where labeling (i.e., synthesizing and testing a compound) is extremely expensive.

## Project Workflow and Code Description

The project is implemented as a single, sequential Jupyter Notebook.

### Step 1: Environment Setup
-   **Libraries:** The notebook begins by installing essential libraries, including `chembl_webresource_client` for data acquisition, `rdkit` for cheminformatics tasks (via Conda for stable installation in Colab), and `scikit-learn` for machine learning.
-   **Google Drive:** It mounts Google Drive to ensure data persistence for the raw and preprocessed datasets.

### Step 2: Data Acquisition and Preprocessing
-   **Target:** Bioactivity data is acquired from the ChEMBL database for a specific biological target: **SARS coronavirus 3C-like proteinase (CHEMBL3927)**.
-   **Data Cleaning:** The raw data is filtered for compounds with a standard `IC50` value. It is then cleaned by removing entries with missing values.
-   **pIC50 Calculation:** The `IC50` values (in nM) are converted to the logarithmic `pIC50` scale (`pIC50 = -log10(IC50_Molar)`), which is more suitable for regression modeling.
-   **Bioactivity Classification:** Compounds are classified as `active` (IC50 ≤ 1000 nM), `inactive` (IC50 ≥ 10000 nM), or `intermediate`.

### Step 3: Feature Engineering with Morgan Fingerprints
-   To enable the machine learning model to understand molecular structures, each compound's SMILES string is converted into a fixed-length numerical vector.
-   **Morgan Fingerprints** (a type of Extended-Connectivity Fingerprint, or ECFP) are used. These fingerprints encode the presence of various circular chemical substructures within a molecule, creating a robust representation for QSAR modeling. A 2048-bit vector is generated for each molecule.

### Step 4: Baseline Model Training
-   Before initiating the agent, a baseline model is established to provide a performance benchmark.
-   The dataset is split into three parts:
    1.  **Initial Training Set (20%):** A small set to train the very first model.
    2.  **Discovery Pool (60%):** The agent's "environment"—a large set of compounds whose bioactivity is considered unknown to the agent.
    3.  **Final Test Set (20%):** A held-out set used exclusively for evaluating the performance of the model at each iteration. It is never seen during training or selection.
-   A `RandomForestRegressor` is trained on the initial training set, and its performance (R² and RMSE) is evaluated on the final test set. This is "Iteration 0."

### Step 5: The AI Agent Simulation (Active Learning Loop)
This is the core of the project, simulating the agent's decision-making process over several iterations. The agent is defined by its goal, environment, perceptions, actions, and learning mechanism.

-   **Goal:** To build the most accurate QSAR model by efficiently selecting compounds to "test."
-   **Environment:** The `discovery_X_pool` and `discovery_y_pool`.
-   **The Agent's Iterative Loop:**
    1.  **Perception:** The agent uses its current model to predict the `pIC50` for every compound in the discovery pool.
    2.  **Decision (Policy):** It employs an *exploitation* strategy by sorting the compounds by their predicted `pIC50` and selecting the top 10 candidates it believes are most active.
    3.  **Action (Simulated Experiment):** The agent "tests" these 10 compounds. In the simulation, this involves retrieving their true `pIC50` values from the `y_discovery_pool`.
    4.  **Learning:** These 10 newly "labeled" compounds are removed from the discovery pool and added to the agent's training set.
-   **Model Refinement:** The agent's internal `RandomForestRegressor` model is retrained with this newly augmented dataset. Its performance is re-evaluated on the fixed final test set, and the results are stored. This loop repeats for a defined number of iterations.

### Step 6: Interactive Prediction Interface
-   The final, refined model produced by the agent at the end of the active learning loop is used to power an interactive widget.
-   A user can input a novel SMILES string, and the agent's model will predict its `pIC50` and bioactivity class.
-   Additionally, it performs a **Lipinski's Rule of Five** analysis on the input compound to assess its potential for oral bioavailability, adding another layer of drug-likeness evaluation.

## Results and Discussion
The simulation generates learning curves that plot the model's R-squared (R²) and Root Mean Squared Error (RMSE) against the number of active learning iterations.



The plots clearly show that as the agent iteratively selects promising compounds and retrains its model, its performance on the unseen final test set generally improves. The upward trend in R² and the downward trend in RMSE validate the active learning approach. This demonstrates that by intelligently selecting which data to acquire, the agent can build a more predictive model more efficiently than a passive strategy that relies on random sampling.

## How to Run the Simulation
This project is designed to run seamlessly in Google Colab.

1.  Click the "Open in Colab" badge at the top of this README.
2.  In the Colab interface, click `Runtime` -> `Run all` from the menu bar.
3.  The notebook will install all necessary dependencies, run the data pipeline, train the baseline model, execute the active learning simulation, and finally display the interactive prediction widget. The entire process may take a few minutes to complete.

No local installation is required.

## Potential Extensions and Future Work
-   **Advanced Active Learning Strategies:** Implement more sophisticated selection strategies beyond simple exploitation, such as uncertainty sampling (choosing compounds the model is least sure about) or diversity sampling (choosing compounds from underrepresented chemical spaces).
-   **Different Machine Learning Models:** Replace the Random Forest with other algorithms like Gradient Boosting models (XGBoost, LightGBM) or more advanced Graph Neural Networks (GNNs) that can learn directly from the graph structure of molecules.
-   **Application to Different Targets:** Adapt the pipeline to a different biological target by changing the ChEMBL ID in the data acquisition step.
-   **Integration with Generative Models:** Combine the predictive agent with a generative model that designs novel molecules, creating a closed-loop system for de novo drug design.

## Citation
If you use this code or concepts from this project in your work, please cite this repository. Also you are welcome to contribute

```
@misc{chembl_agent_simulation_2024,
  author = https://github.com/Rishiatweb,
  title = {AI Agent for QSAR-Driven Drug Discovery: An Active Learning Simulation},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/[YourUsername]/[YourRepoName]}}
}
