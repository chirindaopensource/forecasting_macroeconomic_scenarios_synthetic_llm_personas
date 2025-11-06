### **`README.md`**

# Replication of "*Prompting for Policy: Forecasting Macroeconomic Scenarios with Synthetic LLM Personas*"

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.02458v1-b31b1b.svg)](https://arxiv.org/abs/2511.02458)
[![Conference](https://img.shields.io/badge/Conference-ACM%20ICAIF%202025-9cf)](https://icaif.acm.org/2025/)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/forecasting_macroeconomic_scenarios_synthetic_llm_personas)
[![Discipline](https://img.shields.io/badge/Discipline-Computational%20Economics-00529B)](https://github.com/chirindaopensource/forecasting_macroeconomic_scenarios_synthetic_llm_personas)
[![Data Source](https://img.shields.io/badge/Data%20Source-ECB%20SPF-003299)](https://www.ecb.europa.eu/stats/ecb_surveys/survey_of_professional_forecasters/html/index.en.html)
[![Data Source](https://img.shields.io/badge/Data%20Source-PersonaHub-FFD21E)](https://huggingface.co/datasets/proj-persona/PersonaHub)
[![Core Method](https://img.shields.io/badge/Method-LLM%20Forecasting%20%7C%20Ablation%20Study-orange)](https://github.com/chirindaopensource/forecasting_macroeconomic_scenarios_synthetic_llm_personas)
[![Analysis](https://img.shields.io/badge/Analysis-Time%20Series%20%7C%20Hypothesis%20Testing-red)](https://github.com/chirindaopensource/forecasting_macroeconomic_scenarios_synthetic_llm_personas)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)](https://openai.com/index/hello-gpt-4o/)
[![spaCy](https://img.shields.io/badge/spaCy-%2309A3D5.svg?style=flat&logo=spaCy&logoColor=white)](https://spacy.io/)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-2E4053-blue)](https://www.sbert.net/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)

**Repository:** `https://github.com/chirindaopensource/forecasting_macroeconomic_scenarios_synthetic_llm_personas`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Prompting for Policy: Forecasting Macroeconomic Scenarios with Synthetic LLM Personas"** by:

*   Giulia Iadisernia
*   Carolina Camassa

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from rigorous data validation and cleansing to a large-scale, multi-stage persona filtering process, high-volume asynchronous forecast generation, and the final statistical analysis, including the central ablation study.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `run_synthetic_economist_study`](#key-callable-run_synthetic_economist_study)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Iadisernia and Camassa (2025). The core of this repository is the iPython Notebook `forecasting_macroeconomic_scenarios_synthetic_llm_personas_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline is designed to be a robust and scalable system for evaluating the impact of persona-based prompting on the macroeconomic forecasting performance of Large Language Models (LLMs).

The paper's central research question is whether sophisticated persona descriptions improve LLM performance in a real-world forecasting task. This codebase operationalizes the paper's experimental design, allowing users to:
-   Rigorously validate and manage the entire experimental configuration via a single `config.yaml` file.
-   Execute a multi-stage filtering pipeline on the ~370M-entry PersonaHub corpus to distill a set of 2,368 high-quality, domain-specific expert personas.
-   Systematically replicate 50 rounds of the ECB Survey of Professional Forecasters using GPT-4o.
-   Generate over 120,000 individual forecasts across a main "persona" arm and a "no-persona" baseline for a controlled ablation study.
-   Perform a comprehensive statistical analysis comparing the accuracy (MAE, win-share) and disagreement (dispersion) of the AI panels against the human expert panel.
-   Run hypothesis tests (Monte Carlo and exact binomial) to assess the statistical significance of the findings.
-   Execute the final ablation study tests (paired t-test, Kolmogorov-Smirnov test) to formally evaluate the impact of personas.

## Theoretical Background

The implemented methods are grounded in the principles of experimental design, computational linguistics, and time-series forecast evaluation.

**1. Persona-Based Prompting:**
The core hypothesis is that providing an LLM with a detailed "persona" or "role" can improve its performance on domain-specific reasoning tasks. This is tested via an ablation study, where the performance of prompts with personas is compared to identical prompts without them.

**2. Forecast Evaluation Metrics:**
-   **Mean Absolute Error (MAE):** A standard metric for point forecast accuracy, measuring the average magnitude of forecast errors.
    $$
    \mathrm{MAE}_{vh} = \frac{1}{n_{vh}} \sum_{r=1}^{n_{vh}} | \hat{y}_{rvh} - y_{rvh} |
    $$
-   **Win-Share:** A head-to-head comparison metric that counts the proportion of forecast rounds where one panel's forecast was strictly more accurate than another's, excluding ties.
    $$
    w_{vh} = \frac{W_{vh}}{n_{vh}}, \quad \text{where } W_{vh} = \sum_{r} \mathbf{1}\{ e_{rvh}^{\mathrm{AI}} < e_{rvh}^{\mathrm{H}} \}
    $$

**3. Hypothesis Testing:**
-   **Null Hypothesis:** The AI panel and human panel have an equal probability of producing a more accurate forecast ($H_0: p=0.5$).
-   **Test for Large Samples (In-Sample):** The null distribution is approximated using a **Monte Carlo simulation** with $N=10,000$ draws from a Binomial distribution, $W^* \sim \text{Binom}(n_{vh}, 0.5)$.
-   **Test for Small Samples (Out-of-Sample):** An **exact Binomial test** is used, calculating probabilities directly from the Binomial PMF.

## Features

The provided iPython Notebook (`forecasting_macroeconomic_scenarios_synthetic_llm_personas_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The entire pipeline is broken down into 25 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All study parameters are managed in an external `config.yaml` file.
-   **Scalable Data Processing:** Includes streaming validators and processors for the large-scale PersonaHub dataset.
-   **Resilient API Orchestration:** A robust, asynchronous framework for managing over 120,000 API calls with concurrency control, rate limiting, automatic retries, and resumability via checkpointing.
-   **Advanced Persona Filtering:** A four-stage pipeline combining keyword filtering, NER, semantic deduplication (via embeddings and HNSW), and a multi-run, majority-vote LLM-as-a-judge triage.
-   **Rigorous Statistical Analysis:** Implements all specified forecast evaluation metrics and hypothesis tests with high fidelity.
-   **Complete Replication:** A single top-level function call can execute the entire study from raw data to final result tables.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Validation & Cleansing (Tasks 1-4):** Ingests and validates all raw inputs, including the ~370M-entry PersonaHub file and all analytical datasets.
2.  **Persona Filtering (Tasks 5-9):** Executes the four-stage filtering pipeline to derive the final `persona_final_df` of 2,368 personas. Includes a Cohen's kappa validation of the LLM judge.
3.  **Forecast Generation (Tasks 10-12):** Assembles all 123,400 prompts and executes the API calls for both the persona and baseline arms.
4.  **Analysis & Scoring (Tasks 13-22):** Consolidates and QCs all forecasts, computes AI panel medians, aligns all data sources, calculates dispersion, errors, MAE, and win-shares, and runs all hypothesis tests.
5.  **Ablation Study (Tasks 24-25):** Runs the final paired t-test and Kolmogorov-Smirnov test on the aligned results to formally test the paper's main hypothesis.

## Core Components (Notebook Structure)

The `forecasting_macroeconomic_scenarios_synthetic_llm_personas_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the 25 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `run_synthetic_economist_study`

The project is designed around a single, top-level user-facing interface function:

-   **`run_synthetic_economist_study`:** This master orchestrator function, located in the final section of the notebook, runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project.

## Prerequisites

-   Python 3.9+
-   An OpenAI API key.
-   Core dependencies: `pandas`, `numpy`, `pyyaml`, `pyarrow`, `openai`, `spacy`, `sentence-transformers`, `networkx`, `hnswlib`, `scipy`, `scikit-learn`, `tqdm`, `faker`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/forecasting_macroeconomic_scenarios_synthetic_llm_personas.git
    cd forecasting_macroeconomic_scenarios_synthetic_llm_personas
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Download the spaCy model:**
    ```sh
    python -m spacy download en_core_web_trf
    ```

5.  **Set your OpenAI API Key:**
    ```sh
    export OPENAI_API_KEY='your-key-here'
    ```

## Input Data Structure

The pipeline requires several input files with specific schemas, which are rigorously validated. A synthetic data generator is included in the notebook for a self-contained demonstration.
1.  **`persona_hub_raw.parquet`**: The large-scale persona dataset.
2.  **`contextual_data.csv`**: Time-series data for the 50 SPF rounds.
3.  **`human_benchmark.csv`**: Human expert panel median forecasts.
4.  **`human_micro.csv`**: Individual human expert forecasts.
5.  **`realized_outcomes.csv`**: Ground-truth macroeconomic data.
6.  **`human_annotations.csv`**: Human judgments for the kappa validation.

All other parameters are controlled by the `config.yaml` file.

## Usage

The `forecasting_macroeconomic_scenarios_synthetic_llm_personas_draft.ipynb` notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell of the notebook, which demonstrates how to use the top-level `run_synthetic_economist_study` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Define paths and load configuration.
    run_dir = Path("./synthetic_economist_run")
    raw_data_dir = run_dir / "raw_data"
    output_dir = run_dir / "output"
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Generate a full set of synthetic data files for the demonstration.
    # (The generation functions are defined in the notebook)
    data_paths = setup_synthetic_data(raw_data_dir, config)
    
    # 3. Execute the entire replication study.
    final_artifacts = run_synthetic_economist_study(
        data_paths=data_paths,
        config=config,
        output_dir=str(output_dir),
        total_persona_rows=10000, # Use the size of our synthetic dataset
        run_kappa_validation=True
    )
    
    # 4. Inspect final results.
    print("--- Ablation Paired T-Test Report ---")
    print(final_artifacts['ablation_ttest_report'])
```

## Output Structure

The pipeline generates a structured output directory:
-   **`output/processed/`**: Contains intermediate, processed data files (e.g., cleansed and filtered persona sets).
-   **`output/checkpoints/`**: Contains raw JSONL results from all API calls, enabling resumability.
-   **`output/results/`**: Contains all final output tables (MAE, win-share, dispersion) as CSV files and a comprehensive `full_pipeline_report.json`.
-   **`output/pipeline_run.log`**: A detailed log file for the entire run.

## Project Structure

```
forecasting_macroeconomic_scenarios_synthetic_llm_personas/
│
├── forecasting_macroeconomic_scenarios_synthetic_llm_personas_draft.ipynb
├── config.yaml
├── requirements.txt
│
├── study_run/
│   ├── raw_data/
│   └── output/
│       ├── processed/
│       ├── results/
│       ├── checkpoints/
│       └── pipeline_run.log
│
├── LICENSE
└── README.md
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify all study parameters, including model names, API settings, filtering thresholds, and file paths, without altering the core Python code.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Evaluating Different LLMs:** The modular design allows for easy substitution of the `model_name` in the config to test other models (e.g., from Anthropic, Google).
-   **Density Forecasting:** Extending the prompts to ask for probability distributions (as in the real SPF) and evaluating the quality of the LLM's density forecasts.
-   **Alternative Prompting Strategies:** Implementing and testing other prompting techniques, such as chain-of-thought or adversarial prompting, to see if they can generate more diverse or accurate forecasts.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@inproceedings{iadisernia2025prompting,
  author = {Iadisernia, Giulia and Camassa, Carolina},
  title = {Prompting for Policy: Forecasting Macroeconomic Scenarios with Synthetic LLM Personas},
  year = {2025},
  booktitle = {Proceedings of the 6th ACM International Conference on AI in Finance},
  series = {ICAIF '25}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Production-Grade Replication of "Prompting for Policy: Forecasting Macroeconomic Scenarios with Synthetic LLM Personas".
GitHub repository: https://github.com/chirindaopensource/forecasting_macroeconomic_scenarios_synthetic_llm_personas
```

## Acknowledgments

-   Credit to **Giulia Iadisernia and Carolina Camassa** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, spaCy, Sentence-Transformers, NetworkX, Scipy, Scikit-learn, and OpenAI**.

--

*This README was generated based on the structure and content of the `forecasting_macroeconomic_scenarios_synthetic_llm_personas_draft.ipynb` notebook and follows best practices for research software documentation.*
