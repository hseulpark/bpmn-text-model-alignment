# BPMN Text–Model Alignment (Bachelor's Thesis)

This repository contains a research prototype for evaluating alignment between **BPMN process model structure** (CPEE XML) and **natural-language process descriptions**. The main goal is to detect mismatches between the user text specification and the model, including **task-level** and **control-flow/gateway-level** errors.

---

## Contents (What is included in this repo)

This repository is structured to support reproducibility and thesis submission requirements:

- **README.md** (this file): setup + how to run everything  
- **requirements.txt**: installable dependency list  
- **Similarity + matching implementation**: implemented in `compare_text_model.py` and its imported functions  
- **Comparator CLI**: `compare_text_model.py` (text–model comparison, writes JSON report)  
- **Similarity usage example (non-CLI)**: `examples/sim_example.py`  
- **Error injection implementation**: `inject_errors.py` (inject controlled errors into CPEE XML)  
- **Injection usage example (non-CLI)**: `examples/inj_example.py`  
- **Test data**: `./test/` (small set of example models/texts + generated reports)  
- **Examples**: `./examples/` (small “real demo” inputs + outputs)  
- **Evaluation pipeline code**: `./evaluation/` (batch scripts used in the thesis)  
- **Generated evaluation models**: `models_with_error/` (models used in evaluation runs)  
- **Generated evaluation reports**: `report_batch_eval/` (reports produced during balanced evaluation)  
- **Evaluation artifacts**: `evaluation/artifacts/` (CSV/XLSX table)

---

## Dataset note (ground_truth / process_description not included)

The original model repository is maintained by the Chair of Information Systems and Business Process Management (TUM):

- https://github.com/com-pot-93/cpee-models

This submission repository **does not include** the full original `ground_truth/` and `process_description/` folders.
Instead, it contains:

- small representative **example models + texts** under `examples/` (for demo runs), and
- the **generated models** used for evaluation under `models_with_error/`.

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 3. Sanity check

```bash
python3 compare_text_model.py --help
python3 inject_errors.py --help
```

---

## Core Workflows

### A. Compare one text file against one XML model (CLI)

This runs the comparator and writes a JSON report.

```bash
python3 compare_text_model.py \
  --log "test/text/8. and-or.txt" \
  --bpmn "test/models/8. and-or.xml" \
  --report_root test/reports
```

**Output:** A JSON report is written into `test/reports/` (or your specified `--report_root`).

Each report contains:
- paths (`log_path`, `bpmn_path`)
- extracted tasks (`user_tasks`, `model_tasks`)
- detected issues (`comparison`)
- original user text and optional model-generated text (`user_text`, `output_text`)

> If your file name contains spaces, wrap the path in quotes (recommended), or escape spaces with `\ `.

---

### B. Similarity / mapping usage outside the CLI (`examples/sim_example.py`)

This demonstrates how to use the similarity + mapping + comparator logic **from Python**, without using the CLI as the primary interface.

Example:

```bash
python3 examples/sim_example.py \
  --xml examples/models/healthcare_bpmai_2.xml \
  --txt examples/text/healthcare_bpmai_2.autobpmn.txt \
  --report_dir examples/reports/comparison_results
```

What this example should show:
- loading text + model programmatically
- extracting tasks
- computing similarity and greedy 1:1 matches
- generating and saving a JSON report under `examples/reports/comparison_results`

---

### C. Inject errors into one model (CLI)

Inject multiple random errors:

```bash
python3 inject_errors.py \
  --in_xml examples/models/healthcare_bpmai_2.xml \
  --out_dir models_with_error/examples \
  --n_errors 3 \
  --seed 42 \
  --log_csv evaluation/artifacts/error_injection_log.csv
```

Inject exactly one forced error:

```bash
python3 inject_errors.py \
  --in_xml examples/models/healthcare_bpmai_2.xml \
  --out_dir models_with_error/examples \
  --n_errors 1 \
  --seed 42 \
  --force_error and_to_seq \
  --log_csv evaluation/artifacts/error_injection_log.csv
```

Supported `--force_error` values:

Task-level errors:
- `missing_task`
- `additional_task`
- `merged`
- `two_wrong_sequences`
- `random_sequences`

Gateway/control-flow errors:
- `and_to_xor`
- `and_to_seq`
- `xor_to_and`
- `xor_to_seq`

---

### D. Injection usage outside the CLI (`examples/inj_example.py`)

To demonstrate how to call the injection logic from Python code:

```bash
python3 examples/inj_example.py \
  --xml examples/models/healthcare_bpmai_2.xml \
  --force_error additional_task \
  --seed 42 \
  --out_dir examples/reports/injected
```

Or inject multiple random errors:

```bash
python3 examples/inj_example.py \
  --xml examples/models/gdpr_7_right_to_be_forgotten.json.xml \
  --n_errors 3 \
  --seed 42 \
  --out_dir examples/reports/injected
```

What this example should show:
- loading one XML model
- applying one forced error or N random errors programmatically
- saving the output model and printing which error was applied

---

## Batch Evaluation (Thesis)

All evaluation scripts used in the thesis are stored under:

- `./evaluation/`

Typical workflow:
1. generate / assemble an evaluation set (including injected errors)
2. run batch comparison
3. aggregate JSON reports into tables / spreadsheets

Example command used for balanced evaluation runs:

```bash
python3 evaluation/run_compare_eval_balanced.py \
  --models_dir models_with_error/eval_balanced \
  --script compare_text_model.py \
  --report_root report_batch_eval
```

Outputs:
- JSON reports stored under `report_batch_eval/` grouped by error category (folder names)

Evaluation artifacts (tables/figures) are under:
- `evaluation/artifacts/`

---

## Test Data (`./test`)

This repository includes a small test suite under `./test/`:

- `test/models/` : small XML models  
- `test/text/` : corresponding process descriptions  
- `test/reports/` : generated JSON reports  

---

## Project Layout (High-level)

- `compare_text_model.py` : main comparator CLI (produces JSON reports)  
- `inject_errors.py` : error injection CLI (produces mutated XML + logs)  
- `examples/sim_example.py` : similarity demo without CLI  
- `examples/inj_example.py` : injection demo without CLI  
- `examples/models/`, `examples/text/` : demo inputs  
- `examples/reports/` : demo outputs (comparison results, injected models)  
- `evaluation/` : scripts used for thesis evaluation (batch runs, aggregation)  
- `models_with_error/` : generated models used in evaluation  
- `report_batch_eval/` : generated evaluation JSON reports  
- `test/` : minimal test data + example reports  

---

## Notes on External Dependencies

- The comparator uses spaCy (`en_core_web_lg`) for task extraction.
- SBERT embeddings use `sentence-transformers/all-MiniLM-L6-v2`.
- Gateway analysis may call an external API (AutoBPMN); internet access may be required for those steps.

---

## Troubleshooting

- Check script help:
  - `python3 compare_text_model.py --help`
  - `python3 inject_errors.py --help`
- If spaCy model is missing:
  - `python -m spacy download en_core_web_lg`

---

## License / Attribution

Original base models are from:
- Chair of Information Systems and Business Process Management (TUM): https://github.com/com-pot-93/cpee-models

This repository contains generated derivatives (e.g., error-injected models and reports) for research evaluation.
