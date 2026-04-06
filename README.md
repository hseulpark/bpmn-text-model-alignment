# BPMN Text–Model Alignment (Research Prototype)

This repository contains a research prototype for evaluating alignment between **process model structure** (CPEE XML, BPMN-like) and **natural-language process descriptions**. The main goal is to detect mismatches between the user text specification and the model, including **task-level** and **control-flow/gateway-level** errors.

---

## Contents (What is included in this repo)

This repository is structured to support reproducibility and thesis submission requirements:

- **README.md** (this file): setup + how to run everything  
- **requirements.txt**: installable dependency list  
- **Similarity + matching implementation**: included in the comparator code (`compare_text_model.py` and its imported modules)  
- **Comparator CLI**: `compare_text_model.py` (text–model comparison, writes JSON report)  
- **Similarity usage example (non-CLI)**: `sim_example.py`  
- **Error injection implementation**: `inject_errors.py` (inject controlled errors into CPEE XML)  
- **Injection usage example (non-CLI)**: `inj_example.py`  
- **Test data**: `./test/` (small set of example models/texts for quick validation)  
- **Example process models for injection**: `./examples/models/` (2–3 XML models)  
- **Evaluation pipeline code**: `./evaluation/` (batch scripts used in the thesis)  
- **Generated evaluation models**: `models_with_error/` (includes the models used in the evaluation)  
- **Evaluation artifacts**: `./evaluation/artifacts/` (CSV/XLSX tables, figures, result summaries)

### Original model sources

The **ground-truth source models** are not necessarily included in full. They originate from the Chair of Information Systems and Business Process Management (TUM) repository:

- https://github.com/com-pot-93/cpee-models

This project uses subsets such as `domain`, `pet`, and `sapsam` as base models for generating faulty variants via error injection.

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

### A. Compare one process description against one XML model (CLI)

This runs the comparator and writes a JSON report.

```bash
python3 compare_text_model.py \
  --log process_description/domain/gdpr_1_data_breach.autobpmn.txt \
  --bpmn models_with_error/domain/gdpr_1_data_breach.err3.001.xml \
  --report_root report/
```

**Output:** A JSON report is written into `report/` (or your specified `--report_root`).

Each report contains:
- paths (`log_path`, `bpmn_path`)
- extracted tasks (`user_tasks`, `model_tasks`)
- detected issues (`comparison`)
- original user text and optional model-generated text (`user_text`, `output_text`)

---

### B. Similarity / mapping usage outside the CLI (`sim_example.py`)

To demonstrate how to use the similarity + mapping logic directly from Python:

```bash
python3 sim_example.py
```

What this example should show:
- computing similarity between task labels
- matching user tasks to model tasks
- printing matched pairs and similarity scores

---

### C. Inject errors into one model (CLI)

#### Inject multiple random errors (default)

```bash
python3 inject_errors.py \
  --in_xml examples/models/example_1.xml \
  --out_dir models_with_error/examples \
  --n_errors 3 \
  --seed 42 \
  --log_csv error_injection_log.csv
```

#### Inject exactly one forced error (used for balanced evaluation sets)

```bash
python3 inject_errors.py \
  --in_xml examples/models/example_1.xml \
  --out_dir models_with_error/examples \
  --n_errors 1 \
  --seed 42 \
  --force_error and_to_seq \
  --log_csv error_injection_log.csv
```

#### Supported `--force_error` values

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

#### Injection logs

The injection script writes logs that record what was changed:
- CSV log (per generated model): `--log_csv ...`
- JSON history (more details per run): `--log_json ...` (if enabled in your version)

---

### D. Injection usage outside the CLI (`inj_example.py`)

To demonstrate how to call the injection logic from Python code:

```bash
python3 inj_example.py
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
1. generate / assemble a balanced evaluation set (one injected error per instance)
2. run batch comparison
3. aggregate JSON reports into tables / spreadsheets

### Run balanced evaluation batch

Example command (exact command used in the thesis evaluation):

```bash
python3 evaluation/run_compare_eval_balanced.py \
  --models_dir models_with_error/eval_balanced \
  --desc_root process_description \
  --script compare_text_model.py \
  --report_root report_batch_eval
```

Outputs:
- JSON reports stored under `report_batch_eval/` grouped by error category (folder names)

### Evaluation artifacts

All tables/Excel files/figures used for evaluation interpretation are stored in:

- `evaluation/artifacts/`

This typically includes:
- balanced evaluation CSV/XLSX logs
- hit counts / hit rates tables
- exported figures used in the thesis

---

## Test Data (`./test`)

This repository includes a small test suite under `./test/` to validate correctness quickly.

Suggested structure:
- `test/models/` : small XML models  
- `test/text/` : corresponding process descriptions  
- `test/expected/` : notes or expected outcomes per example  

To run the minimal validation (if a test runner is included):

```bash
python3 evaluation/run_one_test.py
```

(If you do not provide a test runner, you can still validate manually by running the comparator on the files in `test/`.)

---

## Project Layout (High-level)

- `compare_text_model.py` : main comparator CLI (produces JSON reports)  
- `inject_errors.py` : error injection CLI (produces mutated XML + logs)  
- `sim_example.py` : similarity demo without CLI  
- `inj_example.py` : injection demo without CLI  
- `examples/models/` : 2–3 example models for quick injection tests  
- `test/` : minimal test data for validation  
- `evaluation/` : scripts used for thesis evaluation (batch runs, aggregation)  
- `models_with_error/` : generated models (includes evaluation models)  
- `process_description/` : natural language descriptions used as user inputs  
- `report/`, `report_batch_eval/` : generated JSON reports  

---

## Notes on External Dependencies

- The comparator uses spaCy (`en_core_web_lg`) for task extraction.
- SBERT embeddings use `sentence-transformers/all-MiniLM-L6-v2`.
- Some workflows may call an external API (if enabled in your generator script). Internet access is required for such steps.

---

## How to get help / troubleshoot

- Check script help:
  - `python3 compare_text_model.py --help`
  - `python3 inject_errors.py --help`
- If spaCy model is missing:
  - `python -m spacy download en_core_web_lg`
- If you hit HuggingFace warnings or rate limits:
  - set `HF_TOKEN` in your environment (optional)

---

## License / Attribution

Original base models were obtained from:
- Chair of Information Systems and Business Process Management (TUM): https://github.com/com-pot-93/cpee-models

This repository contains generated derivatives (error-injected models) for research evaluation.
