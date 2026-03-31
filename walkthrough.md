# Week 9 — AST+NLP ML Training Walkthrough

## What Was Built

The current pipeline uses AST-derived features as model input and natural-language docstrings as model targets.

### Active ML Files

| File | Purpose |
|------|---------|
| `src/ml/ast_feature_formatter.py` | Converts AST/context signals into structured model input text |
| `src/ml/ast_dataset_builder.py` | Builds AST+NLP training pairs from CodeSearchNet + stdlib |
| `src/ml/ast_comment_model.py` | T5-based comment model used for ML generation |
| `src/ml/trainer.py` | Train/evaluate/save flow for AST+NLP |
| `src/ml/evaluator.py` | BLEU-4, ROUGE-L, exact-match |
| `tests/test_ast_nlp_model.py` | Core AST+NLP ML tests |

### Runtime Engines

- `Rule-Based`: AST deterministic generation (`generate_comments`)
- `AST+NLP ML`: AST-driven model generation (`ml_generate_comments`)

## Training Data Contract

Every training sample follows:

- `input_text`: structured AST feature text from `ast_feature_formatter`
- `target_text`: first sentence of a function docstring

`training_report.json` explicitly tracks this under:

- `data_profile.input_mode = "AST structured features"`
- `data_profile.target_mode = "NLP docstring sentence"`
- `data_profile.ast_nlp_pairs`

## Output Files

```text
outputs/
├── ast_train_dataset.json
├── ast_train_dataset.csv
├── training_report.json
├── eval_report.json
└── model/
    └── ast_model/
```

## How to Run

```bash
# Train AST+NLP model
python3 -m src.main --train --output-dir outputs

# Fast smoke-training configuration
python3 -m src.main --train --output-dir outputs \
  --codesearchnet-max 20 --max-stdlib-files 10 --epochs 1 --batch-size 2

# Rule-based generation
python3 -m src.main tests/inputs/complex_sample.py

# AST+NLP ML generation
python3 -m src.main tests/inputs/complex_sample.py --ml

# Targeted tests
python3 -m pytest tests/test_ast_nlp_model.py -q
```

## Strict ML Behavior

- `--ml` now requires a trained AST+NLP model.
- If the model is missing or cannot be loaded, the command exits with an explicit error and training hint.
- Rule-Based generation is available only when ML mode is not selected.

## Medium Quality Preset (RTX 4050)

Use this as the recommended quality/speed balance:

```bash
python3 -m src.main --train \
  --codesearchnet-max 8000 \
  --max-stdlib-files 600 \
  --epochs 3 \
  --batch-size 8 \
  --grad-accum-steps 1
```
