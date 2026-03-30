# Code Comment Generator — AST + NLP

This project generates Python comments/docstrings using two engines:

- `Rule-Based` (deterministic AST analysis)
- `AST+NLP ML` (model trained on structured AST features -> natural-language docstrings)

## Current ML Architecture

The active ML pipeline is centered on `src/ml/`:

- `ast_feature_formatter.py` — serializes extracted AST/context features into model input text
- `ast_dataset_builder.py` — builds `(input_text, target_text)` training pairs from:
  - CodeSearchNet Python
  - Python stdlib fallback
- `ast_comment_model.py` — T5-based generator used for AST+NLP comment generation
- `trainer.py` — end-to-end training/evaluation/report saving
- `evaluator.py` — BLEU-4 / ROUGE-L / exact-match metrics

## CLI Usage

### 1) Train AST+NLP model

```bash
python3 -m src.main --train
```

Useful tuning flags:

```bash
python3 -m src.main --train \
  --epochs 1 \
  --batch-size 2 \
  --codesearchnet-max 20 \
  --max-stdlib-files 10
```

### 2) Rule-based generation

```bash
python3 -m src.main tests/inputs/complex_sample.py
```

### 3) AST+NLP ML generation

```bash
python3 -m src.main tests/inputs/complex_sample.py --ml
```

If no trained model exists, `--ml` automatically falls back to Rule-Based mode.

## Output Artifacts

After training, artifacts are written in `outputs/`:

- `outputs/model/ast_model/` — trained AST+NLP model
- `outputs/ast_train_dataset.json` — saved AST/NLP dataset
- `outputs/ast_train_dataset.csv` — dataset in CSV format
- `outputs/training_report.json` — training metadata
- `outputs/eval_report.json` — evaluation metrics

`training_report.json` includes:

- `data_profile.input_mode = "AST structured features"`
- `data_profile.target_mode = "NLP docstring sentence"`
- `data_profile.ast_nlp_pairs`
