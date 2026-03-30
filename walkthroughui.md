# Code Comment Generator GUI — Walkthrough

## Summary of Changes
This GUI wraps the CLI pipeline with a PyQt6 interface and now aligns with the current two-engine architecture:

- Rule-Based (AST deterministic)
- ML-Based (AST+NLP model)

## What Was Implemented

### 1. Main Navigation System (`src/gui/main_window.py`)
- We built a `QMainWindow` hosting a fixed left-sidebar and a central `QStackedWidget` for tabless workspace routing.
- The sidebar implements the exact visual hierarchy of the design, giving developers easy top-level access to the active workspace.
- A functional Top Header containing the **"Generate & Attach Comments"** global action and real-time status notifications. 

### 2. Comment Generator Workspace (`src/gui/generator_workspace.py`)
- **Split-Pane Editor:** On the left, a syntax-highlighted code editor with a "Pick File" hardware picker. On the right, a read-only output viewer.
- **Engine Toggle:** Users can switch between `Rule-Based` and `ML-Based (AST+NLP)`.
- **Asynchronous Execution:** Generation is handled inside a `QThread` (`GeneratorWorker`) so UI stays responsive.
- **ML Loading Path:** The workspace loads `load_ast_model(...)` and passes `ast_model` into `run_pipeline(...)`. If no trained model exists, it falls back to Rule-Based mode with a warning.

### 3. Insights & Code Visualizer (`src/gui/insights_workspace.py`)
Instead of having to pass CLI flags (`--show-features`, `--ir`, `--analysis`), the GUI automatically parses the outputs of a successful comment generation pass and constructs comprehensive views.
- **AST Features:** JSON viewer displaying detected imports, functions, and classes.
- **Context Graph:** Detailed relationships between the code logic.
- **IR Dump & Analysis:** The Intermediate Representation (IR) blocks mapped out alongside localized pattern anomalies (e.g. cyclomatic complexity alerts or security flags).

### 4. ML Models Dashboard (`src/gui/training_workspace.py`)
- A dedicated workspace for running `--train` behavior without opening terminal.
- Uses asynchronous `TrainerWorker` for non-blocking training.
- Dashboard text now reflects AST+NLP-only training reports:
  - dataset totals/splits
  - AST+NLP pair count
  - BLEU-4 / ROUGE-L / exact-match

## Manual Verification
You can test the result smoothly from your terminal:

```bash
python run_gui.py
```

### Quick Test Flow
1. Open the app.
2. In Generator, click **Pick File** and choose a Python file (e.g. `tests/inputs/complex_sample.py`).
3. Select `Rule-Based` and click **Generate & Attach Comments**.
4. Switch to `ML-Based (AST+NLP)` and run again.
5. Open **Insights** to inspect extracted AST/context/analysis output.
