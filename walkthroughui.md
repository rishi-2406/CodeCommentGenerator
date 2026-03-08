# Code Comment Generator GUI — Walkthrough

## Summary of Changes
We have successfully built a massive graphical wrapper around your existing AI-powered code comment generator CLI using **PyQt6**. Taking direct inspiration from the requested Stitch design (`#197fe6` blue accents, deep Slate background colors, rounded mono-space fonts), we engineered a multi-threaded desktop application that exposes every corner of your compiler-design pipeline.

The application serves as a central hub for generating comments, visualizing Abstract Syntax Trees, finetuning ML models, and browsing historical logs.

![Stitch Screen Input](/home/rishi/.gemini/antigravity/brain/b943a775-aeb5-44b8-85f0-f2c5eb423195/artifacts/stitch_design.png)

## What Was Implemented

### 1. Main Navigation System ([src/gui/main_window.py](file:///media/XPG/College/college/SEM%204/compiler-design/lab/week9/src/gui/main_window.py))
- We built a `QMainWindow` hosting a fixed left-sidebar and a central `QStackedWidget` for tabless workspace routing.
- The sidebar implements the exact visual hierarchy of the design, giving developers easy top-level access to the active workspace.
- A functional Top Header containing the **"Generate & Attach Comments"** global action and real-time status notifications. 

### 2. Comment Generator Workspace ([src/gui/generator_workspace.py](file:///media/XPG/College/college/SEM%204/compiler-design/lab/week9/src/gui/generator_workspace.py))
- **Split-Pane Editor:** On the left, a syntax-highlighted code editor with a "Pick File" hardware picker. On the right, a read-only output viewer.
- **Engine Toggle:** Users can switch between the Rule-Based Generator and the ML-Based CodeT5 generator smoothly.
- **Asynchronous Execution:** Generation is handled inside a `QThread` ([GeneratorWorker](file:///media/XPG/College/college/SEM%204/compiler-design/lab/week9/src/gui/generator_workspace.py#17-70)). This ensures that parsing, NLP generation, and AST-context graph analysis never freeze the GUI.

### 3. Insights & Code Visualizer ([src/gui/insights_workspace.py](file:///media/XPG/College/college/SEM%204/compiler-design/lab/week9/src/gui/insights_workspace.py))
Instead of having to pass CLI flags (`--show-features`, `--ir`, `--analysis`), the GUI automatically parses the outputs of a successful comment generation pass and constructs comprehensive views.
- **AST Features:** JSON viewer displaying detected imports, functions, and classes.
- **Context Graph:** Detailed relationships between the code logic.
- **IR Dump & Analysis:** The Intermediate Representation (IR) blocks mapped out alongside localized pattern anomalies (e.g. cyclomatic complexity alerts or security flags).

### 4. ML Models Dashboard ([src/gui/training_workspace.py](file:///media/XPG/College/college/SEM%204/compiler-design/lab/week9/src/gui/training_workspace.py))
- A dedicated workspace for executing the massive `--train` workflow without needing a CLI.
- Handled by an asynchronous [TrainerWorker](file:///media/XPG/College/college/SEM%204/compiler-design/lab/week9/src/gui/training_workspace.py#10-29), allowing the user to initiate the multi-stage setup: Dataset splitting → TF-IDF Model Fits → Template Ranking → CodeT5 Fine-Tuning.
- Automatically captures the `training_report.json` and `eval_report.json` to paint a clean dashboard view of key metrics (BLEU-4, ROUGE-L, TF-IDF CV Accuracy).

## Manual Verification
You can test the result smoothly from your terminal:

```bash
python run_gui.py
```

### Quick Test Flow
1. Open the App. 
2. In the Generator Workspace, click **Pick File** and select any python file (even [src/main.py](file:///media/XPG/College/college/SEM%204/compiler-design/lab/week9/src/main.py)).
3. Click the bright blue **Generate & Attach Comments** button at the top.
4. Watch the status bar update. Once successful, the generated code will appear on the right pane!
5. Navigate to the **Insights** tab to explore the AST data that was just extracted.
