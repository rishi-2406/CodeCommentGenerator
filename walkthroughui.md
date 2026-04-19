# Code Comment Generator GUI — Walkthrough

## Summary of Changes

This GUI wraps the CLI pipeline with a PyQt6 interface supporting three generation engines:

- **Rule-Based** — deterministic AST-driven generation
- **ML-Based (AST+NLP)** — T5 model generates summary from structured AST features, now with full Args/Returns/Raises sections
- **Neurosymbolic** — confidence-gated fusion: ML summary + symbolic validation + structural sections

## Workspaces

### 1. Generator (`src/gui/generator_workspace.py`)
- **Split-Pane Editor:** Input code (left) and annotated output (right) with syntax highlighting
- **Engine Toggle:** Three-way radio: Rule-Based / Neurosymbolic / ML-Based
- **% Unsafe Indicator:** Real-time display of security analysis results (green/yellow/red)
- **Asynchronous Execution:** `GeneratorWorker` (QThread) for non-blocking pipeline runs
- Engine type is passed to `run_pipeline()` via the `engine` parameter

### 2. Insights & Code Visualizer (`src/gui/insights_workspace.py`)
Eight tabs with full visualization of pipeline results:
- **AST Features** — tree widget showing functions, classes, params, complexity
- **AST Graph** — interactive tree graph (QPainter-rendered, pan/zoom/hover tooltips)
  - Color-coded nodes: Module=gray, Import=yellow, Class=green, Function=blue, AsyncFunction=purple, Param=cyan
  - Hover for detail info; click-drag to pan; scroll to zoom
- **Context Graph** — tree widget with complexity coloring and security warnings
- **Call Graph** — force-directed directed graph visualization (QPainter)
  - Nodes colored by complexity (green/yellow/orange/red)
  - Red border on nodes with security issues
  - Arrow edges show call relationships; hover for details
- **IR Dump** — syntax-highlighted intermediate representation text
- **Analysis Report** — table of pattern findings (P001–P007)
- **Evaluation** — matplotlib charts: training loss curve, metrics bar chart, confidence vs quality scatter
- **Security** — matplotlib charts: safety pie chart, severity bars, per-function scores, issue types

### 3. ML Training (`src/gui/training_workspace.py`)
- Train/retrain AST+NLP model from the GUI
- Uses `TrainerWorker` (QThread) for non-blocking training
- Displays AST+NLP training reports with metrics

### 4. Security (`src/gui/security_workspace.py`)
- **Module Safety Score** — large indicator showing % safe (green/yellow/red)
- **% Unsafe** — prominent display of unsafe code percentage
- **Security Charts** — pie chart (safe vs unsafe), severity bars, function scores, issue type distribution
- **Issues Table** — severity, pattern ID, function, line, message + remediation

### 5. Logs (`src/gui/logs_workspace.py`)
- File tree browser for pipeline logs, training outputs, reports
- Syntax-highlighted viewer for JSON and log files

## Graph Widget Architecture

| Widget | Rendering | File |
|--------|-----------|------|
| AST Graph | QPainter (custom tree layout) | `src/gui/ast_graph_widget.py` |
| Call Graph | QPainter (force-directed layout) | `src/gui/context_graph_widget.py` |
| Evaluation Charts | matplotlib FigureCanvas | `src/gui/eval_graph_widget.py` |
| Security Charts | matplotlib FigureCanvas | `src/gui/security_graph_widget.py` |

All graph widgets support:
- Pan (click-drag)
- Zoom (scroll wheel)
- Hover tooltips with detail information

## Pipeline Data Flow

```
Generation completes
  → results_dict includes: mf, cg, ir, analysis, security_report
  → InsightsWorkspace.populate_insights(results_dict)
      → AST Features tree + AST Graph
      → Context tree + Call Graph
      → IR Dump + Analysis Table
      → Evaluation Charts
      → Security Charts
  → SecurityWorkspace.populate(security_report)
      → Safety score + % unsafe
      → Security charts + Issues table
  → GeneratorWorkspace.update_unsafe_pct(security_report)
      → Header % unsafe indicator
```

## Repository Cleanup (Phase 7)

- Removed 33 tracked `.pyc` files from git index (already gitignored, but had been committed earlier)
- Removed pipeline logs and GUI output files from `out/` tracking
- Added `out/` to `.gitignore` — pipeline artifacts no longer clutter version control

## Manual Verification

```bash
python run_gui.py
```

### Quick Test Flow
1. Open the app.
2. In Generator, click **Pick File** and choose `tests/inputs/test_security.py`.
3. Select **Neurosymbolic** and click **Generate & Attach Comments**.
4. Check the **% unsafe** indicator in the Generator header.
5. Switch to **Insights** to explore all 8 tabs including AST Graph and Call Graph.
6. Switch to **Security** to see the safety score and security issues.
7. Select **Rule-Based** and run again to compare output quality.
