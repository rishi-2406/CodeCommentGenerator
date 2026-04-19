try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel


class SecurityGraphWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        if not _MPL_OK:
            self._label = QLabel("Install matplotlib for security charts: pip install matplotlib")
            self._label.setStyleSheet("color: #6c757d; padding: 20px;")
            self._layout.addWidget(self._label)
            self._fig = None
            return

        self._fig = Figure(figsize=(8, 6), facecolor="#ffffff")
        self._canvas = FigureCanvas(self._fig)
        self._layout.addWidget(self._canvas)

    def set_data(self, security_report=None):
        if not _MPL_OK or not self._fig:
            return

        self._fig.clear()
        self._fig.set_facecolor("#ffffff")

        if security_report is None:
            ax = self._fig.add_subplot(111, facecolor="#f8f9fa")
            ax.text(0.5, 0.5, "No security data.\nRun analysis first.",
                    ha="center", va="center", fontsize=14, color="#6c757d")
            ax.set_xticks([])
            ax.set_yticks([])
            self._canvas.draw()
            return

        scores = security_report.function_scores
        issues = security_report.issues
        by_sev = security_report.by_severity
        safe_pct = security_report.module_safe_pct

        ax1 = self._fig.add_subplot(2, 2, 1, facecolor="#f8f9fa")
        safe_count = sum(1 for s in scores.values() if s >= 80)
        unsafe_count = len(scores) - safe_count
        if scores:
            labels = ["Safe", "Unsafe"]
            sizes = [safe_count, unsafe_count]
            colors = ["#16a34a", "#dc2626"]
            explode = [0.05 if unsafe_count > 0 else 0]
            wedges, texts, autotexts = ax1.pie(
                sizes, labels=labels, colors=colors, autopct="%1.0f%%",
                startangle=90, textprops={"color": "#1a1a2e", "fontsize": 10},
            )
            for t in autotexts:
                t.set_fontsize(11)
                t.set_fontweight("bold")
        else:
            ax1.text(0.5, 0.5, "N/A", ha="center", va="center", color="#6c757d", fontsize=16)
        ax1.set_title(f"Safety: {safe_pct}% Safe", color="#1a1a2e", fontsize=12, fontweight="bold")

        ax2 = self._fig.add_subplot(2, 2, 2, facecolor="#f8f9fa")
        if by_sev:
            sev_order = ["critical", "high", "medium", "low"]
            sev_colors = {"critical": "#dc2626", "high": "#ea580c", "medium": "#ca8a04", "low": "#2563eb"}
            labels = [s for s in sev_order if by_sev.get(s, 0) > 0]
            values = [by_sev.get(s, 0) for s in labels]
            colors = [sev_colors.get(s, "#6b7280") for s in labels]
            ax2.bar(labels, values, color=colors, edgecolor="#d0d5dd", width=0.5)
            for i, v in enumerate(values):
                ax2.text(i, v + 0.1, str(v), ha="center", va="bottom", color="#1a1a2e", fontsize=10)
        ax2.set_title("Issues by Severity", color="#1a1a2e", fontsize=12, fontweight="bold")
        ax2.tick_params(colors="#6c757d")
        ax2.set_ylabel("Count", color="#495057", fontsize=10)
        for spine in ax2.spines.values():
            spine.set_color("#d0d5dd")
        ax2.grid(True, axis="y", color="#e0e4e8", alpha=0.5)

        ax3 = self._fig.add_subplot(2, 2, 3, facecolor="#f8f9fa")
        if scores:
            names = list(scores.keys())[:12]
            vals = [scores[n] for n in names]
            bar_colors = ["#16a34a" if v >= 80 else "#ca8a04" if v >= 50 else "#dc2626" for v in vals]
            y_pos = range(len(names))
            ax3.barh(y_pos, vals, color=bar_colors, edgecolor="#d0d5dd", height=0.6)
            ax3.set_yticks(list(y_pos))
            ax3.set_yticklabels(names, color="#1a1a2e", fontsize=8)
            ax3.set_xlim(0, 100)
            ax3.axvline(x=80, color="#16a34a", linestyle="--", alpha=0.5, linewidth=1)
        ax3.set_title("Function Security Scores", color="#1a1a2e", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Score (0-100)", color="#495057", fontsize=10)
        ax3.tick_params(colors="#6c757d")
        for spine in ax3.spines.values():
            spine.set_color("#d0d5dd")
        ax3.grid(True, axis="x", color="#e0e4e8", alpha=0.5)

        ax4 = self._fig.add_subplot(2, 2, 4, facecolor="#f8f9fa")
        if issues:
            pattern_counts = {}
            for i in issues:
                pid = i.pattern_id
                pattern_counts[pid] = pattern_counts.get(pid, 0) + 1
            sorted_pats = sorted(pattern_counts.items(), key=lambda x: -x[1])[:8]
            if sorted_pats:
                pids = [p[0] for p in sorted_pats]
                counts = [p[1] for p in sorted_pats]
                ax4.bar(pids, counts, color="#7c3aed", edgecolor="#d0d5dd", width=0.5)
                for i, v in enumerate(counts):
                    ax4.text(i, v + 0.1, str(v), ha="center", va="bottom", color="#1a1a2e", fontsize=9)
        ax4.set_title("Issue Types", color="#1a1a2e", fontsize=12, fontweight="bold")
        ax4.tick_params(colors="#6c757d", labelsize=7)
        ax4.set_ylabel("Count", color="#495057", fontsize=10)
        for spine in ax4.spines.values():
            spine.set_color("#d0d5dd")
        ax4.grid(True, axis="y", color="#e0e4e8", alpha=0.5)

        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()
