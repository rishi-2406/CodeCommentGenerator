try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel


class SecurityGraphWidget(QWidget):
    """Widget displaying security analysis as matplotlib charts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        if not _MPL_OK:
            self._label = QLabel("Install matplotlib for security charts: pip install matplotlib")
            self._label.setStyleSheet("color: #94a3b8; padding: 20px;")
            self._layout.addWidget(self._label)
            self._fig = None
            return

        self._fig = Figure(figsize=(8, 6), facecolor="#1e293b")
        self._canvas = FigureCanvas(self._fig)
        self._layout.addWidget(self._canvas)

    def set_data(self, security_report=None):
        if not _MPL_OK or not self._fig:
            return

        self._fig.clear()
        self._fig.set_facecolor("#1e293b")

        if security_report is None:
            ax = self._fig.add_subplot(111, facecolor="#0f172a")
            ax.text(0.5, 0.5, "No security data.\nRun analysis first.",
                    ha="center", va="center", fontsize=14, color="#94a3b8")
            ax.set_xticks([])
            ax.set_yticks([])
            self._canvas.draw()
            return

        scores = security_report.function_scores
        issues = security_report.issues
        by_sev = security_report.by_severity
        safe_pct = security_report.module_safe_pct

        # 1. Safety Pie Chart
        ax1 = self._fig.add_subplot(2, 2, 1, facecolor="#0f172a")
        safe_count = sum(1 for s in scores.values() if s >= 80)
        unsafe_count = len(scores) - safe_count
        if scores:
            labels = ["Safe", "Unsafe"]
            sizes = [safe_count, unsafe_count]
            colors = ["#22c55e", "#ef4444"]
            explode = [0.05 if unsafe_count > 0 else 0]
            wedges, texts, autotexts = ax1.pie(
                sizes, labels=labels, colors=colors, autopct="%1.0f%%",
                startangle=90, textprops={"color": "#cbd5e1", "fontsize": 10},
            )
            for t in autotexts:
                t.set_fontsize(11)
                t.set_fontweight("bold")
        else:
            ax1.text(0.5, 0.5, "N/A", ha="center", va="center", color="#94a3b8", fontsize=16)
        ax1.set_title(f"Safety: {safe_pct}% Safe", color="#e2e8f0", fontsize=12, fontweight="bold")

        # 2. Issues by Severity
        ax2 = self._fig.add_subplot(2, 2, 2, facecolor="#0f172a")
        if by_sev:
            sev_order = ["critical", "high", "medium", "low"]
            sev_colors = {"critical": "#ef4444", "high": "#f97316", "medium": "#eab308", "low": "#3b82f6"}
            labels = [s for s in sev_order if by_sev.get(s, 0) > 0]
            values = [by_sev.get(s, 0) for s in labels]
            colors = [sev_colors.get(s, "#64748b") for s in labels]
            ax2.bar(labels, values, color=colors, edgecolor="#334155", width=0.5)
            for i, v in enumerate(values):
                ax2.text(i, v + 0.1, str(v), ha="center", va="bottom", color="#cbd5e1", fontsize=10)
        ax2.set_title("Issues by Severity", color="#e2e8f0", fontsize=12, fontweight="bold")
        ax2.tick_params(colors="#64748b")
        ax2.set_ylabel("Count", color="#94a3b8", fontsize=10)
        for spine in ax2.spines.values():
            spine.set_color("#334155")
        ax2.grid(True, axis="y", color="#334155", alpha=0.3)

        # 3. Per-Function Security Scores
        ax3 = self._fig.add_subplot(2, 2, 3, facecolor="#0f172a")
        if scores:
            names = list(scores.keys())[:12]
            vals = [scores[n] for n in names]
            bar_colors = ["#22c55e" if v >= 80 else "#eab308" if v >= 50 else "#ef4444" for v in vals]
            y_pos = range(len(names))
            ax3.barh(y_pos, vals, color=bar_colors, edgecolor="#334155", height=0.6)
            ax3.set_yticks(list(y_pos))
            ax3.set_yticklabels(names, color="#cbd5e1", fontsize=8)
            ax3.set_xlim(0, 100)
            ax3.axvline(x=80, color="#22c55e", linestyle="--", alpha=0.5, linewidth=1)
        ax3.set_title("Function Security Scores", color="#e2e8f0", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Score (0-100)", color="#94a3b8", fontsize=10)
        ax3.tick_params(colors="#64748b")
        for spine in ax3.spines.values():
            spine.set_color("#334155")
        ax3.grid(True, axis="x", color="#334155", alpha=0.3)

        # 4. Issue Type Distribution
        ax4 = self._fig.add_subplot(2, 2, 4, facecolor="#0f172a")
        if issues:
            pattern_counts = {}
            for i in issues:
                pid = i.pattern_id
                pattern_counts[pid] = pattern_counts.get(pid, 0) + 1
            sorted_pats = sorted(pattern_counts.items(), key=lambda x: -x[1])[:8]
            if sorted_pats:
                pids = [p[0] for p in sorted_pats]
                counts = [p[1] for p in sorted_pats]
                ax4.bar(pids, counts, color="#8b5cf6", edgecolor="#334155", width=0.5)
                for i, v in enumerate(counts):
                    ax4.text(i, v + 0.1, str(v), ha="center", va="bottom", color="#cbd5e1", fontsize=9)
        ax4.set_title("Issue Types", color="#e2e8f0", fontsize=12, fontweight="bold")
        ax4.tick_params(colors="#64748b", labelsize=7)
        ax4.set_ylabel("Count", color="#94a3b8", fontsize=10)
        for spine in ax4.spines.values():
            spine.set_color("#334155")
        ax4.grid(True, axis="y", color="#334155", alpha=0.3)

        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()
