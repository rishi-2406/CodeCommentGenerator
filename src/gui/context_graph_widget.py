import math
import random
from PyQt6.QtWidgets import QWidget, QToolTip
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
)

_COMPLEXITY_COLORS = {
    "simple": QColor("#22c55e"),
    "moderate": QColor("#eab308"),
    "complex": QColor("#f97316"),
    "very_complex": QColor("#ef4444"),
}


class _CallGraphNode:
    def __init__(self, label, x=0.0, y=0.0, complexity="simple",
                 security_issues=None, calls_made=None):
        self.label = label
        self.x = x
        self.y = y
        self.complexity = complexity
        self.security_issues = security_issues or []
        self.calls_made = calls_made or []
        self.outgoing: list["_CallGraphNode"] = []
        self.vx = 0.0
        self.vy = 0.0
        self.radius = 30


class ContextGraphWidget(QWidget):
    """Custom widget rendering the call graph as a directed graph."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._nodes: list[_CallGraphNode] = []
        self._node_map: dict[str, _CallGraphNode] = {}
        self._edges: list[tuple[_CallGraphNode, _CallGraphNode]] = []
        self._hovered: _CallGraphNode | None = None
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._scale = 1.0
        self._dragging = False
        self._last_mouse = None
        self._layout_done = False
        self.setMinimumSize(400, 300)

    def set_data(self, context_graph):
        self._nodes.clear()
        self._node_map.clear()
        self._edges.clear()
        self._layout_done = False

        for fc in context_graph.function_contexts:
            node = _CallGraphNode(
                label=fc.name,
                complexity=fc.complexity_label,
                security_issues=getattr(fc, 'security_issues', []),
                calls_made=fc.calls_internal + fc.calls_external,
            )
            self._nodes.append(node)
            self._node_map[fc.name] = node

        module_fn_names = context_graph.module_function_names
        for fc in context_graph.function_contexts:
            src = self._node_map.get(fc.name)
            if not src:
                continue
            for call_name in fc.calls_internal:
                dst = self._node_map.get(call_name)
                if dst and dst is not src:
                    if (src, dst) not in self._edges:
                        src.outgoing.append(dst)
                        self._edges.append((src, dst))

        self._force_layout()
        self.update()

    def _force_layout(self):
        if not self._nodes:
            return
        w = max(self.width(), 600)
        h = max(self.height(), 400)

        for n in self._nodes:
            n.x = random.uniform(80, w - 80)
            n.y = random.uniform(80, h - 80)

        for _ in range(200):
            for n in self._nodes:
                n.vx = 0.0
                n.vy = 0.0

            for i, a in enumerate(self._nodes):
                for j in range(i + 1, len(self._nodes)):
                    b = self._nodes[j]
                    dx = b.x - a.x
                    dy = b.y - a.y
                    dist = math.sqrt(dx * dx + dy * dy) + 0.1
                    force = 8000.0 / (dist * dist)
                    fx = force * dx / dist
                    fy = force * dy / dist
                    a.vx -= fx
                    a.vy -= fy
                    b.vx += fx
                    b.vy += fy

            for src, dst in self._edges:
                dx = dst.x - src.x
                dy = dst.y - src.y
                dist = math.sqrt(dx * dx + dy * dy) + 0.1
                force = (dist - 150) * 0.05
                fx = force * dx / dist
                fy = force * dy / dist
                src.vx += fx
                src.vy += fy
                dst.vx -= fx
                dst.vy -= fy

            for n in self._nodes:
                n.vx -= n.x * 0.01
                n.vy -= n.y * 0.01
                n.vx *= 0.85
                n.vy *= 0.85
                n.x += n.vx
                n.y += n.vy
                n.x = max(60, min(w - 60, n.x))
                n.y = max(60, min(h - 60, n.y))

        cx = sum(n.x for n in self._nodes) / max(len(self._nodes), 1)
        cy = sum(n.y for n in self._nodes) / max(len(self._nodes), 1)
        for n in self._nodes:
            n.x -= cx - w / 2
            n.y -= cy - h / 2

        self._layout_done = True

    def paintEvent(self, event):
        if not self._nodes:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(self._offset_x, self._offset_y)
        painter.scale(self._scale, self._scale)

        for src, dst in self._edges:
            self._draw_edge(painter, src, dst)

        for node in self._nodes:
            self._draw_node(painter, node)

    def _draw_edge(self, painter, src: _CallGraphNode, dst: _CallGraphNode):
        painter.setPen(QPen(QColor("#a855f7"), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        path.moveTo(src.x, src.y)
        dx = dst.x - src.x
        dy = dst.y - src.y
        dist = math.sqrt(dx * dx + dy * dy) + 0.1
        ctrl_x = (src.x + dst.x) / 2 + dy * 0.15
        ctrl_y = (src.y + dst.y) / 2 - dx * 0.15
        path.quadTo(ctrl_x, ctrl_y, dst.x, dst.y)
        painter.drawPath(path)

        arrow_len = 8
        angle = math.atan2(dy, dx)
        tip_x = dst.x - dst.radius * math.cos(angle)
        tip_y = dst.y - dst.radius * math.sin(angle)
        p1x = tip_x - arrow_len * math.cos(angle - 0.4)
        p1y = tip_y - arrow_len * math.sin(angle - 0.4)
        p2x = tip_x - arrow_len * math.cos(angle + 0.4)
        p2y = tip_y - arrow_len * math.sin(angle + 0.4)
        arrow = QPainterPath()
        arrow.moveTo(tip_x, tip_y)
        arrow.lineTo(p1x, p1y)
        arrow.lineTo(p2x, p2y)
        arrow.closeSubpath()
        painter.setBrush(QBrush(QColor("#a855f7")))
        painter.drawPath(arrow)

    def _draw_node(self, painter, node: _CallGraphNode):
        color = _COMPLEXITY_COLORS.get(node.complexity, QColor("#64748b"))
        is_hovered = node is self._hovered

        if node.security_issues:
            border_pen = QPen(QColor("#ef4444"), 3)
        elif is_hovered:
            border_pen = QPen(QColor("#ffffff"), 2.5)
        else:
            border_pen = QPen(color.darker(120), 1.5)

        painter.setPen(border_pen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(node.x, node.y), node.radius, node.radius)

        painter.setPen(QPen(QColor("#ffffff")))
        font = QFont("Inter", 8, QFont.Weight.Bold)
        painter.setFont(font)
        text = node.label[:10] if len(node.label) > 10 else node.label
        rect = QRectF(node.x - node.radius, node.y - node.radius,
                       node.radius * 2, node.radius * 2)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _node_at(self, pos: QPointF) -> _CallGraphNode | None:
        for node in self._nodes:
            dx = pos.x() - node.x
            dy = pos.y() - node.y
            if dx * dx + dy * dy <= node.radius * node.radius:
                return node
        return None

    def mouseMoveEvent(self, event):
        pos = QPointF(
            (event.position().x() - self._offset_x) / self._scale,
            (event.position().y() - self._offset_y) / self._scale,
        )
        if self._dragging and self._last_mouse:
            dx = event.position().x() - self._last_mouse.x()
            dy = event.position().y() - self._last_mouse.y()
            self._offset_x += dx
            self._offset_y += dy
            self._last_mouse = event.position()
            self.update()
            return

        hovered = self._node_at(pos)
        if hovered != self._hovered:
            self._hovered = hovered
            if hovered:
                details = [f"Function: {hovered.label}",
                           f"Complexity: {hovered.complexity}",
                           f"Calls: {', '.join(hovered.calls_made[:5])}"]
                if hovered.security_issues:
                    details.append(f"Security: {len(hovered.security_issues)} issues")
                QToolTip.showText(event.globalPosition().toPoint(), "\n".join(details), self)
            else:
                QToolTip.hideText()
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_mouse = event.position()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._last_mouse = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._scale *= factor
        self._scale = max(0.2, min(5.0, self._scale))
        self.update()

    def resizeEvent(self, event):
        if self._nodes and not self._layout_done:
            self._force_layout()
            self.update()
