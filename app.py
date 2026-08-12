import random
import sys

from PyQt5.QtCore import Qt, QTimer, QElapsedTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QComboBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QSlider, QSizePolicy,
)

# ==========================================================================
# Sudoku core: generation + backtracking, decoupled from any UI toolkit.
# ==========================================================================

DIFFICULTIES = {"Fácil": 40, "Medio": 50, "Difícil": 60}


def create_full_board():
    """Randomized, fully-solved 9x9 sudoku via the classic band/stack shuffle."""
    base, side = 3, 9

    def pattern(r, c):
        return (base * (r % base) + r // base + c) % side

    def shuffle(seq):
        return random.sample(seq, len(seq))

    r_base = range(base)
    rows = [g * base + r for g in shuffle(r_base) for r in shuffle(r_base)]
    cols = [g * base + c for g in shuffle(r_base) for c in shuffle(r_base)]
    nums = shuffle(range(1, base * base + 1))

    return [[nums[pattern(r, c)] for c in cols] for r in rows]


def make_puzzle(difficulty):
    board = create_full_board()
    empty_cells = DIFFICULTIES[difficulty]
    removed = 0
    while removed < empty_cells:
        r, c = random.randint(0, 8), random.randint(0, 8)
        if board[r][c] != 0:
            board[r][c] = 0
            removed += 1
    return board


def has_conflict(board, row, col, num):
    """True if `num` already appears among the (row, col, box) peers of (row, col)."""
    for i in range(9):
        if i != col and board[row][i] == num:
            return True
        if i != row and board[i][col] == num:
            return True
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            rr, cc = box_row + i, box_col + j
            if (rr, cc) != (row, col) and board[rr][cc] == num:
                return True
    return False


def backtrack_solve(board, pos=0):
    """Recursive backtracking solver expressed as a generator so the caller can
    animate it: yields ('try'|'set'|'unset', row, col, num) events."""
    if pos == 81:
        return True
    row, col = divmod(pos, 9)
    if board[row][col] != 0:
        return (yield from backtrack_solve(board, pos + 1))

    for num in range(1, 10):
        yield ('try', row, col, num)
        if not has_conflict(board, row, col, num):
            board[row][col] = num
            yield ('set', row, col, num)
            if (yield from backtrack_solve(board, pos + 1)):
                return True
            board[row][col] = 0
            yield ('unset', row, col, None)
    return False


def solve_with_nn(board):
    """Solve using the pre-trained CNN (model_structure.json / best_weights.hdf5).
    Imported lazily so the app still runs without tensorflow/keras installed."""
    import numpy as np
    from keras.models import model_from_json

    with open('model_structure.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('best_weights.hdf5')
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    grid = ((np.array(board, dtype=float).reshape((9, 9, 1))) / 9) - 0.5
    while True:
        predictions = model.predict(grid.reshape((1, 9, 9, 1)), verbose=0).squeeze()
        pred = np.argmax(predictions, axis=1).reshape((9, 9)) + 1
        prob = np.around(np.max(predictions, axis=1).reshape((9, 9)), 2)

        grid = ((grid + 0.5) * 9).reshape((9, 9))
        mask = grid == 0
        if mask.sum() == 0:
            break
        idx = np.argmax(prob * mask)
        x, y = idx // 9, idx % 9
        grid[x][y] = pred[x][y]
        grid = (grid / 9) - 0.5

    # `grid` is already back in real digit-value space from the last loop
    # iteration's conversion (line 102) — converting again here would double it.
    return grid.astype(int).tolist()


# ==========================================================================
# Theming
# ==========================================================================

DARK = dict(
    bg="#0f1117", card="#1b1f2a", card_border="#2a2f3d",
    line_thick="#3a4152", line_thin="#262b38",
    cell_bg="#1b1f2a", cell_bg_given="#20242f",
    text="#f1f2f6", text_secondary="#9aa1b2", text_muted="#656d80",
    accent="#6366f1", accent_strong="#818cf8", accent_soft="#2b2c4a",
    success="#34d399", warning="#fbbf24", warning_soft="#4a3b19",
    danger="#f87171", danger_soft="#4a2323",
)

LIGHT = dict(
    bg="#f4f5f8", card="#ffffff", card_border="#e3e5ec",
    line_thick="#c3c8d6", line_thin="#dfe2ea",
    cell_bg="#ffffff", cell_bg_given="#eef0f6",
    text="#1a1d29", text_secondary="#5a6072", text_muted="#8b93a7",
    accent="#4f46e5", accent_strong="#4338ca", accent_soft="#e3e1fb",
    success="#059669", warning="#d97706", warning_soft="#fbe3bf",
    danger="#dc2626", danger_soft="#fad2d2",
)


def build_stylesheet(t):
    return f"""
    QMainWindow, QWidget#root {{ background: {t['bg']}; }}
    QLabel {{ color: {t['text']}; }}
    QLabel[role="title"] {{ font-size: 15px; font-weight: 600; }}
    QLabel[role="field"] {{ color: {t['text_secondary']}; font-size: 11px; }}
    QLabel[role="muted"] {{ color: {t['text_muted']}; font-size: 10px; }}
    QLabel[role="metric"] {{ color: {t['text']}; font-size: 20px; font-weight: 700; }}
    QLabel[role="status"] {{ color: {t['text_secondary']}; font-size: 12px; }}
    QLabel[role="brand"] {{ color: {t['text']}; font-size: 17px; font-weight: 700; }}

    QFrame#card {{
        background: {t['card']}; border: 1px solid {t['card_border']};
        border-radius: 14px;
    }}
    QFrame#board {{ background: {t['line_thick']}; border-radius: 14px; }}
    QFrame#box {{ background: {t['line_thin']}; }}

    QPushButton#cell {{
        background: {t['cell_bg']}; color: {t['text']};
        border: none; font-family: 'Consolas'; font-size: 20px; font-weight: 500;
    }}
    QPushButton#cell[given="true"] {{ background: {t['cell_bg_given']}; font-weight: 700; }}
    QPushButton#cell[solved="true"] {{ color: {t['success']}; font-weight: 600; }}
    QPushButton#cell[sameValue="true"] {{ background: {t['accent_soft']}; }}
    QPushButton#cell[selected="true"] {{
        background: {t['accent_soft']}; border: 2px solid {t['accent_strong']};
    }}
    QPushButton#cell[trying="true"] {{ background: {t['warning_soft']}; color: {t['warning']}; }}
    QPushButton#cell[error="true"] {{ background: {t['danger_soft']}; color: {t['danger']}; }}

    QPushButton#numBtn {{
        background: {t['card']}; color: {t['text']}; border: 1px solid {t['card_border']};
        border-radius: 8px; font-family: 'Consolas'; font-weight: 600; font-size: 14px;
    }}
    QPushButton#numBtn:hover {{ border-color: {t['accent']}; background: {t['accent_soft']}; }}

    QPushButton[cls="primary"] {{
        background: {t['accent']}; color: white; border: none;
        border-radius: 8px; padding: 9px 14px; font-weight: 600;
    }}
    QPushButton[cls="primary"]:hover {{ background: {t['accent_strong']}; }}
    QPushButton[cls="primary"]:disabled {{ background: {t['card_border']}; color: {t['text_muted']}; }}

    QPushButton[cls="secondary"] {{
        background: {t['bg']}; color: {t['text']}; border: 1px solid {t['card_border']};
        border-radius: 8px; padding: 9px 14px; font-weight: 600;
    }}
    QPushButton[cls="secondary"]:hover {{ border-color: {t['accent']}; }}
    QPushButton[cls="secondary"]:disabled {{ color: {t['text_muted']}; }}

    QPushButton[cls="ghost"] {{
        background: transparent; color: {t['text_secondary']}; border: 1px solid {t['card_border']};
        border-radius: 8px; padding: 8px 14px;
    }}
    QPushButton[cls="ghost"]:hover {{ color: {t['danger']}; border-color: {t['danger']}; }}

    QPushButton#iconBtn {{
        background: {t['card']}; border: 1px solid {t['card_border']}; border-radius: 8px;
        font-size: 14px;
    }}
    QPushButton#iconBtn:hover {{ border-color: {t['accent']}; }}

    QComboBox {{
        background: {t['bg']}; color: {t['text']}; border: 1px solid {t['card_border']};
        border-radius: 8px; padding: 6px 8px;
    }}
    QSlider::groove:horizontal {{ background: {t['card_border']}; height: 4px; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: {t['accent_strong']}; width: 16px; margin: -6px 0; border-radius: 8px;
    }}
    """


def repolish(widget):
    widget.style().unpolish(widget)
    widget.style().polish(widget)


# ==========================================================================
# Widgets
# ==========================================================================

class Cell(QPushButton):
    def __init__(self, row, col, on_click):
        super().__init__()
        self.row, self.col = row, col
        self.setObjectName("cell")
        self.setFlat(True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.clicked.connect(lambda: on_click(row, col))

    def set_flags(self, **flags):
        changed = False
        for name, value in flags.items():
            if self.property(name) != value:
                self.setProperty(name, value)
                changed = True
        if changed:
            repolish(self)


class SudokuWindow(QMainWindow):
    SPEED_DELAYS = [140, 30, 0]  # ms per animated step: slow, normal, instant
    SPEED_LABELS = ["Lento", "Normal", "Instantáneo"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sudoku Solver — Backtracking Visualizer")
        self.resize(980, 700)

        self.theme_name = "dark"
        self.board = [[0] * 9 for _ in range(9)]
        self.given = [[False] * 9 for _ in range(9)]
        self.solved_by_algo = [[False] * 9 for _ in range(9)]
        self.selected = None

        self.cells = [[None] * 9 for _ in range(9)]
        self.solver = None
        self.solving = False
        self.paused = False
        self.steps = 0
        self.timer = QElapsedTimer()
        self.step_timer = QTimer(self)
        self.step_timer.timeout.connect(self._solve_step)
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self._update_metrics)

        self._build_ui()
        self.apply_theme("dark")
        self.new_puzzle("Medio")

    # ---------------------------------------------------------------- UI --

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_topbar())

        layout = QHBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)
        layout.addLayout(self._build_board_panel(), stretch=2)
        layout.addWidget(self._build_control_panel(), stretch=1)
        outer.addLayout(layout)

    def _build_topbar(self):
        bar = QFrame()
        row = QHBoxLayout(bar)
        row.setContentsMargins(20, 14, 20, 14)
        brand = QLabel("Sudoku Solver")
        brand.setProperty("role", "brand")
        row.addWidget(brand)
        row.addStretch()
        self.theme_btn = QPushButton("☀")
        self.theme_btn.setObjectName("iconBtn")
        self.theme_btn.setFixedSize(36, 36)
        self.theme_btn.clicked.connect(self.toggle_theme)
        row.addWidget(self.theme_btn)
        return bar

    def _build_board_panel(self):
        col = QVBoxLayout()
        col.setSpacing(16)

        board_frame = QFrame()
        board_frame.setObjectName("board")
        board_grid = QGridLayout(board_frame)
        board_grid.setContentsMargins(3, 3, 3, 3)
        board_grid.setSpacing(3)

        for box_index in range(9):
            box = QFrame()
            box.setObjectName("box")
            box_grid = QGridLayout(box)
            box_grid.setContentsMargins(1, 1, 1, 1)
            box_grid.setSpacing(1)
            box_row, box_col = divmod(box_index, 3)
            for i in range(9):
                r = box_row * 3 + i // 3
                c = box_col * 3 + i % 3
                cell = Cell(r, c, self.select_cell)
                self.cells[r][c] = cell
                box_grid.addWidget(cell, i // 3, i % 3)
            board_grid.addWidget(box, box_row, box_col)

        board_frame.setMinimumSize(360, 360)
        col.addWidget(board_frame, stretch=1)

        legend = QHBoxLayout()
        for text in ("● Inicial", "● Resuelto", "● Probando", "● Conflicto"):
            lbl = QLabel(text)
            lbl.setProperty("role", "muted")
            legend.addWidget(lbl)
        legend.addStretch()
        col.addLayout(legend)

        numpad = QGridLayout()
        numpad.setSpacing(6)
        for n in range(1, 10):
            btn = QPushButton(str(n))
            btn.setObjectName("numBtn")
            btn.setMinimumHeight(40)
            btn.clicked.connect(lambda _, n=n: self.input_number(n))
            numpad.addWidget(btn, 0, n - 1)
        clear_btn = QPushButton("⌫")
        clear_btn.setObjectName("numBtn")
        clear_btn.setMinimumHeight(40)
        clear_btn.clicked.connect(lambda: self.input_number(0))
        numpad.addWidget(clear_btn, 0, 9)
        col.addLayout(numpad)

        return col

    def _card(self, title):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setProperty("role", "title")
        layout.addWidget(heading)
        return card, layout

    def _build_control_panel(self):
        panel = QWidget()
        panel.setFixedWidth(300)
        col = QVBoxLayout(panel)
        col.setSpacing(16)
        col.setContentsMargins(0, 0, 0, 0)

        # -- puzzle card
        card, layout = self._card("Puzzle")
        field_lbl = QLabel("Dificultad")
        field_lbl.setProperty("role", "field")
        layout.addWidget(field_lbl)
        self.difficulty_box = QComboBox()
        self.difficulty_box.addItems(list(DIFFICULTIES.keys()))
        self.difficulty_box.setCurrentText("Medio")
        layout.addWidget(self.difficulty_box)
        new_btn = QPushButton("🎲 Nuevo ejemplo")
        new_btn.setProperty("cls", "secondary")
        new_btn.clicked.connect(lambda: self.new_puzzle(self.difficulty_box.currentText()))
        layout.addWidget(new_btn)
        col.addWidget(card)

        # -- solve card
        card, layout = self._card("Resolver")
        speed_lbl = QLabel("Velocidad de animación")
        speed_lbl.setProperty("role", "field")
        layout.addWidget(speed_lbl)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 2)
        self.speed_slider.setValue(1)
        layout.addWidget(self.speed_slider)
        speed_labels_row = QHBoxLayout()
        for text in self.SPEED_LABELS:
            lbl = QLabel(text)
            lbl.setProperty("role", "muted")
            speed_labels_row.addWidget(lbl)
        layout.addLayout(speed_labels_row)

        btn_row = QHBoxLayout()
        self.solve_btn = QPushButton("▶ Resolver")
        self.solve_btn.setProperty("cls", "primary")
        self.solve_btn.clicked.connect(self.start_solve)
        btn_row.addWidget(self.solve_btn)
        self.pause_btn = QPushButton("⏸ Pausar")
        self.pause_btn.setProperty("cls", "secondary")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        btn_row.addWidget(self.pause_btn)
        layout.addLayout(btn_row)

        self.reset_btn = QPushButton("↺ Limpiar tablero")
        self.reset_btn.setProperty("cls", "ghost")
        self.reset_btn.clicked.connect(self.reset_board)
        layout.addWidget(self.reset_btn)

        self.ai_btn = QPushButton("🧠 Resolver con IA (red neuronal)")
        self.ai_btn.setProperty("cls", "secondary")
        self.ai_btn.clicked.connect(self.solve_with_ai)
        layout.addWidget(self.ai_btn)
        col.addWidget(card)

        # -- metrics card
        card, layout = self._card("Rendimiento")
        metrics_row = QHBoxLayout()
        time_col = QVBoxLayout()
        time_field = QLabel("Tiempo")
        time_field.setProperty("role", "field")
        self.time_label = QLabel("0.00s")
        self.time_label.setProperty("role", "metric")
        time_col.addWidget(time_field)
        time_col.addWidget(self.time_label)
        steps_col = QVBoxLayout()
        steps_field = QLabel("Pasos")
        steps_field.setProperty("role", "field")
        self.steps_label = QLabel("0")
        self.steps_label.setProperty("role", "metric")
        steps_col.addWidget(steps_field)
        steps_col.addWidget(self.steps_label)
        metrics_row.addLayout(time_col)
        metrics_row.addLayout(steps_col)
        layout.addLayout(metrics_row)
        col.addWidget(card)

        self.status_label = QLabel("")
        self.status_label.setProperty("role", "status")
        self.status_label.setWordWrap(True)
        col.addWidget(self.status_label)
        col.addStretch()

        for w in (self.solve_btn, self.pause_btn, self.reset_btn, self.ai_btn, new_btn):
            repolish(w)

        return panel

    # ------------------------------------------------------------ Theme --

    def toggle_theme(self):
        self.apply_theme("light" if self.theme_name == "dark" else "dark")

    def apply_theme(self, name):
        self.theme_name = name
        theme = DARK if name == "dark" else LIGHT
        self.setStyleSheet(build_stylesheet(theme))
        self.theme_btn.setText("☾" if name == "dark" else "☀")

    # ------------------------------------------------------------ Board --

    def new_puzzle(self, difficulty):
        self.stop_solve()
        self.board = make_puzzle(difficulty)
        self.given = [[self.board[r][c] != 0 for c in range(9)] for r in range(9)]
        self.solved_by_algo = [[False] * 9 for _ in range(9)]
        self.selected = None
        self.steps = 0
        self.time_label.setText("0.00s")
        self.steps_label.setText("0")
        self.status_label.setText(f"Nuevo puzzle ({difficulty}).")
        self.render_board()

    def reset_board(self):
        self.stop_solve()
        self.board = [[0] * 9 for _ in range(9)]
        self.given = [[False] * 9 for _ in range(9)]
        self.solved_by_algo = [[False] * 9 for _ in range(9)]
        self.selected = None
        self.steps = 0
        self.time_label.setText("0.00s")
        self.steps_label.setText("0")
        self.status_label.setText("Tablero limpio.")
        self.render_board()

    def render_board(self):
        for r in range(9):
            for c in range(9):
                cell = self.cells[r][c]
                val = self.board[r][c]
                cell.setText(str(val) if val else "")
                cell.set_flags(given=self.given[r][c], solved=self.solved_by_algo[r][c],
                                trying=False, error=False)
        self.update_selection_visuals()

    def select_cell(self, row, col):
        if self.solving:
            return
        self.selected = (row, col)
        self.update_selection_visuals()

    def update_selection_visuals(self):
        selected_value = self.board[self.selected[0]][self.selected[1]] if self.selected else 0
        for r in range(9):
            for c in range(9):
                is_selected = self.selected == (r, c)
                same_value = selected_value != 0 and self.board[r][c] == selected_value and not is_selected
                self.cells[r][c].set_flags(selected=is_selected, sameValue=same_value)

    def input_number(self, num):
        if self.solving or not self.selected:
            return
        row, col = self.selected
        if self.given[row][col]:
            return
        self.board[row][col] = num
        self.solved_by_algo[row][col] = False
        cell = self.cells[row][col]
        cell.setText(str(num) if num else "")
        cell.set_flags(solved=False)
        if num and has_conflict(self.board, row, col, num):
            cell.set_flags(error=True)
            QTimer.singleShot(400, lambda: cell.set_flags(error=False))
        self.update_selection_visuals()

    def keyPressEvent(self, event):
        if self.solving or not self.selected:
            return super().keyPressEvent(event)
        row, col = self.selected
        key = event.key()
        if Qt.Key_1 <= key <= Qt.Key_9:
            self.input_number(key - Qt.Key_0)
        elif key in (Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_0):
            self.input_number(0)
        elif key == Qt.Key_Up:
            self.select_cell(max(0, row - 1), col)
        elif key == Qt.Key_Down:
            self.select_cell(min(8, row + 1), col)
        elif key == Qt.Key_Left:
            self.select_cell(row, max(0, col - 1))
        elif key == Qt.Key_Right:
            self.select_cell(row, min(8, col + 1))
        else:
            return super().keyPressEvent(event)

    # ------------------------------------------------------------ Solve --

    def start_solve(self):
        if self.solving:
            return
        for r in range(9):
            for c in range(9):
                if self.board[r][c] and has_conflict(self.board, r, c, self.board[r][c]):
                    self.status_label.setText("Hay números en conflicto — corrígelos antes de resolver.")
                    return
        self.solving = True
        self.paused = False
        self.steps = 0
        self.timer.start()
        self.solve_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("⏸ Pausar")
        self.difficulty_box.setEnabled(False)
        self.status_label.setText("Resolviendo…")

        self.working_board = [row[:] for row in self.board]
        self.generator = backtrack_solve(self.working_board)

        speed = self.speed_slider.value()
        if speed == 2:  # instant: run to completion, no per-step animation
            solved = False
            try:
                while True:
                    ev = next(self.generator)
                    if ev[0] == 'try':
                        self.steps += 1
            except StopIteration as stop:
                solved = bool(stop.value)
            self._finish_solve(solved)
        else:
            self.metrics_timer.start(100)
            self.step_timer.start(self.SPEED_DELAYS[speed])

    def _solve_step(self):
        if self.paused:
            return
        try:
            ev_type, r, c, num = next(self.generator)
        except StopIteration as stop:
            self.step_timer.stop()
            self.metrics_timer.stop()
            self._finish_solve(bool(stop.value))
            return

        cell = self.cells[r][c]
        if ev_type == 'try':
            self.steps += 1
            cell.setText(str(num))
            cell.set_flags(trying=True)
        elif ev_type == 'set':
            cell.setText(str(num))
            cell.set_flags(trying=False, solved=True)
        elif ev_type == 'unset':
            cell.setText("")
            cell.set_flags(trying=False, solved=False)

    def _finish_solve(self, solved):
        self.solving = False
        self.solve_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.difficulty_box.setEnabled(True)
        if solved:
            self.board = self.working_board
            for r in range(9):
                for c in range(9):
                    if not self.given[r][c] and self.board[r][c] != 0:
                        self.solved_by_algo[r][c] = True
            self.render_board()
            self.status_label.setText("¡Resuelto!")
        else:
            self.status_label.setText("No se encontró solución (revisa los números ingresados).")
        self._update_metrics()

    def solve_with_ai(self):
        if self.solving:
            return
        self.status_label.setText("Resolviendo con red neuronal…")
        QApplication.processEvents()
        try:
            solved = solve_with_nn(self.board)
        except Exception as exc:  # missing model files, tensorflow not installed, etc.
            self.status_label.setText(f"IA no disponible: {exc}")
            return
        self.board = solved
        for r in range(9):
            for c in range(9):
                if not self.given[r][c]:
                    self.solved_by_algo[r][c] = True
        self.render_board()
        self.status_label.setText("¡Resuelto con IA!")

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.setText("▶ Reanudar" if self.paused else "⏸ Pausar")

    def stop_solve(self):
        self.step_timer.stop()
        self.metrics_timer.stop()
        self.solving = False
        self.paused = False
        self.solve_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.difficulty_box.setEnabled(True)

    def _update_metrics(self):
        self.time_label.setText(f"{self.timer.elapsed() / 1000:.2f}s")
        self.steps_label.setText(str(self.steps))


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = SudokuWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
