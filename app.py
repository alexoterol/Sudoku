import random
import numpy as np
import sys
import random

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QComboBox, QVBoxLayout, QWidget, QGridLayout, QInputDialog
from PyQt5.QtCore import QTimer
from keras.models import model_from_json

class SolucionMachineLearning:
    def __init__(self, tablero) -> None:
        self.tablero = tablero.tablero  # Obtener el tablero del objeto Tablero
        # Cargar el modelo de Keras
        with open('model_structure.json', 'r') as json_file:
            model_json = json_file.read()
        self.model = model_from_json(model_json)
        self.model.load_weights('best_weights.hdf5')
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def solucionar(self):
        puzzle = self.format_sudoku()
        puzzle = puzzle.replace('\n', '').replace(' ', '')
        initial_board = np.array([int(j) for j in puzzle]).reshape((9, 9, 1))
        initial_board = (initial_board / 9) - 0.5

        while True:
            predictions = self.model.predict(initial_board.reshape((1, 9, 9, 1))).squeeze()
            pred = np.argmax(predictions, axis=1).reshape((9, 9)) + 1
            prob = np.around(np.max(predictions, axis=1).reshape((9, 9)), 2)

            initial_board = ((initial_board + 0.5) * 9).reshape((9, 9))
            mask = (initial_board == 0)

            if mask.sum() == 0:
                # Puzzle is solved
                break

            prob_new = prob * mask

            ind = np.argmax(prob_new)
            x, y = (ind // 9), (ind % 9)

            val = pred[x][y]
            initial_board[x][y] = val
            initial_board = (initial_board / 9) - 0.5

        solved_puzzle = ''.join(map(str, initial_board.flatten().astype(int)))
        self.tablero = self.string_to_matrix_9x9(solved_puzzle)

    def print_sudoku(self):
        for row in self.tablero:
            print(" ".join(str(cell) if cell != 0 else "." for cell in row))
    
    def format_sudoku(self):
        formatted_sudoku = ""
        for row in self.tablero:
            formatted_row = " ".join(str(cell) for cell in row)
            formatted_sudoku += formatted_row + "\n"
        return formatted_sudoku
    
    def string_to_matrix_9x9(self, sudoku_string):
        if len(sudoku_string) != 81:
            raise ValueError("La cadena debe tener exactamente 81 caracteres.")
        
        # Crear la matriz 9x9 dividiendo la cadena en partes de 9 caracteres
        matrix = []
        for i in range(0, 81, 9):
            row = [int(char) for char in sudoku_string[i:i+9]]  # Convertir cada bloque de 9 caracteres en una fila
            matrix.append(row)
        return matrix

class SolucionManual:
    def __init__(self, tablero):
        self.tablero = tablero.tablero

    def solucionar(self):
        self.backtrack(0, 0)

    def backtrack(self, row, col):
        if row == 9:
            return True
        if col == 9:
            return self.backtrack(row + 1, 0)
        if self.tablero[row][col] != 0:
            return self.backtrack(row, col + 1)

        for num in range(1, 10):
            if self.is_valid(row, col, num):
                self.tablero[row][col] = num
                if self.backtrack(row, col + 1):
                    return True
                self.tablero[row][col] = 0

        return False

    def is_valid(self, row, col, num):
        for i in range(9):
            if self.tablero[row][i] == num or self.tablero[i][col] == num:
                return False

        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if self.tablero[box_row + i][box_col + j] == num:
                    return False
        return True

class Tablero:
    def __init__(self, dificultad) -> None:
        self.tablero = self.create_sudoku()  # Crear el tablero completo
        self.tablero = self.remove_numbers(dificultad)  # Quitar números según la dificultad

    def create_sudoku(self):
        base = 3
        side = base * base

        def pattern(r, c):
            return (base * (r % base) + r // base + c) % side

        def shuffle(s):
            return random.sample(s, len(s))

        rBase = range(base)
        rows = [g * base + r for g in shuffle(rBase) for r in shuffle(rBase)]
        cols = [g * base + c for g in shuffle(rBase) for c in shuffle(rBase)]
        nums = shuffle(range(1, base * base + 1))

        board = [[nums[pattern(r, c)] for c in cols] for r in rows]
        return board

    def remove_numbers(self, difficulty):
        if difficulty == 'easy':
            empty_cells = 40
        elif difficulty == 'medium':
            empty_cells = 50
        elif difficulty == 'hard':
            empty_cells = 60
        else:
            raise ValueError("Invalid difficulty level")

        cells_removed = 0
        while cells_removed < empty_cells:
            row = random.randint(0, 8)
            col = random.randint(0, 8)
            if self.tablero[row][col] != 0:
                self.tablero[row][col] = 0
                cells_removed += 1

        return self.tablero

class SudokuGame(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sudoku Game")
        self.setGeometry(100, 100, 600, 400)
        self.initUI()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.time_elapsed = 0

    def initUI(self):
        widget = QWidget(self)
        layout = QVBoxLayout()

        self.difficulty_label = QLabel("Select Difficulty:")
        self.difficulty_select = QComboBox()
        self.difficulty_select.addItems(["Easy", "Medium", "Hard"])

        layout.addWidget(self.difficulty_label)
        layout.addWidget(self.difficulty_select)

        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.start_play_mode)
        layout.addWidget(self.play_button)

        self.auto_button = QPushButton("AutoComplete", self)
        self.auto_button.clicked.connect(self.start_autocomplete_mode)
        layout.addWidget(self.auto_button)

        self.nn_button = QPushButton("Solucion with NN", self)
        self.nn_button.clicked.connect(self.start_nn_mode)
        layout.addWidget(self.nn_button)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def start_play_mode(self):
        """Start Play mode: user solves the Sudoku."""
        self.clear_screen()
        self.setWindowTitle("Sudoku - Play Mode")

        widget = QWidget(self)
        layout = QVBoxLayout()

        self.grid_layout = QGridLayout()
        self.grid_buttons = []

        for row in range(9):
            button_row = []
            for col in range(9):
                button = QPushButton(self)
                button.setFixedSize(50, 50)
                button.clicked.connect(lambda ch, r=row, c=col: self.cell_clicked(r, c))
                self.grid_layout.addWidget(button, row, col)
                button_row.append(button)
            self.grid_buttons.append(button_row)

        layout.addLayout(self.grid_layout)

        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.tablero = Tablero(self.difficulty_select.currentText().lower())
        self.update_sudoku_grid()

        self.time_elapsed = 0
        self.timer.start(1000)

    def start_autocomplete_mode(self):
        """Start AutoComplete mode: use SolucionManual."""
        self.clear_screen()
        self.setWindowTitle("Sudoku - AutoComplete Mode")

        widget = QWidget(self)
        layout = QVBoxLayout()

        self.grid_layout = QGridLayout()
        self.grid_buttons = []

        for row in range(9):
            button_row = []
            for col in range(9):
                button = QPushButton(self)
                button.setFixedSize(50, 50)
                button.setEnabled(False)
                self.grid_layout.addWidget(button, row, col)
                button_row.append(button)
            self.grid_buttons.append(button_row)

        layout.addLayout(self.grid_layout)

        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.tablero = Tablero(self.difficulty_select.currentText().lower())
        self.solver = SolucionManual(self.tablero)
        self.solver.solucionar()

        self.update_sudoku_grid()
        self.time_elapsed = 0
        self.timer.start(1000)

    def start_nn_mode(self):
        """Start Solucion with NN mode: use SolucionMachineLearning."""
        self.clear_screen()
        self.setWindowTitle("Sudoku - Solucion with NN Mode")

        widget = QWidget(self)
        layout = QVBoxLayout()

        self.grid_layout = QGridLayout()
        self.grid_buttons = []

        # Create the grid and disable buttons for NN solution
        for row in range(9):
            button_row = []
            for col in range(9):
                button = QPushButton(self)
                button.setFixedSize(50, 50)
                button.setEnabled(False)
                self.grid_layout.addWidget(button, row, col)
                button_row.append(button)
            self.grid_buttons.append(button_row)

        layout.addLayout(self.grid_layout)

        # Back button to return to the main menu
        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.time_elapsed = 0
        self.timer.start(1000)
        # Initialize the Sudoku puzzle
        self.tablero = Tablero(self.difficulty_select.currentText().lower())

        # Update the UI first, then solve the puzzle in a small delay
        QTimer.singleShot(1000, self.solve_with_nn)

    def solve_with_nn(self):
        """Solve Sudoku using neural network after UI is set up."""
        self.solver = SolucionMachineLearning(self.tablero)
        self.solver.solucionar()
        print(self.solver.tablero)
        self.tablero.tablero = self.solver.tablero
        # Update the grid with the solved puzzle
        self.update_sudoku_grid()

    def update_sudoku_grid(self):
        """Updates the Sudoku grid with the current puzzle or solution."""
        for row in range(9):
            for col in range(9):
                value = self.tablero.tablero[row][col]
                if value != 0:
                    self.grid_buttons[row][col].setText(str(value))
                else:
                    self.grid_buttons[row][col].setText("")


    def cell_clicked(self, row, col):
        num, ok = QInputDialog.getInt(self, f"Enter a number for cell ({row}, {col})", "Number (1-9):", 1, 1, 9)
        if ok:
            self.grid_buttons[row][col].setText(str(num))
            self.tablero.tablero[row][col] = num

    def clear_screen(self):
        central_widget = self.centralWidget()
        if central_widget is not None:
            central_widget.deleteLater()

    def go_back(self):
        """Return to the main screen and reset."""
        self.timer.stop()
        self.initUI()

    def update_timer(self):
        """Update the timer every second."""
        self.time_elapsed += 1
        print(f"Time Elapsed: {self.time_elapsed} seconds")


def main():
    app = QApplication(sys.argv)
    window = SudokuGame()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
