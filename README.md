# Sudoku Solver with AI Integration

## Overview
This project is a **Sudoku solver** built in Python, combining traditional solving techniques and **Machine Learning**. The application offers an interactive Sudoku game, along with an AI-driven solution using a **Neural Network** model trained on large datasets.

Using **Keras**, **PyQt5**, and **NumPy**, this project aims to provide both a fun user experience for playing Sudoku and a cutting-edge AI solution that can solve puzzles automatically.

The AI model is trained on **9 million Sudoku puzzles** from Kaggle: [Sudoku Dataset](https://www.kaggle.com/datasets/rohanrao/sudoku/data). The model is designed to predict the correct numbers for any incomplete Sudoku puzzle.

## Features
- **Interactive Sudoku Game**: Play Sudoku with a graphical interface using **PyQt5**.
- **Manual Solution**: The program can solve a Sudoku puzzle step-by-step using a **backtracking algorithm**.
- **AI Solution with Neural Networks**: Solves the puzzle using a **trained neural network model**.
- **Difficulty Levels**: Choose between Easy, Medium, or Hard levels.
  
## Modes
The app is a single unified board: pick a difficulty, play manually, or solve it — no separate screens to navigate.

1. **Manual Play**:
   - Fill in cells yourself using the keyboard or the on-screen numpad.
   - Conflicting entries flash red immediately.

2. **Backtracking Solve** (with visualization):
   - Watch the recursive backtracking algorithm work cell by cell, with an adjustable animation speed (slow / normal / instant) and a live step counter + timer.

3. **AI Solve**:
   - Solves using a pre-trained **neural network** model, loaded from `model_structure.json` and `best_weights.hdf5` via **Keras**.

## Installation & Setup

### Requirements
Ensure you have Python 3.x installed. Then, install the necessary libraries by running the following command:

```bash
pip install -r requirements.txt
```

### Setup Instructions:
Clone or copy the project repository to your local machine. <br/>
Install dependencies with the command above. <br/>
Make sure you have the following files available: <br/>
model_structure.json: Contains the architecture of the trained model. <br/>
best_weights.hdf5: Contains the trained weights for the neural network.  <br/>

### Running the Application:
Run the main application by executing the following command:

```bash
python app.py
```

The game window will appear, and you can choose between different modes from the interface.

### How It Works
Neural Network Mode: <br/>
The AI uses a Convolutional Neural Network (CNN) model trained on millions of Sudoku puzzles. It predicts the missing numbers in the puzzle by processing the grid as an image. The model is based on the architecture in model_structure.json and weights stored in best_weights.hdf5. <br/>

The input to the model is a 9x9 grid where missing values are represented as 0s. <br/>
The model predicts the most probable values for the empty cells. <br/>
The output is a fully solved puzzle represented as a string. <br/>
Example Input (Puzzle with missing values):<br/>
```bash
  0 0 0 7 0 0 0 9 6
  0 0 3 0 6 9 1 7 8 
  0 0 7 2 0 0 5 0 0
  0 7 5 0 0 0 0 0 0
  9 0 1 0 0 0 3 0 0
  0 0 0 0 0 0 0 0 0
  0 0 9 0 0 0 0 0 1
  3 1 8 0 2 0 4 0 7
  2 4 0 0 0 5 0 0 0
```
Example Output (Solved Puzzle):<br/>
```bash
"184753296523469178697281543875312964961547382432698715759834621318926457246175839"
```

## Backtracking Solution Mode:
The backtracking algorithm is a classic technique for solving Sudoku puzzles. It starts by filling cells with numbers from 1-9, checking whether the number is valid for the current row, column, and 3x3 grid. If the number is valid, it proceeds to the next cell. If it reaches a point where no valid numbers exist, it backtracks and tries a different number.

## Sudoku Grid Generation:
The program generates Sudoku puzzles using a randomized Sudoku generator that creates a valid, fully-filled grid and removes numbers based on the selected difficulty level (Easy, Medium, or Hard).

## User Interface (GUI)
The graphical interface is built using PyQt5, featuring:<br/>

A 9x9 grid with clearly differentiated 3x3 boxes and a dark/light theme toggle.<br/>
A control panel with difficulty selection, solve speed, pause/resume, reset, and live time/step metrics.<br/>
Visual states for given, solved, currently-tried, and conflicting cells.<br/>

## Controls:<br/>
Click a cell (or use arrow keys) to select it.<br/>
Enter numbers with the keyboard (1-9, Backspace/Delete) or the on-screen numpad.<br/>
Press **Resolver** to watch the backtracking algorithm solve it, or **Resolver con IA** for the neural network solution.<br/>

