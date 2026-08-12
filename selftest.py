# Run with: python selftest.py
from app import DIFFICULTIES, create_full_board, make_puzzle, has_conflict, backtrack_solve


def is_valid_full_board(b):
    for i in range(9):
        if len(set(b[i])) != 9 or len(set(row[i] for row in b)) != 9:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = {b[br + i][bc + j] for i in range(3) for j in range(3)}
            if len(box) != 9:
                return False
    return True


assert is_valid_full_board(create_full_board()), "generated full board must be a valid solved sudoku"

for difficulty, expected_empty in DIFFICULTIES.items():
    puzzle = make_puzzle(difficulty)
    empties = sum(row.count(0) for row in puzzle)
    assert empties == expected_empty, f"{difficulty} puzzle should have {expected_empty} empty cells"

    board = [row[:] for row in puzzle]
    gen = backtrack_solve(board)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        solved = bool(stop.value)
    assert solved, f"{difficulty} puzzle should be solvable"
    assert is_valid_full_board(board), f"{difficulty} solved board must be valid"

conflict_board = [[5, 0, 0, 0, 0, 0, 0, 0, 0]] + [[0] * 9 for _ in range(8)]
assert has_conflict(conflict_board, 0, 3, 5) is True
assert has_conflict([[0] * 9 for _ in range(9)], 0, 0, 5) is False

print("selftest ok")
