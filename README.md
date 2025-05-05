The main goal of this project is to create a bot that allows users to interact with Stockfish in a simple and intuitive way—just by clicking a script.
There is no intention of promoting cheating or misuse; this project is solely for academic purposes.

How the scripts work:

First, the user should arrange the screen so that the script can properly recognize the chessboard. To do this, run rilevaScacchiera.py.

Once the chessboard has been correctly detected, the user can run testStockfish.py.

This second script will not only recognize the position of the chess pieces on the board, but also provide additional useful information, such as:

Example output:

FEN notation: 2k5/p1qn4/Ppp1p3/7p/2Q1p3/1P4r1/1BP3P1/R5K1 w KQkq - 0 1

Stockfish analysis:
Best move: c4e6  
Score: PovScore(Cp(+545), WHITE) (centipawn)  
Depth: 28 plies
As shown above, the script returns the FEN (Forsyth–Edwards Notation), the best move suggested by Stockfish, the evaluation score, and the search depth reached by the engine.

The user can also customize the amount of time given to Stockfish by modifying the parameters of the analyze_with_stockfish() function in testStockfish.py.

As stated earlier, this project is developed solely for academic purposes, specifically for the Computer Vision course.

AUTHORS: Raffaele Cammi, Michele Di Frisco Ramirez
Date: March–April 2025
