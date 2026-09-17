This is a neural network designed to replace my evaluation function within my chess engine.

Implementation:
The network is composed of 4 fully connected Layers. The activation function is GELU. The network takes in a set of bitboards one for each piece, a bitboard for the en pessant square, and castling rights, and outputs an evaluation score.
Training:
- Optimizer: Adam
- loss: Mean Squared Error
- Learning Rate: 0.001

Goal:
The goal of this project is to explore using neural networks as a substitute to hardcoded evaluation functions, and eventually replace my chess engine's evaluation with this network.

Status:
This is an unfinished project and has been put on pause while I learn more about machine learning. I intend to come back and finish the network at a future date.
