import chess
import chess.engine
import torch

chyler = chess.engine.SimpleEngine.popen_uci("./chyler")
stockfish = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")

inputs = []
scores = []

games = [
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
         "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
         "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
         "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
         "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
         ]
for game in games:
    board = chess.Board(game)
    while(not board.is_game_over()): # need to handle mate scores
        info = stockfish.analyse(board, chess.engine.Limit(depth=14))
        score = info["score"].white().score(mate_score=100000) / 100

        castleRights = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castleRights |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castleRights |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            castleRights |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            castleRights |= 8

        EnPessant = 0
        if board.ep_square is not None:
            EnPessant = 1 << board.ep_square

        count = 0
        features = torch.zeros(837)
        bitboards = []
        bitboards.append(int(board.pieces(chess.PAWN, chess.WHITE)))
        bitboards.append(int(board.pieces(chess.KNIGHT, chess.WHITE)))
        bitboards.append(int(board.pieces(chess.BISHOP, chess.WHITE)))
        bitboards.append(int(board.pieces(chess.ROOK, chess.WHITE)))
        bitboards.append(int(board.pieces(chess.QUEEN, chess.WHITE)))
        bitboards.append(int(board.pieces(chess.KING, chess.WHITE)))
        bitboards.append(int(board.pieces(chess.PAWN, chess.BLACK)))
        bitboards.append(int(board.pieces(chess.KNIGHT, chess.BLACK)))
        bitboards.append(int(board.pieces(chess.BISHOP, chess.BLACK)))
        bitboards.append(int(board.pieces(chess.ROOK, chess.BLACK)))
        bitboards.append(int(board.pieces(chess.QUEEN, chess.BLACK)))
        bitboards.append(int(board.pieces(chess.KING, chess.BLACK)))

        for bb in range(12):
            for i in range(64):
                features[bb * 64 + i] = (bitboards[bb] >> i) & 1

        features[768] = not board.turn
        for right in range(4):
            features[769 + right] = (castleRights >> right) & 1

        for sq in range(64):
            features[773 + sq] = (EnPessant >> sq) & 1

        inputs.append(features)
        scores.append(score)

        move = chyler.play(board, chess.engine.Limit(depth=5))
        board.push(move.move)

inputs = torch.stack(inputs)
scores = torch.tensor(scores)

data = {
"inputs": inputs,
"evals": scores
}

torch.save(data, "trainingData.pt")

chyler.quit()
stockfish.quit()