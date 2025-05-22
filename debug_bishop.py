from keisei.shogi import ShogiGame, Color, PieceType
from keisei.shogi.shogi_core_definitions import Piece
from keisei.shogi.shogi_rules_logic import generate_piece_potential_moves

def print_board(game):
    print(game.to_string())

def main():
    game = ShogiGame()
    
    # Print initial board
    print("Initial board:")
    print_board(game)
    
    # Get the bishop at (1,7) and its potential moves
    bishop = game.get_piece(1, 7)
    if bishop:
        print(f"\nBishop at (1,7): {bishop.type}, {bishop.color}")
        moves = generate_piece_potential_moves(game, bishop, 1, 7)
        print(f"Potential moves for bishop at (1,7): {moves}")
    else:
        print("No bishop at (1,7)")
    
    # Make the moves from the test
    print("\nMaking moves from test...")
    game.make_move((6, 4, 5, 4, False))
    game.make_move((2, 3, 3, 3, False))
    game.make_move((5, 4, 4, 4, False))
    
    print("\nBoard after moves:")
    print_board(game)
    
    # Now let's see if bishop has any moves
    bishop = game.get_piece(1, 7)
    if bishop:
        print(f"\nBishop at (1,7): {bishop.type}, {bishop.color}")
        moves = generate_piece_potential_moves(game, bishop, 1, 7)
        print(f"Potential moves for bishop at (1,7): {moves}")
        
        # Let's print what's at the squares surrounding the bishop
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = 1 + dr, 7 + dc
                if game.is_on_board(r, c):
                    piece = game.get_piece(r, c)
                    print(f"Piece at ({r}, {c}): {piece.type if piece else 'None'}, {piece.color if piece else 'None'}")
    else:
        print("No bishop at (1,7)")

if __name__ == "__main__":
    main()
