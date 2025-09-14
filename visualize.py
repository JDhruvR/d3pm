import chess
import chess.svg
from PIL import Image
from io import BytesIO
import torch
import os

from chess_utils import tensor_to_fen, fen_to_tensor

def display_board_from_tensor(board_tensor: torch.Tensor, size=400, save_path=None, show_image=False):
    """
    Generates a visual representation of the board from a tensor.

    Args:
        board_tensor (torch.Tensor): The 1D tensor of shape (64,) representing the board.
        size (int): The size of the output image in pixels.
        save_path (str, optional): Path to save the image file (e.g., 'board.png').
                                   If None, the image is not saved. Defaults to None.
        show_image (bool, optional): If True, attempts to open the image in the default
                                     system viewer. Defaults to False.
    """
    # Convert the tensor to a FEN string. We use default values for game state
    # as they don't affect the visual piece placement.
    fen = tensor_to_fen(board_tensor)

    # Create a board object from the FEN
    board = chess.Board(fen)

    # Generate an SVG image of the board
    svg_data = chess.svg.board(board=board, size=size)

    # Convert SVG to a PNG and handle it
    try:
        from cairosvg import svg2png
        png_data = svg2png(bytestring=svg_data.encode('utf-8'))
        img = Image.open(BytesIO(png_data))

        if save_path:
            # Ensure the directory exists before saving
            output_dir = os.path.dirname(save_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            img.save(save_path)
            print(f"Board image saved to: {save_path}")

        if show_image:
            img.show() # This will open the image in your default image viewer

        return img
    except ImportError:
        print("CairoSVG not found. Cannot display or save image.")
        print("Board FEN:", fen)
        # The SVG data can still be useful for debugging
        if save_path and save_path.endswith(".svg"):
             with open(save_path, "w") as f:
                f.write(svg_data)
             print(f"Saved board as SVG to: {save_path}")
        else:
            print("Board SVG data:\n", svg_data)
        return None

if __name__ == "__main__":

    fen_str = "r2q1Bk1/p3pp1p/1pnp2p1/2p5/P3P3/1P1P1N2/2P2KnP/R2Q3R b - - 0 14"

    # 1. Parse the original FEN to get both the board and the game state
    parts = fen_str.split(' ')
    piece_placement = parts[0]
    active_color = parts[1]
    castling = parts[2]
    en_passant = parts[3]
    halfmove_clock = int(parts[4])
    fullmove_number = int(parts[5])

    # 2. Convert the piece placement to our tensor
    board_tensor = fen_to_tensor(fen_str)

    # 3. Reconstruct the FEN using the *original* game state values
    reconstructed_fen = tensor_to_fen(
        board_tensor,
        active_color=active_color,
        castling=castling,
        en_passant=en_passant,
        halfmove_clock=halfmove_clock,
        fullmove_number=fullmove_number
    )

    # 4. Validate that the full FEN string is now identical
    assert fen_str == reconstructed_fen

    # 5. Visualize the board from the tensor to confirm it's correct
    display_board_from_tensor(board_tensor, save_path='sample_image.png', show_image=False)

    print("SUCCESS: Full FEN round-trip validation passed.")
