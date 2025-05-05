

import cv2
import numpy as np
import os

def extract_board(image_path):
    """Estrae la scacchiera dall'immagine."""
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Impossibile leggere l'immagine: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    board_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(board_contour)
    board = img[y:y+h, x:x+w]
    
    return board

def split_board_into_squares(board_img):
    """Divide l'immagine della scacchiera in 64 caselle."""
    height, width = board_img.shape[:2]
    square_height = height // 8
    square_width = width // 8
    
    squares = []
    for row in range(8):
        row_squares = []
        for col in range(8):
            y_start = row * square_height
            y_end = (row + 1) * square_height
            x_start = col * square_width
            x_end = (col + 1) * square_width
            square = board_img[y_start:y_end, x_start:x_end]
            row_squares.append(square)
        squares.append(row_squares)
    
    return squares

def create_piece_templates_from_reference(reference_image_path):
    """Crea template dei pezzi da un'immagine di riferimento della posizione iniziale."""
    ref_board = extract_board(reference_image_path)
    ref_squares = split_board_into_squares(ref_board)
    
    initial_position = [
        ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
        ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
        ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    ]
    
    piece_templates = {}
    empty_templates = {'light': None, 'dark': None}
    
    for row_idx, row in enumerate(initial_position):
        for col_idx, piece in enumerate(row):
            square = ref_squares[row_idx][col_idx]
            square_resized = cv2.resize(square, (50, 50))
            if piece != '.':
                piece_templates[piece] = square_resized
            else:
                gray = cv2.cvtColor(square_resized, cv2.COLOR_BGR2GRAY) if len(square_resized.shape) == 3 else square_resized
                equalized = cv2.equalizeHist(gray)
                brightness = np.mean(equalized)
                if brightness > 150:
                    if empty_templates['light'] is None:
                        empty_templates['light'] = square_resized
                else:
                    if empty_templates['dark'] is None:
                        empty_templates['dark'] = square_resized
    
    return piece_templates, empty_templates

def enhance_contrast(image):
    """Applica la correzione gamma per migliorare il contrasto."""
    gamma = 1.2
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, look_up_table)

def match_piece_with_templates(square_img, piece_templates, empty_templates):
    """Confronta una casella con i template per identificare il pezzo."""
    square_resized = cv2.resize(square_img, (50, 50))
    square_gray = cv2.cvtColor(square_resized, cv2.COLOR_BGR2GRAY) if len(square_resized.shape) == 3 else square_resized
    square_gray = enhance_contrast(square_gray)
    square_gray = cv2.equalizeHist(square_gray)
    
    is_empty = False
    for empty_type, template in empty_templates.items():
        if template is not None:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
            result = cv2.matchTemplate(square_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            if np.max(result) > 0.8:
                is_empty = True
                break
    
    if is_empty:
        return '.'
    
    best_match = None
    highest_score = -float('inf')
    for piece_name, template in piece_templates.items():
        if template is not None:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
            result = cv2.matchTemplate(square_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            score = np.max(result)
            if score > highest_score:
                highest_score = score
                best_match = piece_name
    
    return best_match if highest_score >= 0.4 else '.'

def analyze_board_with_templates(squares, piece_templates, empty_templates):
    """Analizza la scacchiera utilizzando i template dei pezzi."""
    board_state = []
    for row in squares:
        row_state = [match_piece_with_templates(square, piece_templates, empty_templates) for square in row]
        board_state.append(row_state)
    return board_state

def alternative_piece_detection(squares):
    """Metodo alternativo per rilevare i pezzi quando i template non funzionano bene."""
    board_state = []
    for row in squares:
        row_state = []
        for square in row:
            resized = cv2.resize(square, (50, 50))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
            equalized = cv2.equalizeHist(gray)
            mean_val = np.mean(equalized)
            std_val = np.std(equalized)
            if std_val > 50:
                if mean_val < 100:
                    row_state.append('p')
                elif mean_val > 150:
                    row_state.append('P')
                else:
                    row_state.append('.')
            else:
                row_state.append('.')
        board_state.append(row_state)
    return board_state

def generate_fen(board_state):
    """Genera la notazione FEN dalla matrice del stato della scacchiera."""
    fen_parts = []
    for row in board_state:
        empty_count = 0
        fen_row = ""
        for cell in row:
            if cell == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_parts.append(fen_row)
    fen = '/'.join(fen_parts)
    fen += " w KQkq - 0 1"
    return fen

def analyze_chessboard(image_path, reference_image_path=None):
    """Analizza una scacchiera da un'immagine e genera la notazione FEN."""
    try:
        board_img = extract_board(image_path)
        squares = split_board_into_squares(board_img)
        if reference_image_path:
            piece_templates, empty_templates = create_piece_templates_from_reference(reference_image_path)
            board_state = analyze_board_with_templates(squares, piece_templates, empty_templates)
        else:
            board_state = alternative_piece_detection(squares)
        fen = generate_fen(board_state)
        return fen, board_state
    except Exception as e:
        print(f"Errore nell'analisi della scacchiera: {e}")
        return None, None

def visualize_board_state(board_state):
    """Visualizza lo stato della scacchiera in forma testuale."""
    print("  a b c d e f g h")
    print(" +-----------------+")
    for i, row in enumerate(board_state):
        print(f"{8-i}| {' '.join([piece if piece != '.' else '.' for piece in row])} |")
    print(" +-----------------+")
    print("  a b c d e f g h")

    
def draw_board_with_pieces(board_img, board_state):
    """Disegna i nomi dei pezzi sulla scacchiera analizzata."""
    height, width = board_img.shape[:2]
    square_height = height // 8
    square_width = width // 8

    board_with_pieces = board_img.copy()

    for row_idx, row in enumerate(board_state):
        for col_idx, piece in enumerate(row):
            if piece != '.':  
                x = col_idx * square_width
                y = row_idx * square_height
                
                text_x = x + square_width // 4
                text_y = y + square_height // 2
                cv2.putText(
                    board_with_pieces, piece, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
                )

    # Mostra l'immagine con i pezzi
    cv2.imshow("Scacchiera Analizzata", board_with_pieces)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = "Foto4.jpg"
    reference_image_path = "ScacchieraCompleta1.jpg"
    if not os.path.exists(image_path):
        print(f"ERRORE: L'immagine {image_path} non esiste.")
        return
    if reference_image_path and not os.path.exists(reference_image_path):
        print(f"ERRORE: L'immagine di riferimento {reference_image_path} non esiste.")
        reference_image_path = None
    fen, board_state = analyze_chessboard(image_path, reference_image_path)
    if fen:
        print(f"\nNotazione FEN: {fen}")
        print("\nStato della scacchiera rilevato:")
        visualize_board_state(board_state)


        board_img = extract_board(image_path)
        draw_board_with_pieces(board_img, board_state)
    else:
        print("Impossibile generare la notazione FEN.")

if __name__ == "__main__":
    main()