import cv2
import numpy as np
import os

def extract_board(image_path):
    """Estrae la scacchiera dall'immagine."""
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Impossibile leggere l'immagine: {image_path}")
    
    # Converti in scala di grigi per facilitare il rilevamento dei bordi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Applica sfocatura per ridurre il rumore
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Rileva i bordi
    edges = cv2.Canny(blurred, 50, 150)
    
    # Trova i contorni
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Trova il contorno più grande (presumibilmente la scacchiera)
    board_contour = max(contours, key=cv2.contourArea)
    
    # Trova il rettangolo che racchiude il contorno
    x, y, w, h = cv2.boundingRect(board_contour)
    
    # Estrai la scacchiera
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
            # Estrai la casella corrente
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
    # Carica l'immagine di riferimento
    ref_board = extract_board(reference_image_path)
    ref_squares = split_board_into_squares(ref_board)
    
    # Mappatura della posizione iniziale standard (secondo le coordinate della matrice)
    initial_position = [
        ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],  # Pezzi neri (riga superiore)
        ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],  # Pedoni neri
        ['.', '.', '.', '.', '.', '.', '.', '.'],  # Casella vuota chiara/scura
        ['.', '.', '.', '.', '.', '.', '.', '.'],  # Casella vuota scura/chiara
        ['.', '.', '.', '.', '.', '.', '.', '.'],  # Casella vuota chiara/scura
        ['.', '.', '.', '.', '.', '.', '.', '.'],  # Casella vuota scura/chiara
        ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],  # Pedoni bianchi
        ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']   # Pezzi bianchi (riga inferiore)
    ]
    
    # Dizionario per i template dei pezzi
    piece_templates = {}
    empty_templates = {'light': None, 'dark': None}
    
    # Estrai i template per ogni tipo di pezzo
    for row_idx, row in enumerate(initial_position):
        for col_idx, piece in enumerate(row):
            if piece != '.':
                # Questo è un pezzo
                square = ref_squares[row_idx][col_idx]
                # Redimensiona per consistenza
                square_resized = cv2.resize(square, (50, 50))
                piece_templates[piece] = square_resized
            else:
                # Questo è uno spazio vuoto
                # Verifichiamo se è una casella chiara o scura
                square = ref_squares[row_idx][col_idx]
                square_resized = cv2.resize(square, (50, 50))
                
                # Converti in scala di grigi
                if len(square_resized.shape) == 3:
                    gray = cv2.cvtColor(square_resized, cv2.COLOR_BGR2GRAY)
                else:
                    gray = square_resized
                
                # Equalizzazione dell'istogramma
                equalized = cv2.equalizeHist(gray)
                
                # Calcola la luminosità media
                brightness = np.mean(equalized)
                
                # Determina se è una casella chiara o scura
                if brightness > 150:  # Prova con una soglia più alta
                    if empty_templates['light'] is None:
                        empty_templates['light'] = square_resized
                else:
                    if empty_templates['dark'] is None:
                        empty_templates['dark'] = square_resized
    
    # Salva i template per debug
    debug_templates_dir = "debug_templates"
    if not os.path.exists(debug_templates_dir):
        os.makedirs(debug_templates_dir)
    for piece, template in piece_templates.items():
        cv2.imwrite(f"{debug_templates_dir}/{piece}.png", template)
    
    return piece_templates, empty_templates

def enhance_contrast(image):
    """Applica la correzione gamma per migliorare il contrasto."""
    gamma = 1.2  # Valore di gamma per la correzione
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, look_up_table)

def match_piece_with_templates(square_img, piece_templates, empty_templates):
    """Confronta una casella con i template per identificare il pezzo."""
    # Redimensiona la casella per coerenza
    square_resized = cv2.resize(square_img, (50, 50))
    
    # Migliora il contrasto della casella
    if len(square_resized.shape) == 3:
        square_gray = cv2.cvtColor(square_resized, cv2.COLOR_BGR2GRAY)
    else:
        square_gray = square_resized

    # Migliora il contrasto con la correzione gamma
    square_gray = enhance_contrast(square_gray)

    # Equalizzazione dell'istogramma per migliorare il contrasto
    square_gray = cv2.equalizeHist(square_gray)
    
    best_match = None
    highest_score = -float('inf')
    
    # Controlla prima se è una casella vuota
    is_empty = False
    for empty_type, template in empty_templates.items():
        if template is not None:
            # Converti in scala di grigi se necessario
            if len(square_resized.shape) == 3 and len(template.shape) == 3:
                square_gray = cv2.cvtColor(square_resized, cv2.COLOR_BGR2GRAY)
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            elif len(square_resized.shape) == 3:
                square_gray = cv2.cvtColor(square_resized, cv2.COLOR_BGR2GRAY)
                template_gray = template
            elif len(template.shape) == 3:
                square_gray = square_resized
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                square_gray = square_resized
                template_gray = template
            
            # Utilizza il metodo di corrispondenza del template
            result = cv2.matchTemplate(square_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            score = np.max(result)
            
            if score > 0.8:  # Alta somiglianza indica una casella vuota
                is_empty = True
                break
    
    if is_empty:
        return '.'
    
    # Controlla con tutti i template dei pezzi
    for piece_name, template in piece_templates.items():
        if template is None:
            continue
            
        # Converti in scala di grigi se necessario
        if len(square_resized.shape) == 3 and len(template.shape) == 3:
            square_gray = cv2.cvtColor(square_resized, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        elif len(square_resized.shape) == 3:
            square_gray = cv2.cvtColor(square_resized, cv2.COLOR_BGR2GRAY)
            template_gray = template
        elif len(template.shape) == 3:
            square_gray = square_resized
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            square_gray = square_resized
            template_gray = template
        
        # Utilizza il metodo di corrispondenza del template
        result = cv2.matchTemplate(square_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        score = np.max(result)
        
        if score > highest_score:
            highest_score = score
            best_match = piece_name
    
    # Se il punteggio di corrispondenza è troppo basso, potrebbe essere una casella vuota
    threshold = 0.4  # Riduci la soglia per migliorare il rilevamento dei pezzi bianchi
    if highest_score < threshold:
        return '.'
    
    return best_match

def analyze_board_with_templates(squares, piece_templates, empty_templates):
    """Analizza la scacchiera utilizzando i template dei pezzi."""
    board_state = []
    
    for row in squares:
        row_state = []
        for square in row:
            piece = match_piece_with_templates(square, piece_templates, empty_templates)
            row_state.append(piece)
        board_state.append(row_state)
    
    return board_state

def simple_color_analysis(square):
    """Analisi semplice del colore per identificare pezzi senza clustering."""
    # Redimensiona per velocizzare l'elaborazione
    resized = cv2.resize(square, (50, 50))
    
    # Converti in scala di grigi
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    # Equalizzazione dell'istogramma
    equalized = cv2.equalizeHist(gray)
    
    # Calcola caratteristiche statistiche
    mean_val = np.mean(equalized)
    std_val = np.std(equalized)
    
    # Applica threshold adattivo per aiutare a separare i pezzi dallo sfondo
    thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Conta i pixel bianchi/neri dopo il threshold
    white_pixels = np.sum(thresh == 255)
    black_pixels = np.sum(thresh == 0)
    
    # Determina se c'è un pezzo e di che colore
    if std_val > 50:  # Alta deviazione standard può indicare la presenza di un pezzo
        if mean_val < 100:  # Valore medio basso può suggerire un pezzo nero
            return 'p'  # Usa 'p' come segnaposto per i pezzi neri
        elif mean_val > 150:  # Valore medio alto può suggerire un pezzo bianco
            return 'P'  # Usa 'P' come segnaposto per i pezzi bianchi
        else:
            return '.'  # Ambiguo, consideralo vuoto
    else:
        return '.'  # Casella vuota

def alternative_piece_detection(squares):
    """Metodo alternativo per rilevare i pezzi quando i template non funzionano bene."""
    board_state = []
    
    for row in squares:
        row_state = []
        for square in row:
            piece = simple_color_analysis(square)
            row_state.append(piece)
        
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
                # Se c'era uno spazio vuoto prima, aggiungilo
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                
                # Aggiungi il pezzo
                fen_row += cell
        
        # Se la riga finisce con spazi vuoti
        if empty_count > 0:
            fen_row += str(empty_count)
        
        fen_parts.append(fen_row)
    
    # Unisci le parti con '/' come separatore
    fen = '/'.join(fen_parts)
    
    # Aggiungi le informazioni aggiuntive della notazione FEN
    fen += " w KQkq - 0 1"
    
    return fen

def analyze_chessboard(image_path, reference_image_path=None):
    """Analizza una scacchiera da un'immagine e genera la notazione FEN."""
    try:
        # Estrai la scacchiera dall'immagine
        board_img = extract_board(image_path)
        
        # Salva l'immagine della scacchiera estratta per debug
        cv2.imwrite("debug_extracted_board.png", board_img)
        
        # Dividi la scacchiera in caselle
        squares = split_board_into_squares(board_img)
        
        # Se è fornita un'immagine di riferimento, crea template da essa
        if reference_image_path:
            piece_templates, empty_templates = create_piece_templates_from_reference(reference_image_path)
            
            # Analizza la scacchiera usando i template
            board_state = analyze_board_with_templates(squares, piece_templates, empty_templates)
        else:
            # Usa un metodo alternativo se non è disponibile un'immagine di riferimento
            board_state = alternative_piece_detection(squares)
        
        # Debug: salva le caselle estratte per verifica
        debug_output_dir = "debug_squares"
        if not os.path.exists(debug_output_dir):
            os.makedirs(debug_output_dir)
            
        for i, row in enumerate(squares):
            for j, square in enumerate(row):
                cv2.imwrite(f"{debug_output_dir}/square_{i}_{j}.png", square)
        
        # Genera la notazione FEN
        fen = generate_fen(board_state)
        
        return fen, board_state
        
    except Exception as e:
        print(f"Errore nell'analisi della scacchiera: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def visualize_board_state(board_state):
    """Visualizza lo stato della scacchiera in forma testuale."""
    print("  a b c d e f g h")
    print(" +-----------------+")
    for i, row in enumerate(board_state):
        print(f"{8-i}| {' '.join([piece if piece != '.' else '.' for piece in row])} |")
    print(" +-----------------+")
    print("  a b c d e f g h")

def compare_histograms(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

def main():
    # Percorso all'immagine della scacchiera
    image_path = "Foto1.jpg"  # Modifica con il percorso della tua immagine
    
    # Percorso all'immagine di riferimento (posizione iniziale)
    reference_image_path = "ScacchieraCompleta1.jpg"  # Modifica con il percorso della tua immagine di riferimento
    
    print(f"Analisi dell'immagine: {image_path}")
    print(f"Usando l'immagine di riferimento: {reference_image_path}")
    
    # Verifica che le immagini esistano
    if not os.path.exists(image_path):
        print(f"ERRORE: L'immagine {image_path} non esiste.")
        return
        
    if reference_image_path and not os.path.exists(reference_image_path):
        print(f"ERRORE: L'immagine di riferimento {reference_image_path} non esiste.")
        print("Procedendo senza immagine di riferimento...")
        reference_image_path = None
    
    # Analizza la scacchiera e genera la notazione FEN
    fen, board_state = analyze_chessboard(image_path, reference_image_path)
    
    if fen:
        print(f"\nNotazione FEN: {fen}")
        
        # Visualizza lo stato della scacchiera per debug
        print("\nStato della scacchiera rilevato:")
        visualize_board_state(board_state)
        
        print("\nLe immagini di debug sono state salvate nella directory 'debug_squares'")
    else:
        print("Impossibile generare la notazione FEN.")

if __name__ == "__main__":
    main()