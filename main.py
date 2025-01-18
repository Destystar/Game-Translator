import cv2
import pytesseract
from PIL import Image
import pygame
from google.cloud import translate_v2 as translate
import numpy as np

# Initialize Google Cloud Translation client
client = translate.Client()

def detect_and_translate(text):
    # Translate text from Chinese to English
    result = client.translate(text, target_language='en-US')
    return result['translatedText']

def process_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROLOGY_SIMPLE)
    
    # Filter contours to keep only those with area > 1000 pixels
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]
    
    # Sort contours by area
    sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
    
    # Process top 5 largest contours
    processed_texts = []
    for contour in sorted_contours[:5]:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop the contour region
        roi = gray[y:y+h, x:x+w]
        
        # Resize the ROI to fit Tesseract input size
        resized_roi = cv2.resize(roi, (500, 200))
        
        # Extract text using Tesseract-OCR
        text = pytesseract.image_to_string(Image.fromarray(resized_roi), lang='chi_sim')
        processed_texts.append(text.strip())
    
    return ' '.join(processed_texts)

def main():
    # Initialize Pygame
    pygame.init()
    width, height = 1920, 1080
    game_window = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Game Text Translator')
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Capture screen
        screen = cv2.cvtColor(pygame.surfarray.array3d(game_window), cv2.COLOR_RGB2BGR)

        # Process frame
        text = process_frame(screen)

        # Translate detected text
        translated_text = detect_and_translate(text)

        # Render translated text on screen
        game_window.fill((0, 0, 0))
        font = pygame.font.SysFont('arial', 24)
        text_surface = font.render(translated_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(width // 2, height // 2))
        game_window.blit(text_surface, text_rect)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
