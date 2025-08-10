import os
import random
import glob
from PIL import Image, ImageDraw, ImageFont

def generate_synthetic_image(is_legal: bool, output_path: str):
    """
    Generates a synthetic document-like image for training.
    """
    width, height = 800, 1100
    background_color = (240, 240, 240)
    font_path = "arial.ttf" # You may need to change this path or use a system font
    
    try:
        font_large = ImageFont.truetype(font_path, 28)
        font_small = ImageFont.truetype(font_path, 16)
    except IOError:
        print("Warning: Font not found. Using default PIL font.")
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    img = Image.new('RGB', (width, height), color=background_color)
    draw = ImageDraw.Draw(img)
    
    # Add a header
    header_text = "LEGAL AGREEMENT" if is_legal else "COMPANY NEWSLETTER"
    draw.text((50, 50), header_text, fill=(30, 30, 30), font=font_large)
    
    # Add body text
    body_start_y = 100
    line_spacing = 24
    
    if is_legal:
        lines = [
            "This agreement is made between Party A and Party B on the date below.",
            "WHEREAS Party A agrees to provide services, and WHEREAS Party B agrees to pay compensation,",
            "NOW, THEREFORE, in consideration of the mutual covenants contained herein, the parties agree as follows:",
            "1. Party A shall provide consulting services in accordance with the Statement of Work.",
            "2. Party B shall pay the agreed compensation of $10,000 upon execution of this agreement.",
            "3. This agreement shall be governed by and construed in accordance with the laws of the State.",
            "IN WITNESS WHEREOF, the parties have executed this agreement as of the date first written above.",
            "",
            "_____________________",
            "/s/ John Doe",
            "Party A",
            "",
            "_____________________",
            "/s/ Jane Smith",
            "Party B"
        ]
        text_color = (50, 50, 150)
    else:
        lines = [
            "Welcome to our quarterly newsletter! We are excited to share our latest updates.",
            "This quarter, we've launched a new product and hosted several successful events.",
            "Our team is committed to providing the best possible experience for our users.",
            "We hope you enjoy this issue and look forward to seeing you at our next event!",
            "Thank you for being a valued customer.",
            "",
            "Contact us at info@company.com"
        ]
        text_color = (80, 80, 80)
    
    for i, line in enumerate(lines):
        draw.text((50, body_start_y + i * line_spacing), line, fill=text_color, font=font_small)

    img.save(output_path)
    
def generate_dataset(num_samples_per_class: int = 10, val_ratio: float = 0.2):
    """
    Generates a small synthetic dataset for vision training, including a validation set.
    """
    base_dir = os.path.join("data")
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    legal_train_dir = os.path.join(train_dir, "legal")
    non_legal_train_dir = os.path.join(train_dir, "non_legal")
    legal_val_dir = os.path.join(val_dir, "legal")
    non_legal_val_dir = os.path.join(val_dir, "non_legal")
    
    # Create all directories, including val
    os.makedirs(legal_train_dir, exist_ok=True)
    os.makedirs(non_legal_train_dir, exist_ok=True)
    os.makedirs(legal_val_dir, exist_ok=True)
    os.makedirs(non_legal_val_dir, exist_ok=True)
    
    num_train_samples = int(num_samples_per_class * (1 - val_ratio))
    num_val_samples = num_samples_per_class - num_train_samples
    
    print(f"Generating {num_train_samples} training images and {num_val_samples} validation images per class...")

    # Generate legal documents
    print("Generating synthetic legal training documents...")
    for i in range(num_train_samples):
        generate_synthetic_image(True, os.path.join(legal_train_dir, f"legal_doc_{i}.png"))
    print("Generating synthetic legal validation documents...")
    for i in range(num_val_samples):
        generate_synthetic_image(True, os.path.join(legal_val_dir, f"legal_doc_{i}.png"))
        
    # Generate non-legal documents
    print("Generating synthetic non-legal training documents...")
    for i in range(num_train_samples):
        generate_synthetic_image(False, os.path.join(non_legal_train_dir, f"non_legal_doc_{i}.png"))
    print("Generating synthetic non-legal validation documents...")
    for i in range(num_val_samples):
        generate_synthetic_image(False, os.path.join(non_legal_val_dir, f"non_legal_doc_{i}.png"))

    print(f"\nGenerated a total of {(num_train_samples + num_val_samples) * 2} images.")
    print("Dataset ready in 'data/train/' and 'data/val/'.")

if __name__ == "__main__":
    generate_dataset(num_samples_per_class=10, val_ratio=0.2)

