import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import cv2
from typing import List, Tuple, Dict, Optional

class DocumentImageDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None, image_size: int = 224):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            # Assume it's already a numpy array
            image = Image.fromarray(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

class DocumentAugmentation:
    def __init__(self):
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ])
    
    def augment_image(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations to image"""
        return self.augmentation_transforms(image)
    
    def create_augmented_dataset(self, image_paths: List[str], labels: List[int], 
                               augment_factor: int = 2) -> Tuple[List, List]:
        """Create augmented dataset"""
        augmented_paths = []
        augmented_labels = []
        
        for path, label in zip(image_paths, labels):
            # Add original
            augmented_paths.append(path)
            augmented_labels.append(label)
            
            # Add augmented versions
            for _ in range(augment_factor):
                augmented_paths.append(path)
                augmented_labels.append(label)
        
        return augmented_paths, augmented_labels

def create_data_loaders(train_paths: List[str], train_labels: List[int],
                       val_paths: List[str], val_labels: List[int],
                       batch_size: int = 32, image_size: int = 224,
                       augment: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=(-3, 3)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]) if augment else transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DocumentImageDataset(train_paths, train_labels, train_transform, image_size)
    val_dataset = DocumentImageDataset(val_paths, val_labels, val_transform, image_size)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def preprocess_document_image(image: np.ndarray) -> np.ndarray:
    """Preprocess document image for better analysis"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Noise reduction
    denoised = cv2.medianBlur(gray, 3)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Edge detection for document structure
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Morphological operations to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return morphed

def extract_document_features(image: np.ndarray) -> Dict:
    """Extract visual features from document image"""
    # Preprocess image
    processed = preprocess_document_image(image)
    
    # Calculate basic statistics
    height, width = processed.shape[:2]
    
    # Text region analysis
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_area = (width * height) * 0.001  # 0.1% of image area
    text_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Calculate features
    features = {
        'image_width': width,
        'image_height': height,
        'aspect_ratio': width / height,
        'text_regions': len(text_contours),
        'text_density': sum(cv2.contourArea(c) for c in text_contours) / (width * height),
        'avg_text_region_size': np.mean([cv2.contourArea(c) for c in text_contours]) if text_contours else 0,
        'edge_density': np.sum(processed > 0) / (width * height)
    }
    
    # Layout analysis
    if text_contours:
        # Get bounding rectangles
        rects = [cv2.boundingRect(c) for c in text_contours]
        
        # Analyze text distribution
        x_coords = [r[0] for r in rects]
        y_coords = [r[1] for r in rects]
        
        features.update({
            'text_horizontal_spread': (max(x_coords) - min(x_coords)) / width if x_coords else 0,
            'text_vertical_spread': (max(y_coords) - min(y_coords)) / height if y_coords else 0,
            'text_alignment_variance': np.var(x_coords) if x_coords else 0
        })
    
    return features
