import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import glob

# --- VISION LAYER IMPORTS ---
from .model import create_model
from .dataset import create_data_loaders

class ModelTrainer:
    # ... (ModelTrainer class remains the same as before) ...
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(config.get('learning_rate', 1e-4)), 
            weight_decay=config.get('weight_decay', 1e-4)
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        ) 
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'forward') and len(self.model.forward.__code__.co_varnames) > 2:
                outputs, _ = self.model(data)
            else:
                outputs = self.model(data)
            
            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                if hasattr(self.model, 'forward') and len(self.model.forward.__code__.co_varnames) > 2:
                    outputs, _ = self.model(data)
                else:
                    outputs = self.model(data)
                
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        metrics = self.calculate_metrics(all_targets, all_predictions, all_probabilities)
        return epoch_loss, epoch_acc, metrics
    
    def calculate_metrics(self, targets: List[int], predictions: List[int], 
                          probabilities: List[List[float]]) -> Dict:
        report = classification_report(targets, predictions, output_dict=True)
        cm = confusion_matrix(targets, predictions)
        
        if len(np.unique(targets)) == 2:
            probs_positive = [prob[1] for prob in probabilities]
            auc = roc_auc_score(targets, probs_positive)
        else:
            auc = None
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'auc_score': auc,
            'accuracy': report['accuracy'],
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1_score': report['macro avg']['f1-score']
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, save_path: str) -> Dict:
        # ... (rest of train method remains the same) ...
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_metrics = self.validate_epoch(val_loader)
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_metrics': val_metrics,
                    'config': self.config
                }, os.path.join(save_path, 'best_model.pth'))
            
            epoch_time = time.time() - epoch_start
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')
            print(f'Epoch Time: {epoch_time:.2f}s')
            if val_metrics['auc_score']:
                print(f'Val AUC: {val_metrics["auc_score"]:.4f}')
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time/60:.2f} minutes')
        print(f'Best validation accuracy: {self.best_val_acc:.2f}%')
        return {
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'training_time': total_time,
            'final_metrics': val_metrics
        }
    
    def save_training_plots(self, save_path: str):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[1, 0].plot(self.history['learning_rates'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        axes[1, 1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()

def train_document_vision_model(config: dict, train_loader: DataLoader, 
                                val_loader: DataLoader, save_path: str) -> Dict:
    # ... (rest of train_document_vision_model remains the same) ...
    from .model import create_model
    
    model = create_model(
        model_type=config.get('model_type', 'cnn'),
        num_classes=config.get('num_classes', 2),
        backbone=config.get('backbone', 'resnet50'),
        pretrained=config.get('pretrained', True)
    )
    
    trainer = ModelTrainer(model, config)
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('epochs', 15),
        save_path=save_path
    )
    
    trainer.save_training_plots(save_path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'results': results
    }, os.path.join(save_path, 'vision_model.pth'))
    return results

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    def get_image_paths_from_folders(data_dir: str, subset: str) -> List[str]:
        """Dynamically get all image paths from a given directory."""
        image_paths = []
        for label_dir in ['legal', 'non_legal']:
            search_path = Path(data_dir) / subset / label_dir
            for ext in ['jpg', 'jpeg', 'png', 'tiff', 'tif']:
                image_paths.extend(glob.glob(str(search_path / f'*.{ext}')))
        return image_paths

    def get_labels_from_paths(paths: List[str]) -> List[int]:
        """Generate labels (1 for legal, 0 for non_legal) from file paths."""
        labels = []
        for path in paths:
            if 'legal' in path:
                labels.append(1)
            elif 'non_legal' in path:
                labels.append(0)
            else:
                raise ValueError(f"Could not determine label for path: {path}")
        return labels

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    data_dir = config['paths']['data_dir']
    models_dir = config['paths']['models_dir']
    
    try:
        from .dataset import create_data_loaders
        
        train_image_paths = get_image_paths_from_folders(data_dir, 'train')
        train_labels = get_labels_from_paths(train_image_paths)
        
        val_image_paths = get_image_paths_from_folders(data_dir, 'val')
        val_labels = get_labels_from_paths(val_image_paths)
        
        if not train_image_paths:
            raise FileNotFoundError(f"No training images found in '{Path(data_dir) / 'train'}'")
        if not val_image_paths:
            raise FileNotFoundError(f"No validation images found in '{Path(data_dir) / 'val'}'")

        train_loader, val_loader = create_data_loaders(
            train_paths=train_image_paths,
            train_labels=train_labels,
            val_paths=val_image_paths,
            val_labels=val_labels,
            batch_size=config['model']['vision']['batch_size']
        )
    except FileNotFoundError as e:
        print(f"Error: Vision dataset not found. Please check your data directory. Error: {e}")
        exit()
    
    print("Starting Vision model training pipeline...")
    os.makedirs(models_dir, exist_ok=True)
    training_results = train_document_vision_model(
        config=config['model']['vision'],
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=models_dir
    )
    
    print("\nVision model training and saving completed successfully!")
