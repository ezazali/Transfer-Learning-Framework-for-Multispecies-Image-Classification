 import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AnimalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_dataset(dataset_path):
    """Load dataset from the folder structure"""
    # Map Italian animal names to English
    class_mapping = {
        'cane': 'dog',
        'cavallo': 'horse', 
        'elefante': 'elephant',
        'farfalla': 'butterfly',
        'gallina': 'chicken',
        'gatto': 'cat',
        'mucca': 'cow',
        'pecora': 'sheep',
        'ragno': 'spider',
        'scoiattolo': 'squirrel'
    }
    
    image_paths = []
    labels = []
    class_names = []
    
     
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            # Map Italian to English class name
            class_name = class_mapping.get(folder, folder)
            class_names.append(class_name)
            
            # Get all images in the folder
            for image_file in os.listdir(folder_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_paths.append(os.path.join(folder_path, image_file))
                    labels.append(len(class_names) - 1)  # Use index as label
    
    return image_paths, labels, sorted(list(set(class_names)))

def create_data_loaders(image_paths, labels, class_names, batch_size=32):
    """Create train and test data loaders with 80:20 split"""
    
     
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception v3 input size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
     
    train_dataset = AnimalDataset(X_train, y_train, train_transform)
    test_dataset = AnimalDataset(X_test, y_test, test_transform)
    
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, len(class_names)

def create_model(num_classes):
    """Create GoogLeNet Inception v3 model"""
    # Load pretrained Inception v3
    model = models.inception_v3(pretrained=True)
    
 
    for param in model.parameters():
        param.requires_grad = False
    
   
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    
    for param in model.Mixed_7c.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    
 
    if hasattr(model, 'AuxLogits'):
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    
    return model.to(device)

def train_model(model, train_loader, test_loader, num_epochs=10):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
             
            if model.training:
                outputs, aux_outputs = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{running_loss/(train_pbar.n+1):.4f}',
                'Acc': f'{100.*correct_train/total_train:.2f}%'
            })
        
       
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct_train / total_train
        val_acc = 100. * correct_val / total_val
        avg_loss = running_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        scheduler.step()
    
    return model, train_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, class_names):
    """Evaluate the model and return predictions"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_predictions)

def calculate_metrics(y_true, y_pred, class_names):
    """Calculate all evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }

def plot_training_history(train_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return cm

def display_metrics(metrics, class_names):
    """Display all evaluation metrics"""
    print("="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Overall Precision: {metrics['precision']:.4f}")
    print(f"Overall Recall: {metrics['recall']:.4f}")
    print(f"Overall F1-Score: {metrics['f1_score']:.4f}")
    print()
    
     print("Per-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<12} {metrics['precision_per_class'][i]:<10.4f} "
              f"{metrics['recall_per_class'][i]:<10.4f} {metrics['f1_per_class'][i]:<10.4f}")
    print("="*60)

def main():
    """Main function to run the complete pipeline"""
    # Configuration
    dataset_path = "/kaggle/input/animals10/raw-img"  # Update this path to your dataset location
    batch_size = 16  # Reduced batch size for better memory management
    num_epochs = 10
    
    print("Loading dataset...")
    image_paths, labels, class_names = load_dataset(dataset_path)
    print(f"Total images: {len(image_paths)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
     
    print("\nCreating data loaders...")
    train_loader, test_loader, num_classes = create_data_loaders(
        image_paths, labels, class_names, batch_size
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    
     
    print("\nCreating GoogLeNet Inception v3 model...")
    model = create_model(num_classes)
    
     
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
     
    print(f"\nTraining model for {num_epochs} epochs...")
    model, train_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, test_loader, num_epochs
    )
    
    
    plot_training_history(train_losses, train_accuracies, val_accuracies)
    
    
    print("\nEvaluating model...")
    y_true, y_pred = evaluate_model(model, test_loader, class_names)
    
     
    metrics = calculate_metrics(y_true, y_pred, class_names)
    
   
    display_metrics(metrics, class_names)
    
    
    print("\nGenerating confusion matrix...")
    cm = plot_confusion_matrix(y_true, y_pred, class_names)
    
   
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
     
    torch.save(model.state_dict(), 'inception_v3_animal_classifier.pth')
    print("\nModel saved as 'inception_v3_animal_classifier.pth'")

if __name__ == "__main__":
    main()
