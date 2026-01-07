"""
CLIP Image Classification Training Script
Fine-tuning CLIP for image classification on custom datasets
"""

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import CLIPProcessor, CLIPModel
import datasets
from pathlib import Path
import argparse
import json
from tqdm import tqdm


class CLIPTrainer:
    """Trainer class for fine-tuning CLIP on image classification tasks"""
    
    def __init__(self, model_name, labels, device='cuda'):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.labels = labels
        self.device = device
        self.model.to(device)
        self.loss_fn = CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        
    def train_epoch(self, dataset, batch_size, optimizer, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        batch_steps = np.arange(0, dataset.num_rows + batch_size, batch_size)
        
        losses = []
        accs = []
        progress_bar = tqdm(range(len(batch_steps) - 1), desc="Training")
        
        for i in progress_bar:
            # Get batch
            inputs = dataset[batch_steps[i]:batch_steps[i+1]]
            inputs_cuda = {k: v.to(self.device) for k, v in inputs.items()}
            batch_labels = inputs_cuda.pop("labels")
            
            # Forward pass
            outputs = self.model(**inputs_cuda)
            logits = outputs.logits_per_image
            
            # Calculate loss and accuracy
            loss = self.loss_fn(logits, batch_labels)
            acc = (torch.argmax(logits, dim=1) == batch_labels).float().mean().item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            
            losses.append(loss.item())
            accs.append(acc)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{np.mean(losses):.4f}',
                'acc': f'{np.mean(accs):.4f}'
            })
        
        return np.mean(losses), np.mean(accs)
    
    def validate(self, dataset, batch_size):
        """Validate the model"""
        self.model.eval()
        batch_steps = np.arange(0, dataset.num_rows + batch_size, batch_size)
        
        losses = []
        accs = []
        
        with torch.no_grad():
            for i in tqdm(range(len(batch_steps) - 1), desc="Validating"):
                inputs = dataset[batch_steps[i]:batch_steps[i+1]]
                inputs_cuda = {k: v.to(self.device) for k, v in inputs.items()}
                batch_labels = inputs_cuda.pop("labels")
                
                outputs = self.model(**inputs_cuda)
                logits = outputs.logits_per_image
                
                loss = self.loss_fn(logits, batch_labels)
                acc = (torch.argmax(logits, dim=1) == batch_labels).float().mean().item()
                
                losses.append(loss.item())
                accs.append(acc)
        
        return np.mean(losses), np.mean(accs)
    
    def predict(self, dataset, batch_size):
        """Generate predictions for test dataset"""
        self.model.eval()
        batch_steps = np.arange(0, dataset.num_rows + batch_size, batch_size)
        
        predictions = []
        pred_probs = []
        
        with torch.no_grad():
            for i in tqdm(range(len(batch_steps) - 1), desc="Predicting"):
                inputs = dataset[batch_steps[i]:batch_steps[i+1]]
                inputs_cuda = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs_cuda)
                logits = outputs.logits_per_image
                probs = self.softmax(logits)
                
                p_max, p_label = torch.max(probs, dim=1)
                predictions += [self.labels[idx] for idx in p_label]
                pred_probs += [p.item() for p in p_max]
        
        return predictions, pred_probs
    
    def save_model(self, save_path):
        """Save the model"""
        self.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train CLIP for image classification')
    parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-large-patch14',
                        help='CLIP model to use')
    parser.add_argument('--img_size', type=int, nargs=2, default=[64, 64],
                        help='Image size (height width)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-6, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--use_pseudo_labels', action='store_true',
                        help='Use pseudo-labeling for semi-supervised learning')
    parser.add_argument('--pseudo_threshold', type=float, default=0.9,
                        help='Confidence threshold for pseudo-labels')
    
    args = parser.parse_args()
    
    # Import data loading utilities
    from data_utils import load_dataset, preprocess_dataset, apply_transforms
    
    # Load data
    print("Loading dataset...")
    dataset, labels = load_dataset(args.train_dir, args.test_dir, img_size=tuple(args.img_size))
    
    # Load model and processor
    print(f"Loading model: {args.model_name}")
    trainer = CLIPTrainer(args.model_name, labels)
    
    # Apply transforms
    print("Applying transforms...")
    dataset = apply_transforms(dataset, trainer.processor, labels, augment=True)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(range(0, dataset["train"].num_rows, args.batch_size)) * args.epochs
    )
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(
            dataset["train"], args.batch_size, optimizer, scheduler
        )
        
        # Validate
        val_loss, val_acc = trainer.validate(dataset["valid"], args.batch_size)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = Path(args.output_dir) / "best_model"
            output_path.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(output_path))
            print(f"New best model! Validation accuracy: {val_acc:.4f}")
    
    # Save final model
    final_path = Path(args.output_dir) / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_path))
    
    # Save training history
    with open(Path(args.output_dir) / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Pseudo-labeling (optional)
    if args.use_pseudo_labels:
        print("\nGenerating pseudo-labels for semi-supervised learning...")
        predictions, probs = trainer.predict(dataset["test"], args.batch_size)
        
        # Filter high-confidence predictions
        from data_utils import create_pseudo_labeled_dataset
        enhanced_dataset = create_pseudo_labeled_dataset(
            dataset, predictions, probs, args.pseudo_threshold
        )
        
        # Retrain with pseudo-labels
        print(f"\nRetraining with {len(enhanced_dataset['train'])} samples (including pseudo-labels)...")
        enhanced_dataset = apply_transforms(enhanced_dataset, trainer.processor, labels, augment=True)
        
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=args.lr * 0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(range(0, enhanced_dataset["train"].num_rows, args.batch_size)) * 2
        )
        
        for epoch in range(2):
            print(f"\nPseudo-label Epoch {epoch + 1}/2")
            train_loss, train_acc = trainer.train_epoch(
                enhanced_dataset["train"], args.batch_size, optimizer, scheduler
            )
            val_loss, val_acc = trainer.validate(enhanced_dataset["valid"], args.batch_size)
            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        # Save pseudo-labeled model
        pseudo_path = Path(args.output_dir) / "pseudo_labeled_model"
        pseudo_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(pseudo_path))
    
    # Generate final predictions
    print("\nGenerating test predictions...")
    predictions, probs = trainer.predict(dataset["test"], args.batch_size)
    
    # Save predictions
    from data_utils import save_predictions
    save_predictions(predictions, Path(args.output_dir) / "predictions.csv")
    
    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
