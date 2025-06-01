import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
import timm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class DeerDatasetFromArrays(Dataset):
    """
    Dataset class that works with your existing numpy arrays
    Converts grayscale to RGB for transfer learning models
    """

    def __init__(self, X, y, transform=None, is_training=True):
        """
        X: numpy array of shape (N, 288, 288, 1) - your grayscale images
        y: numpy array of labels
        transform: torchvision transforms
        """
        self.X = X
        self.y = y
        self.transform = transform
        self.is_training = is_training

        print(f"Dataset created with {len(X)} samples, shape: {X.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Get image and label
        image = self.X[idx]  # Shape: (288, 288, 1)
        label = self.y[idx]

        # Convert grayscale to RGB for transfer learning
        # Squeeze the channel dimension and repeat 3 times
        image_gray = image.squeeze(-1)  # (288, 288)
        image_rgb = np.stack(
            [image_gray, image_gray, image_gray], axis=-1
        )  # (288, 288, 3)

        # Normalize to 0-1 range if not already
        if image_rgb.max() > 1.0:
            image_rgb = image_rgb.astype(np.float32) / 255.0
        else:
            image_rgb = image_rgb.astype(np.float32)

        # Convert to PIL Image for torchvision transforms
        from PIL import Image

        image_pil = Image.fromarray((image_rgb * 255).astype(np.uint8))

        # Apply transforms
        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            # Basic conversion to tensor
            image_tensor = transforms.ToTensor()(image_pil)

        return image_tensor, torch.tensor(label, dtype=torch.long)


class DeerAgeEfficientNet:
    """
    EfficientNet implementation adapted for your existing pipeline
    """

    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        print(f"Using device: {self.device}")

    def create_model(self):
        """
        Create EfficientNet model for deer age classification
        """
        # Load pre-trained EfficientNet-B4
        self.model = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            num_classes=self.num_classes,
            drop_rate=0.3,
            drop_path_rate=0.2,
        )

        self.model = self.model.to(self.device)
        print(
            f"✅ EfficientNet-B4 created with {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )
        return self.model

    def get_transforms(self):
        """
        Data transforms optimized for your deer images
        Since you're already doing augmentation, these are minimal
        """

        # Training transforms - minimal since you already augment
        train_transform = transforms.Compose(
            [
                transforms.Resize((384, 384)),  # Resize to EfficientNet optimal size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                    std=[0.229, 0.224, 0.225],
                ),
                # Optional: small additional augmentations
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ]
        )

        # Validation transforms - no augmentation
        val_transform = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return train_transform, val_transform

    def prepare_data_loaders(self, X_train, y_train, X_val, y_val, batch_size=16):
        """
        Create data loaders from your existing arrays
        """
        print("🔄 Preparing data loaders from your arrays...")

        # Get transforms
        train_transform, val_transform = self.get_transforms()

        # Create datasets
        train_dataset = DeerDatasetFromArrays(
            X_train, y_train, train_transform, is_training=True
        )
        val_dataset = DeerDatasetFromArrays(
            X_val, y_val, val_transform, is_training=False
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        print(f"✅ Training loader: {len(train_loader)} batches")
        print(f"✅ Validation loader: {len(val_loader)} batches")

        return train_loader, val_loader

    def train_model(self, train_loader, val_loader, epochs=50, learning_rate=1e-4):
        """
        Train the model with your data
        """
        print(f"🚀 Starting training for {epochs} epochs...")

        # Optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=0.01
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Training history
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        best_val_acc = 0.0
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                # Statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                # Progress update every 20 batches
                if batch_idx % 20 == 0:
                    current_acc = 100.0 * train_correct / train_total
                    print(
                        f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f} | Acc: {current_acc:.1f}%"
                    )

            # Validation phase
            val_loss, val_acc = self._validate(val_loader, criterion)

            # Update history
            train_acc = 100.0 * train_correct / train_total
            train_loss = train_loss / len(train_loader)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Print epoch results
            print(f"\n📊 Epoch {epoch + 1}/{epochs} Results:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f'   Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "best_deer_efficientnet.pth")
                patience_counter = 0
                print(f"   🎯 New best model! Validation Accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(
                    f"\n⏰ Early stopping after {epoch + 1} epochs (patience: {patience})"
                )
                break

            scheduler.step()
            print("-" * 60)

        # Load best model
        self.model.load_state_dict(torch.load("best_deer_efficientnet.pth"))
        print(f"\n🏆 Training completed! Best validation accuracy: {best_val_acc:.2f}%")

        return history, best_val_acc

    def _validate(self, val_loader, criterion):
        """
        Validation function
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        return val_loss, val_acc

    def evaluate_on_test(self, X_test, y_test, label_mapping):
        """
        Final evaluation on your test set
        """
        print("🎯 Final evaluation on test set...")

        # Create test dataset and loader
        _, val_transform = self.get_transforms()
        test_dataset = DeerDatasetFromArrays(
            X_test, y_test, val_transform, is_training=False
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Get predictions
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_predictions)

        # Create age labels from your mapping
        age_labels = [
            f"Age {list(label_mapping.keys())[i]}" for i in range(len(label_mapping))
        ]

        print(f"\n🎯 FINAL TEST RESULTS:")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"Improvement over your CNN baseline: +{(accuracy - 0.31) * 100:.1f}%")

        if accuracy > 0.578:
            print("🎉 SUCCESS! You've beaten your tree model baseline (57.8%)!")

        # Detailed classification report
        print(f"\n📊 Detailed Classification Report:")
        print(
            classification_report(all_labels, all_predictions, target_names=age_labels)
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=age_labels,
            yticklabels=age_labels,
        )
        plt.title(f"Confusion Matrix - Test Accuracy: {accuracy * 100:.1f}%")
        plt.ylabel("True Age")
        plt.xlabel("Predicted Age")
        plt.tight_layout()
        plt.show()

        return accuracy, all_predictions, all_probabilities

    def plot_training_history(self, history):
        """
        Plot training curves
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1.plot(history["train_loss"], label="Training Loss", linewidth=2)
        ax1.plot(history["val_loss"], label="Validation Loss", linewidth=2)
        ax1.set_title("Model Loss", fontsize=14)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2.plot(history["train_acc"], label="Training Accuracy", linewidth=2)
        ax2.plot(history["val_acc"], label="Validation Accuracy", linewidth=2)
        ax2.set_title("Model Accuracy", fontsize=14)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print final stats
        final_train_acc = history["train_acc"][-1]
        final_val_acc = history["val_acc"][-1]
        best_val_acc = max(history["val_acc"])

        print(f"📈 Training Summary:")
        print(f"   Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"   Final Validation Accuracy: {final_val_acc:.2f}%")
        print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")


def run_efficientnet_on_your_data(
    X_train, y_train, X_val, y_val, X_test, y_test, label_mapping
):
    """
    Complete pipeline using your existing data
    """
    print("🦌 DEER AGE CLASSIFICATION WITH EFFICIENTNET")
    print("=" * 55)
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Number of classes: {len(label_mapping)}")
    print(f"Age mapping: {label_mapping}")
    print("=" * 55)

    # Initialize classifier
    num_classes = len(label_mapping)
    classifier = DeerAgeEfficientNet(num_classes=num_classes)

    # Create model
    model = classifier.create_model()

    # Prepare data loaders
    train_loader, val_loader = classifier.prepare_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=16
    )

    # Train model
    history, best_val_acc = classifier.train_model(
        train_loader, val_loader, epochs=50, learning_rate=1e-4
    )

    # Plot training history
    classifier.plot_training_history(history)

    # Final evaluation on test set
    test_accuracy, predictions, probabilities = classifier.evaluate_on_test(
        X_test, y_test, label_mapping
    )

    print(f"\n🎯 FINAL COMPARISON:")
    print(f"Your previous CNN: 31.0%")
    print(f"Your best tree model: 57.8%")
    print(f"EfficientNet result: {test_accuracy * 100:.1f}%")
    print(f"Improvement: +{(test_accuracy - 0.578) * 100:.1f}% over tree models")

    return classifier, history, test_accuracy


# Debug function to visualize data conversion
def debug_data_conversion(X_sample, y_sample, num_samples=3):
    """
    Debug function to see how your grayscale images are converted
    """
    print("🔍 DEBUGGING DATA CONVERSION")
    print("=" * 35)

    classifier = DeerAgeEfficientNet(num_classes=5)
    train_transform, _ = classifier.get_transforms()

    # Create sample dataset
    dataset = DeerDatasetFromArrays(
        X_sample[:num_samples],
        y_sample[:num_samples],
        train_transform,
        is_training=False,
    )

    for i in range(min(num_samples, len(dataset))):
        # Get original and processed
        original = X_sample[i].squeeze()  # Remove channel dimension for display
        processed_tensor, label = dataset[i]

        # Denormalize processed tensor for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denorm = processed_tensor * std + mean
        denorm = torch.clamp(denorm, 0, 1)

        # Convert back to numpy for display
        processed_rgb = denorm.permute(1, 2, 0).numpy()

        # Display
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(original, cmap="gray")
        plt.title(f"Original Grayscale\nLabel: {label.item()}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(processed_rgb)
        plt.title(f"Converted to RGB\nShape: {processed_tensor.shape}")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(processed_rgb[:, :, 0], cmap="gray")  # Show one channel
        plt.title("RGB Channel (same as original)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        print(f"Sample {i + 1}:")
        print(f"  Original shape: {original.shape}")
        print(f"  Processed shape: {processed_tensor.shape}")
        print(f"  Label: {label.item()}")
        print(
            f"  Tensor range: [{processed_tensor.min():.3f}, {processed_tensor.max():.3f}]"
        )
        print()


print("✅ EfficientNet implementation ready for your pipeline!")
print("Next: Run debug_data_conversion() to verify data handling")
print("Then: Run run_efficientnet_on_your_data() for full training")
