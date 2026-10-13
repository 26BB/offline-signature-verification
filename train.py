import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import SiameseEmbeddingNet, ContrastiveLoss
from dataset import SignaturePairsDataset

def train(dataset_dir="dataset/raw", epochs=20, batch_size=32, lr=0.0005, save_path="models/siamese_signature.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print(f"Warning: No dataset found in '{dataset_dir}'.")
        print("Please place collected student signatures in the dataset directory before training.")
        return

    train_dataset = SignaturePairsDataset(root_dir=dataset_dir, num_pairs=2000, is_training=True)
    if len(train_dataset) == 0:
        print("Dataset does not have enough student data to create training pairs.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SiameseEmbeddingNet(embedding_dim=128).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_loss = float("inf")

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        scheduler.step()
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Average Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model checkpoint to {save_path}")

    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Network for Signature Verification")
    parser.add_argument("--dataset", type=str, default="dataset/raw", help="Path to raw dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="models/siamese_signature.pth", help="Checkpoint save path")
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )
