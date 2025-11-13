import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from convgru import Config, WeatherDataset, ConvGRUModel
from tqdm import tqdm
import copy


def train_model(model, train_loader, future_hours):
    """Train a single ConvGRU model for a given future offset (in hours)."""
    model = model.to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    criterion = torch.nn.MSELoss()

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"[{future_hours}h] Epoch {epoch+1}/{Config.EPOCHS}"):
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Future {future_hours}h | Epoch {epoch+1}/{Config.EPOCHS} | Loss: {avg_loss:.6f}")

    model_path = f"ConvGRU/conv_gru_model_{future_hours}h.pth"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Saved model for {future_hours}h future prediction → {model_path}\n")


def main():
    # Set GPU manually - change this to your preferred GPU
    Config.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {Config.DEVICE}")

    # Train separate models for 1h, 2h, 3h, 4h future predictions
    for future_hours in [2,3,4]:
        print(f"\n🚀 Training ConvGRU model for {future_hours}-hour future prediction...\n")

        # Set the future offset
        Config.FUTURE_OFFSET_HOURS = future_hours

        # Build dataset and loader
        dataset = WeatherDataset(Config.HRIT_PATH)
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)

        # Build a fresh model
        model = ConvGRUModel(input_channels=Config.INPUT_CHANNELS)

        # Train and save
        train_model(model, loader, future_hours)


if __name__ == "__main__":
    main()
