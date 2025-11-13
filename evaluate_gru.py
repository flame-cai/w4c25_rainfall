import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from convgru import Config, WeatherDataset, ConvGRUModel
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# Load valid indices

def load_valid_indices(valid_csv_path, input_frames=4, future_frames=16):
    """Return a set of valid starting indices which have 16 consecutive valid frames."""
    df = pd.read_csv(valid_csv_path)
    valid_indices = set(df['valid_indices'].tolist())
    filtered = []

    for idx in valid_indices:
        # Check if all next 16 frames are valid
        if all((idx + i) in valid_indices for i in range(-input_frames, future_frames)):
            filtered.append(idx)
    
    print(f"✅ Filtered {len(filtered)} valid starting indices (with {future_frames} consecutive valid frames)")
    return set(filtered)

# Filtered dataset wrapper

class FilteredWeatherDataset(Dataset):
    """Wrapper over WeatherDataset that only keeps samples starting at valid indices."""
    def __init__(self, base_dataset, valid_start_indices):
        self.base_dataset = base_dataset
        self.valid_start_indices = list(valid_start_indices)
        # self.valid_start_indices = list(base_dataset.valid_start_indices)  # no filtering upfront

    def __getitem__(self, idx):
        try:
            return self.base_dataset[idx]
        except (ValueError, IndexError, KeyError):
            # If this frame fails, skip it
            return None

    def __len__(self):
        return len(self.valid_start_indices)

# =========================
# SSIM WRAPPER
# =========================
def compute_ssim_batch(img1, img2):
    """
    Calculate SSIM between two batches of images using skimage.
    img1, img2: torch tensors of shape [B, C, H, W]
    Returns: average SSIM value across the batch
    """
    # Convert to numpy and move to CPU
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    
    batch_size = img1_np.shape[0]
    ssim_values = []
    
    for i in range(batch_size):
        # For each image in batch, compute SSIM
        # img shape: [C, H, W]
        img1_sample = img1_np[i]
        img2_sample = img2_np[i]
        
        # If single channel, squeeze it out for skimage
        if img1_sample.shape[0] == 1:
            img1_sample = img1_sample[0]
            img2_sample = img2_sample[0]
            ssim_val = ssim(img1_sample, img2_sample, data_range=img1_sample.max() - img1_sample.min())
        else:
            # Multi-channel: use channel_axis parameter
            ssim_val = ssim(img1_sample, img2_sample, 
                          channel_axis=0, 
                          data_range=img1_sample.max() - img1_sample.min())
        
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


# =========================
# EVALUATION FUNCTIONS
# =========================
def evaluate_convgru_model(model, loader, future_hours):
    """
    Evaluate ConvGRU model for RMSE, SSIM, and Bias.
    Returns tuple: (frame_rmse, frame_ssim, frame_bias) - numpy arrays of length FRAMES_PER_HOUR.
    """
    model.eval()
    model.to(Config.DEVICE)
    criterion = torch.nn.MSELoss(reduction='none')

    F = Config.FRAMES_PER_HOUR
    frame_rmse_sum = np.zeros(F)
    frame_ssim_sum = np.zeros(F)
    frame_bias_sum = np.zeros(F)
    batch_count = 0

    with torch.no_grad():
        for x, y_true in tqdm(loader, desc=f"ConvGRU {future_hours}h"):
            x, y_true = x.to(Config.DEVICE), y_true.to(Config.DEVICE)
            y_pred = model(x)

            # de-normalisation
            y_pred = y_pred * 300
            y_true = y_true * 300

            # RMSE calculation
            mse_tensor = criterion(y_pred, y_true)
            mse_per_frame = mse_tensor.mean(dim=(0, 2, 3, 4))
            frame_rmse_sum += torch.sqrt(mse_per_frame).cpu().numpy()
            
            # Bias calculation (mean error)
            bias_per_frame = (y_pred - y_true).mean(dim=(0, 2, 3, 4))
            frame_bias_sum += bias_per_frame.cpu().numpy()

            # SSIM calculation
            for f in range(F):
                frame_ssim = compute_ssim_batch(y_pred[:, f], y_true[:, f])
                frame_ssim_sum[f] += frame_ssim

            batch_count += 1

    if batch_count == 0:
        raise RuntimeError("No batches processed in evaluate_convgru_model.")

    frame_rmse = frame_rmse_sum / batch_count
    frame_ssim = frame_ssim_sum / batch_count
    frame_bias = frame_bias_sum / batch_count
    
    return frame_rmse, frame_ssim, frame_bias


def evaluate_persistence_model(loader, future_hours):
    """
    Evaluate persistence baseline for RMSE, SSIM, and Bias.
    Uses the last frame of input sequence as prediction for all future frames.
    Returns tuple: (frame_rmse, frame_ssim, frame_bias) - numpy arrays of length FRAMES_PER_HOUR.
    """
    criterion = torch.nn.MSELoss(reduction='none')
    
    F = Config.FRAMES_PER_HOUR
    frame_rmse_sum = np.zeros(F)
    frame_ssim_sum = np.zeros(F)
    frame_bias_sum = np.zeros(F)
    batch_count = 0
    
    with torch.no_grad():
        for x, y_true in tqdm(loader, desc=f"Persistence {future_hours}h"):
            x, y_true = x.to(Config.DEVICE), y_true.to(Config.DEVICE)
            
            # x shape: [B, T, C, H, W] - take the last frame
            last_frame = x[:, -1, :, :, :]
            
            # Repeat for all F future frames: [B, F, C, H, W]
            y_pred = last_frame.unsqueeze(1).repeat(1, F, 1, 1, 1)
            
            # de-normalisation
            y_pred = y_pred * 300
            y_true = y_true * 300
            
            # RMSE calculation
            mse_tensor = criterion(y_pred, y_true)
            mse_per_frame = mse_tensor.mean(dim=(0, 2, 3, 4))
            frame_rmse_sum += torch.sqrt(mse_per_frame).cpu().numpy()
            
            # Bias calculation
            bias_per_frame = (y_pred - y_true).mean(dim=(0, 2, 3, 4))
            frame_bias_sum += bias_per_frame.cpu().numpy()
            
            # SSIM calculation
            for f in range(F):
                frame_ssim = compute_ssim_batch(y_pred[:, f], y_true[:, f])
                frame_ssim_sum[f] += frame_ssim
            
            batch_count += 1
    
    if batch_count == 0:
        raise RuntimeError("No batches processed in evaluate_persistence_model.")
    
    frame_rmse = frame_rmse_sum / batch_count
    frame_ssim = frame_ssim_sum / batch_count
    frame_bias = frame_bias_sum / batch_count
    
    return frame_rmse, frame_ssim, frame_bias


# =========================
# MAIN EVALUATION
# =========================
def main():
    Config.DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("UNIFIED MODEL EVALUATION")
    print(f"Device: {Config.DEVICE}")
    print("Using scikit-image SSIM implementation")
    print("=" * 80)

    valid_csv_path = "indices_2019_training.csv"
    valid_start_indices = load_valid_indices(valid_csv_path, input_frames=4, future_frames=16)

    all_results = []
 
    # Evaluate for each prediction horizon
    for future_hours in [1, 2, 3, 4]:
        print(f"\n{'='*80}")
        print(f"EVALUATING {future_hours}h PREDICTION HORIZON")
        print(f"{'='*80}")
        
        # ✅ FIX: Recreate dataset with correct offset
        Config.FUTURE_OFFSET_HOURS = future_hours
        #base_dataset = WeatherDataset(Config.HRIT_PATH)
        base_dataset = WeatherDataset("/mnt/hdd1/weather4cast_new/data_sample/hrit/2019/roxi_0006.train.reflbt0.ns.h5")
        dataset = FilteredWeatherDataset(base_dataset, valid_start_indices)
        loader = DataLoader(
            dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        # -----------------------------
        # Evaluate ConvGRU Model
        # -----------------------------
        model_path = f"models_gru/conv_gru_model_{future_hours}h.pth"
        if os.path.exists(model_path):
            print(f"\n🔵 Evaluating ConvGRU model...")
            model = ConvGRUModel(input_channels=Config.INPUT_CHANNELS)
            state = torch.load(model_path, map_location=Config.DEVICE)
            model.load_state_dict(state)
            
            frame_rmse, frame_ssim, frame_bias = evaluate_convgru_model(model, loader, future_hours)
            
            # Store frame-wise results
            for i, (rmse, ssim_val, bias_val) in enumerate(zip(frame_rmse, frame_ssim, frame_bias), 1):
                global_index = (future_hours - 1) * Config.FRAMES_PER_HOUR + i
                all_results.append({
                    "Model": "ConvGRU",
                    "Future Hours": future_hours,
                    "Frame Index": i,
                    "Global Frame Index": global_index,
                    "RMSE": float(rmse),
                    "SSIM": float(ssim_val),
                    "Bias": float(bias_val)
                })
            
            # Average for this horizon
            avg_rmse = float(np.mean(frame_rmse))
            avg_ssim = float(np.mean(frame_ssim))
            avg_bias = float(np.mean(frame_bias))
            all_results.append({
                "Model": "ConvGRU",
                "Future Hours": future_hours,
                "Frame Index": "Average",
                "Global Frame Index": "",
                "RMSE": avg_rmse,
                "SSIM": avg_ssim,
                "Bias": avg_bias
            })
            
            print(f"  ConvGRU {future_hours}h - Avg: RMSE={avg_rmse:.6f}, SSIM={avg_ssim:.4f}, Bias={avg_bias:.6f}")
        else:
            print(f"\n❌ ConvGRU model for {future_hours}h not found, skipping...")
        
        # -----------------------------
        # Evaluate Persistence Baseline
        # -----------------------------
        print(f"\n🟡 Evaluating Persistence baseline...")
        
        # ✅ FIX: Recreate dataset again for persistence (ensures fresh iterator)
        Config.FUTURE_OFFSET_HOURS = future_hours
        # base_dataset = WeatherDataset(Config.HRIT_PATH)
        base_dataset = WeatherDataset("/mnt/hdd1/weather4cast_new/data_sample/hrit/2019/roxi_0006.train.reflbt0.ns.h5")
        dataset = FilteredWeatherDataset(base_dataset, valid_start_indices)
        loader = DataLoader(
            dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        frame_rmse, frame_ssim, frame_bias = evaluate_persistence_model(loader, future_hours)
        
        # Store frame-wise results
        for i, (rmse, ssim_val, bias_val) in enumerate(zip(frame_rmse, frame_ssim, frame_bias), 1):
            global_index = (future_hours - 1) * Config.FRAMES_PER_HOUR + i
            all_results.append({
                "Model": "Persistence",
                "Future Hours": future_hours,
                "Frame Index": i,
                "Global Frame Index": global_index,
                "RMSE": float(rmse),
                "SSIM": float(ssim_val),
                "Bias": float(bias_val)
            })
        
        # Average for this horizon
        avg_rmse = float(np.mean(frame_rmse))
        avg_ssim = float(np.mean(frame_ssim))
        avg_bias = float(np.mean(frame_bias))
        all_results.append({
            "Model": "Persistence",
            "Future Hours": future_hours,
            "Frame Index": "Average",
            "Global Frame Index": "",
            "RMSE": avg_rmse,
            "SSIM": avg_ssim,
            "Bias": avg_bias
        })
        
        print(f"  Persistence {future_hours}h - Avg: RMSE={avg_rmse:.6f}, SSIM={avg_ssim:.4f}, Bias={avg_bias:.6f}")

    # -----------------------------
    # Calculate Overall Averages
    # -----------------------------
    print(f"\n{'='*80}")
    print("CALCULATING OVERALL AVERAGES")
    print(f"{'='*80}")
    
    df = pd.DataFrame(all_results)
    
    for model_name in ["ConvGRU", "Persistence"]:
        model_data = df[(df["Model"] == model_name) & (df["Frame Index"] != "Average")]
        if len(model_data) > 0:
            overall_rmse = model_data["RMSE"].mean()
            overall_ssim = model_data["SSIM"].mean()
            overall_bias = model_data["Bias"].mean()
            
            all_results.append({
                "Model": model_name,
                "Future Hours": "All",
                "Frame Index": "Overall Average",
                "Global Frame Index": "",
                "RMSE": overall_rmse,
                "SSIM": overall_ssim,
                "Bias": overall_bias
            })
            
            print(f"\n{model_name} Overall: RMSE={overall_rmse:.6f}, SSIM={overall_ssim:.4f}, Bias={overall_bias:.6f}")

    # -----------------------------
    # Save Results to CSV
    # -----------------------------
    df = pd.DataFrame(all_results)
    output_csv = "evaluation_results_2019_roxi_0006.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*80}")
    print(f"📊 Results saved to: {output_csv}")
    print(f"{'='*80}")
    
    # Display summary
    print("\n📈 SUMMARY TABLE:")
    print("-" * 80)
    summary = df[df["Frame Index"] == "Overall Average"][["Model", "RMSE", "SSIM", "Bias"]]
    print(summary.to_string(index=False))
    print("-" * 80)
    


if __name__ == "__main__":
    main()