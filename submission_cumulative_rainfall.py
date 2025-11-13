import cv2
import numpy as np
import pandas as pd
import h5py
import csv
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from skimage import filters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the output folder
OUTPUT_DIR = '/mnt/cai-data/Weather4Cast/Submissions_convgru'
DATA_DIR = '/mnt/cai-data/Weather4Cast/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration matching training
INPUT_SCALE = 300.0
USE_OTSU = True
PADDED_SIZE = 256
ORIGINAL_SIZE = 252
PAD_SIZE = (PADDED_SIZE - ORIGINAL_SIZE) // 2

# =========================
# PREPROCESSING FUNCTIONS (from training script)
# =========================
def apply_otsu_with_scaling(image):
    if isinstance(image, np.ndarray):
        image = image.astype(np.float32)
    else:
        image = np.array(image, dtype=np.float32)
    
    scaled_img = (image / INPUT_SCALE)
    if np.max(scaled_img) == np.min(scaled_img):
        return scaled_img, 0.0, scaled_img
    
    thresh = filters.threshold_otsu(scaled_img)
    processed_img = np.copy(scaled_img)
    background_mask = scaled_img >= thresh
    processed_img[background_mask] = 1.0
    return processed_img.astype(np.float32), thresh, scaled_img


def preprocess_input(data):
    """Preprocess input data same as training"""
    if USE_OTSU:
        T, H, W = data.shape
        processed_frames = []
        for t in range(T):
            frame = data[t, :, :]
            frame[frame<=0] = 300
            processed_frame, _, _ = apply_otsu_with_scaling(frame)
            processed_frames.append(processed_frame)
        data = np.stack(processed_frames, axis=0)
    else:
        for t in range(data.shape[0]):
            if np.all(data[t] == 0):
                data[t] = np.full_like(data[t], 300.0)
        data = data / INPUT_SCALE
    
    # Pad to 256x256
    data = torch.tensor(data, dtype=torch.float32)
    data = F.pad(data, (PAD_SIZE, PAD_SIZE, PAD_SIZE, PAD_SIZE), mode='constant', value=0)
    
    return data


def postprocess_output(output_tensor):
    """Reverse preprocessing: unpad and scale back"""
    # Remove padding
    if PAD_SIZE > 0:
        output_tensor = output_tensor[:, :, PAD_SIZE:-PAD_SIZE, PAD_SIZE:-PAD_SIZE]
    
    # Scale back to original range
    output_np = output_tensor.cpu().numpy() * INPUT_SCALE
    
    return output_np


# =========================
# MODEL ARCHITECTURE (ConvGRU from training script)
# =========================
class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        
        # Gates: reset and update
        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim, 
            2 * hidden_dim, 
            kernel_size, 
            padding=padding, 
            bias=bias
        )
        
        # Candidate hidden state
        self.conv_candidate = nn.Conv2d(
            input_dim + hidden_dim, 
            hidden_dim, 
            kernel_size, 
            padding=padding, 
            bias=bias
        )

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv_gates(combined)
        
        # Split into reset and update gates
        r, u = torch.split(gates, self.hidden_dim, dim=1)
        r = torch.sigmoid(r)
        u = torch.sigmoid(u)
        
        # Candidate hidden state
        combined_candidate = torch.cat([x, r * h_prev], dim=1)
        h_candidate = torch.tanh(self.conv_candidate(combined_candidate))
        
        # New hidden state
        h_next = (1 - u) * h_prev + u * h_candidate
        return h_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=device)


class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=3, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList()
        
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            self.cell_list.append(ConvGRUCell(cur_input_dim, hidden_dims[i], kernel_size))

    def forward(self, x):
        B, T, C, H, W = x.size()
        
        # Initialize hidden states
        h = []
        for i in range(self.num_layers):
            h_i = self.cell_list[i].init_hidden(B, (H, W), x.device)
            h.append(h_i)
        
        # Process sequence
        outputs = []
        for t in range(T):
            inp = x[:, t, :, :, :]
            for i, cell in enumerate(self.cell_list):
                h[i] = cell(inp, h[i])
                inp = h[i]
            outputs.append(h[-1])
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, h


class ConvGRUModel(nn.Module):
    def __init__(self, input_channels=1, hidden_dims=[32, 64], kernel_size=3, num_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        self.convgru = ConvGRU(32, hidden_dims, kernel_size, num_layers)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.encoder(x)
        x = x.view(B, T, 32, H, W)

        gru_out, _ = self.convgru(x)
        last_output = gru_out[:, -1, :, :, :]

        pred = self.decoder(last_output)
        pred = pred.unsqueeze(1).repeat(1, 4, 1, 1, 1)  # shape: [B, 4, 1, H, W]
        return pred


def load_all_models():
    model_paths = [
        "models/ConvGRU/conv_gru_model_1h.pth",
        "models/ConvGRU/conv_gru_model_2h.pth",
        "models/ConvGRU/conv_gru_model_3h.pth",
        "models/ConvGRU/conv_gru_model_4h.pth",
    ]
    
    models = []
    for path in model_paths:
        print(f"Loading {path} ...")
        # Instantiate ConvGRU model architecture
        model = ConvGRUModel(input_channels=1, hidden_dims=[32, 64], kernel_size=3, num_layers=2)
        # Load weights
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)
    return models


models = load_all_models()


def resample_image(image, target_shape=(1512, 1512)):
    return cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR)


def process_case(models, img, slot_start, slot_end, x_start, x_end, y_start, y_end, device):
    # Extract 4-frame input sequence (1 hour)
    input_seq = img[slot_start:slot_end]  # shape (4, H, W)
    lf = img[slot_end - 1]
    min_val = np.min(lf[lf > 0])
    
    # Preprocess same as training (scale + Otsu + pad)
    input_tensor = preprocess_input(input_seq)  # (4, 256, 256)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(2).to(device)  # (1, 4, 1, 256, 256)

    preds = []
    with torch.no_grad():
        for model in models:
            pred_seq = model(input_tensor)  # (1, 4, 1, 256, 256)
            preds.append(pred_seq)

    all_preds = torch.cat(preds, dim=1)  # (1, 16, 1, H, W)
    
    # Postprocess: unpad and scale back to original range
    all_preds = all_preds.squeeze(0)  # (16, 1, 256, 256)
    all_preds_unscaled = postprocess_output(all_preds).squeeze(1)  # (16, 252, 252)
    

    # Apply frame-specific min_val scaling
    for i in range(all_preds_unscaled.shape[0]):
        if(np.min(all_preds_unscaled[i,:,:]) > min_val):
            sf = (300 - min_val) / (300. - np.min(all_preds_unscaled[i,:,:]))
            all_preds_unscaled[i,:,:] = sf * (all_preds_unscaled[i,:,:] - 300.) + 300.
    
    transformed_frames = []
    for i in range(all_preds_unscaled.shape[0]):
        resampled = resample_image(all_preds_unscaled[i], target_shape=(1512, 1512))

        transformed = 1.1 * np.power(np.maximum(0, 300 - resampled), 0.15)
        transformed_frames.append(transformed)
    
    # Average all transformed frames
    mean_transformed = np.mean(transformed_frames, axis=0)

    # Crop and compute mean
    box_values = mean_transformed[y_start:y_end, x_start:x_end]
    average_value = box_values.mean()

    return average_value


def process_file(models, device, year, file_number):
    year_suffix = str(year)[-2:]
    padded_file_number = f"{file_number:04d}"

    input_hrit = f"{DATA_DIR}/{year}/HRIT/roxi_{padded_file_number}.cum1test{year_suffix}.reflbt0.ns.h5"
    input_csv = f"/mnt/cai-data/Weather4Cast/roxi_{padded_file_number}.cum1test_dictionary.csv"
    output_csv_dir = f"{OUTPUT_DIR}/{year}/"
    os.makedirs(output_csv_dir, exist_ok=True)
    output_csv = os.path.join(output_csv_dir, f"roxi_{padded_file_number}.test.cum4h.csv")

    # Load HRIT
    with h5py.File(input_hrit, 'r') as f:
        img = f['REFL-BT'][:]
        img = img[:, 5, :, :]  # Use only channel 5
        img = np.nan_to_num(img, nan=300.0, posinf=300.0, neginf=300.0)
        print(f"Loaded {input_hrit} with shape {img.shape}")

    # Read CSV
    df = pd.read_csv(input_csv)

    # Process each case
    results = []
    for _, row in df.iterrows():
        if row['year'] == year:
            case_id = row['Case-id']
            slot_start = row['slot-start']
            slot_end = row['slot-end']
            x_start = row['x-top-left']
            x_end = row['x-bottom-right']
            y_start = row['y-top-left']
            y_end = row['y-bottom-right']

            avg_val = process_case(models, img, slot_start, slot_end, x_start, x_end, y_start, y_end, device)
            if avg_val<=2:
                results.append([case_id, np.round(avg_val, 2), 1])
            else:
                if int(avg_val/4)>2: #4
                    for i in range(0,int(avg_val/4),2): #4
                        results.append([case_id, np.round(i,2), 0.0])   
                        results.append([case_id, np.round(i,2), 0.25])  
                results.append([case_id, np.round(avg_val/4, 2), 0.5]) #4
                results.append([case_id, np.round(avg_val/2, 2), 0.75]) #2
                for i in range(int(np.ceil(avg_val/2)*2),120,2): # 120->100 #1.5
                    results.append([case_id, np.round(i,2), 1])    

    # Save output CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

    print(f"Finished {padded_file_number} ({year}) → {output_csv}")


years = [2019, 2020]
file_numbers = [8, 9, 10]

for year in years:
    for file_number in file_numbers:
        process_file(models, device, year, file_number)

print("Processing complete.")
