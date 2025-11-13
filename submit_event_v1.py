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
from scipy import ndimage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the output folder
OUTPUT_DIR = '/mnt/cai-data/Weather4Cast/Submissions_events_v1'
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

            frame[frame<=0]=300
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
# MODEL ARCHITECTURE (from training script)
# =========================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=bias)

    def forward(self, x, hidden_state):
        h_cur, c_cur = hidden_state
        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return (h, c)


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=3, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            self.cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size))

    def forward(self, x):
        B, T, C, H, W = x.size()
        h, c = [], []
        for i in range(self.num_layers):
            h_i, c_i = self.cell_list[i].init_hidden(B, (H, W), x.device)
            h.append(h_i)
            c.append(c_i)
        
        outputs = []
        for t in range(T):
            inp = x[:, t, :, :, :]
            for i, cell in enumerate(self.cell_list):
                h[i], c[i] = cell(inp, (h[i], c[i]))
                inp = h[i]
            outputs.append(h[-1])
        outputs = torch.stack(outputs, dim=1)
        return outputs, (h, c)


class ConvLSTMModel(nn.Module):
    def __init__(self, input_channels=1, hidden_dims=[32, 64], kernel_size=3, num_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        self.convlstm = ConvLSTM(32, hidden_dims, kernel_size, num_layers)

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

        lstm_out, _ = self.convlstm(x)
        last_output = lstm_out[:, -1, :, :, :]  # last time step

        pred = self.decoder(last_output)
        pred = pred.unsqueeze(1).repeat(1, 4, 1, 1, 1)  # shape: [B, 4, 1, H, W]
        return pred


def load_all_models():
    model_paths = [
        "models_new/conv_lstm_model_1h.pth",
        "models_new/conv_lstm_model_2h.pth",
        "models_new/conv_lstm_model_3h.pth",
        "models_new/conv_lstm_model_4h.pth",
    ]
    
    models = []
    for path in model_paths:
        print(f"Loading {path} ...")
        # Instantiate model architecture
        model = ConvLSTMModel(input_channels=1, hidden_dims=[32, 64], kernel_size=3, num_layers=2)
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


def generate_predictions(models, input_seq, device):
    """
    Generate 16 frames of predictions from 4 input frames
    Returns predictions in original 252x252 space
    """
    # Preprocess same as training (scale + Otsu + pad)
    input_tensor = preprocess_input(input_seq)  # (4, 256, 256)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(2).to(device)  # (1, 4, 1, 256, 256)

    preds = []
    with torch.no_grad():
        for model in models:
            pred_seq = model(input_tensor)  # (1, 4, 1, 256, 256)
            preds.append(pred_seq)

    all_preds = torch.cat(preds, dim=1)  # (1, 16, 1, 256, 256)
    
    # Postprocess: unpad and scale back to original range
    all_preds = all_preds.squeeze(0)  # (16, 1, 256, 256)
    all_preds_unscaled = postprocess_output(all_preds).squeeze(1)  # (16, 252, 252)
    
    return all_preds_unscaled


def transform_to_rain_rate(predictions):
    """
    Transform predictions to rain rate and resample to 1512x1512
    Returns: (16, 1512, 1512) array of rain rates
    """
    transformed_frames = []
    for i in range(predictions.shape[0]):
        resampled = resample_image(predictions[i], target_shape=(1512, 1512))
        # Transform to rain rate
        transformed = np.maximum(0, 0.42 * (280 - resampled))
        transformed_frames.append(transformed)
    
    return np.stack(transformed_frames, axis=0)


def find_3d_events(rain_volume, threshold=1.0, top_k=5):
    """
    Find top-k 3D rain events in the volume
    
    Args:
        rain_volume: (T, H, W) array of rain rates
        threshold: minimum rain rate to consider
        top_k: number of top events to return
    
    Returns:
        List of events, each event is a dict with features
    """
    # Apply threshold
    binary_volume = (rain_volume >= threshold).astype(np.uint8)
    
    # Define 18-connectivity structure (face + edge neighbors, no corners)
    # In 3D: 6 face neighbors + 12 edge neighbors = 18
    struct = ndimage.generate_binary_structure(3, 2)  # rank 3, connectivity 2 gives 18-connectivity
    
    # Label connected components
    labeled_volume, num_features = ndimage.label(binary_volume, structure=struct)
    
    #if num_features == 0:
    #    return []
    
    # Find properties of each component
    events = []
    for label_id in range(1, num_features + 1):
        event_mask = (labeled_volume == label_id)
        event_values = rain_volume[event_mask]
        
        # Maximum rain rate
        max_rr = np.max(event_values)
        
        # Temporal extent
        frames_with_event = np.where(np.any(event_mask, axis=(1, 2)))[0]
        start_frame = frames_with_event[0]
        end_frame = frames_with_event[-1]
        duration = len(frames_with_event)  # number of frames spanned
        
        # Middle frame
        middle_frame = (start_frame + end_frame) // 2
        
        # Get 2D slice at middle frame
        middle_slice = event_mask[middle_frame, :, :]
        
        # Find bounding box in middle frame
        y_coords, x_coords = np.where(middle_slice)
        
        if len(y_coords) == 0:
            # Shouldn't happen, but handle edge case
            continue
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Calculate diagonal of bounding box
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        diagonal = np.sqrt(width**2 + height**2)
        
        # Calculate center at middle frame
        mid_x = (x_min + x_max) / 2.0
        mid_y = (y_min + y_max) / 2.0
        
        events.append({
            'max_rr': max_rr,
            'start_frame': start_frame,
            'duration': duration,
            'diagonal': diagonal,
            'mid_x': mid_x,
            'mid_y': mid_y
        })
    
    # Sort by max_rr (descending) and take top k
    events.sort(key=lambda e: e['max_rr'], reverse=True)
    top_events = events[:top_k]
    
    # Pad with dummy events if we have fewer than k
    while len(top_events) < top_k:
        # Add a minimal event
        top_events.append({
            'max_rr': 0.01,  # Small positive value
            'start_frame': 1,
            'duration': 1,
            'diagonal': 1.0,
            'mid_x': 0.0,
            'mid_y': 0.0
        })
    
    return top_events


def extract_features(event):
    """
    Extract features from an event and return as dict
    """
    features = {
        'maxPrec.ln': np.log(event['max_rr']),
        'duration.ln': np.log(event['duration']),
        'start.offset.ln': np.log(event['start_frame'] + 1),  # +1 to handle frame 0
        'mid.diag.ln': np.log(event['diagonal']),
        'mid.x': int(np.round(event['mid_x'])),
        'mid.y': int(np.round(event['mid_y']))
    }
    return features


def process_file(models, device, year, file_number):
    year_suffix = str(year)[-2:]
    padded_file_number = f"{file_number:04d}"

    input_hrit = f"{DATA_DIR}/{year}/HRIT/roxi_{padded_file_number}.ev1test{year_suffix}.reflbt0.ns.h5"
    output_csv_dir = f"{OUTPUT_DIR}/{year}/"
    os.makedirs(output_csv_dir, exist_ok=True)
    output_csv = os.path.join(output_csv_dir, f"roxi_{padded_file_number}.test.events1.csv")

    # Load HRIT
    with h5py.File(input_hrit, 'r') as f:
        img = f['REFL-BT'][:]
        img = img[:, 5, :, :]  # Use only channel 5
        img = np.nan_to_num(img, nan=300.0, posinf=300.0, neginf=300.0)
        print(f"Loaded {input_hrit} with shape {img.shape}")

    num_frames = img.shape[0]
    num_cases = num_frames // 4  # Non-overlapping 4-frame windows
    
    # Process each case (4-frame window)
    results = []
    for case_idx in range(num_cases):
        case_id = f"{case_idx + 1:02d}"  # Format as "01", "02", etc.
        
        # Extract 4-frame input window
        start_idx = case_idx * 4
        end_idx = start_idx + 4
        input_seq = img[start_idx:end_idx]  # shape (4, H, W)
        
        print(f"Processing case {case_id}: frames {start_idx}-{end_idx-1}")
        
        # Generate 16 predictions
        predictions = generate_predictions(models, input_seq, device)  # (16, 252, 252)
        
        # Transform to rain rate in 1512x1512 space
        rain_volume = transform_to_rain_rate(predictions)  # (16, 1512, 1512)
        
        # Find top 5 events
        events = find_3d_events(rain_volume, threshold=2, top_k=5)
        
        # Extract features for each event
        for event_num, event in enumerate(events, start=1):
            features = extract_features(event)
            
            # Add to results in the required format
            for feature_name, feature_value in features.items():
                results.append([case_id, event_num, feature_name, feature_value, 1])
    
    # Save output CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

    print(f"Finished {padded_file_number} ({year}) → {output_csv}")
    print(f"Generated {len(results)} rows for {num_cases} cases")


years = [2019, 2020]
file_numbers = [8, 9, 10]

for year in years:
    for file_number in file_numbers:
        process_file(models, device, year, file_number)

print("Processing complete.")