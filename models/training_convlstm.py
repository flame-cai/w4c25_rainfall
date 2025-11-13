import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage import filters

torch.cuda.empty_cache()

# =========================
# CONFIGURATION
# =========================
class Config:
    HRIT_PATH = "data_sample/hrit/boxi_0015.train.reflbt0.ns.h5"

    NUM_PAST_FRAMES = 4        # 1 hour input
    FRAMES_PER_HOUR = 4        # 15 min per frame
    INPUT_CHANNELS = 1         # Using only 5th channel
    CHANNEL_INDEX = 4          # 5th channel (0-indexed)
    INPUT_SCALE = 300.0
    USE_OTSU = True
    BATCH_SIZE = 25
    LR = 1e-3
    EPOCHS = 20
    DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
    
    ORIGINAL_SIZE = 252
    PADDED_SIZE = 256
    PAD_SIZE = (PADDED_SIZE - ORIGINAL_SIZE) // 2
    
    FUTURE_OFFSET_HOURS = 1

# =========================
# PREPROCESSING FUNCTIONS
# =========================
def apply_otsu_with_scaling(image):
    if isinstance(image, np.ndarray):
        image = image.astype(np.float32)
    else:
        image = np.array(image, dtype=np.float32)
    
    scaled_img = (image / Config.INPUT_SCALE)# - 1.0
    if np.max(scaled_img) == np.min(scaled_img):
        return scaled_img, 0.0, scaled_img
    
    thresh = filters.threshold_otsu(scaled_img)
    processed_img = np.copy(scaled_img)
    background_mask = scaled_img >= thresh
    processed_img[background_mask] = 1.0
    return processed_img.astype(np.float32), thresh, scaled_img


def preprocess_hrit(data, channel_idx=4):
    if isinstance(data, np.ndarray):
        data_np = data
    else:
        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else np.array(data)
    
    data_np = data_np[:, channel_idx:channel_idx+1, :, :]  # select only 1 channel

    if Config.USE_OTSU:
        T, C, H, W = data_np.shape
        processed_frames = []
        for t in range(T):
            frame = data_np[t, 0, :, :]
            processed_frame, _, _ = apply_otsu_with_scaling(frame)
            processed_frames.append(processed_frame)
        data_np = np.stack(processed_frames, axis=0)[:, np.newaxis, :, :]
    else:
        data_np = data_np / Config.INPUT_SCALE
    
    data = torch.tensor(data_np, dtype=torch.float32)
    pad_size = Config.PAD_SIZE
    data = F.pad(data, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
    
    return data

# =========================
# DATASET
# =========================
class WeatherDataset(Dataset):
    def __init__(self, hrit_file):
        self.hrit = h5py.File(hrit_file, 'r')['REFL-BT']
        self.num_frames = self.hrit.shape[0]
        self.future_offset = Config.FUTURE_OFFSET_HOURS * Config.FRAMES_PER_HOUR

    def __len__(self):
        return self.num_frames - Config.NUM_PAST_FRAMES - self.future_offset - Config.FRAMES_PER_HOUR + 1

    def __getitem__(self, idx):
        x = self.hrit[idx:idx + Config.NUM_PAST_FRAMES]  # input: 4 frames
        start_output = idx + Config.NUM_PAST_FRAMES + self.future_offset
        y = self.hrit[start_output:start_output + Config.FRAMES_PER_HOUR]  # output: 4 frames

        x = preprocess_hrit(x, channel_idx=Config.CHANNEL_INDEX)
        y = preprocess_hrit(y, channel_idx=Config.CHANNEL_INDEX)

        return x, y

# =========================
# MODEL (ConvLSTM2D Version)
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
        pred = pred.unsqueeze(1).repeat(1, Config.FRAMES_PER_HOUR, 1, 1, 1)  # shape: [B, 4, 1, H, W]
        return pred

# =========================
# TRAINING
# =========================
def train_model(model, train_loader):
    model = model.to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.MSELoss()

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "ConvLSTM/conv_lstm_model_1h.pth")
    print("✅ Model saved as conv_lstm_model.pth")

# =========================
# MAIN
# =========================
def main():
    print(f"Using device: {Config.DEVICE}")
    dataset = WeatherDataset(Config.HRIT_PATH)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=8)

    model = ConvLSTMModel(input_channels=Config.INPUT_CHANNELS)
    train_model(model, loader)

if __name__ == "__main__":
    main()
