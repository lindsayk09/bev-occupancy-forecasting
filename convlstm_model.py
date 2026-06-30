"""
convlstm_model.py
-----------------
ConvLSTM-based model for temporal BEV occupancy forecasting.

Task: given T=3 consecutive BEV occupancy grids, predict T+1.

Architecture:
  Input [B, T, 1, H, W]
       ↓
  ConvLSTM encoder (2 layers)
       ↓
  Final hidden state [B, hidden, H, W]
       ↓
  Conv decoder → predicted grid [B, 1, H, W]
       ↓
  Sigmoid → occupancy probability [0,1]

Evaluated with:
  - IoU (Intersection over Union) — primary metric
  - Binary cross-entropy loss — training objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ── ConvLSTM cell ─────────────────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell. Processes one timestep.
    Gates are computed with convolutions instead of linear layers,
    preserving spatial structure.
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels

        # All 4 gates in one conv (input, forget, output, cell)
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, h: int, w: int, device):
        zeros = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        return zeros, zeros


# ── ConvLSTM stack ────────────────────────────────────────────────────────────

class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM that processes a sequence of BEV grids.
    Returns the hidden state from the last timestep of the last layer.
    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int = 2, kernel_size: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()

        for i in range(num_layers):
            ic = in_channels if i == 0 else hidden_channels
            self.cells.append(ConvLSTMCell(ic, hidden_channels, kernel_size))

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, C, H, W]
        Returns last hidden state: [B, hidden, H, W]
        """
        B, T, C, H, W = x.shape
        device = x.device

        # Initialise hidden/cell states for each layer
        states = [cell.init_hidden(B, H, W, device) for cell in self.cells]

        for t in range(T):
            inp = x[:, t]          # [B, C, H, W]
            for i, cell in enumerate(self.cells):
                h, c = states[i]
                h, c = cell(inp, h, c)
                states[i] = (h, c)
                inp = h            # next layer input = current hidden state

        return states[-1][0]       # last layer's final hidden state


# ── Full forecasting model ────────────────────────────────────────────────────

class BEVForecastNet(nn.Module):
    """
    Full BEV occupancy forecasting network.

    Encoder: 2-layer ConvLSTM processes T past grids
    Decoder: 3-layer conv head predicts next grid
    """
    def __init__(self, hidden_channels: int = 32, num_layers: int = 2):
        super().__init__()
        self.encoder = ConvLSTM(
            in_channels=1,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 1, H, W]
        Returns: [B, 1, H, W] — predicted occupancy probability map
        """
        h = self.encoder(x)
        return self.decoder(h)


# ── Dataset ───────────────────────────────────────────────────────────────────

class BEVSequenceDataset(Dataset):
    """
    Loads sequences of BEV grids from .npy files.

    Each sample: (input_seq, target)
      input_seq: [T, 1, H, W] — T consecutive past grids
      target:    [1, H, W]    — next grid to predict

    Files must be sorted chronologically (they are, by filename).
    """
    def __init__(self, bev_dir: str, seq_len: int = 3, grid_size: int = 128):
        self.files = sorted(Path(bev_dir).glob("*.npy"))
        self.seq_len = seq_len
        self.grid_size = grid_size

        if len(self.files) < seq_len + 1:
            raise ValueError(
                f"Need at least {seq_len + 1} BEV grids, found {len(self.files)}"
            )

        print(f"Dataset: {len(self.files)} grids → "
              f"{len(self.files) - seq_len} samples (seq_len={seq_len})")

    def __len__(self):
        return len(self.files) - self.seq_len

    def __getitem__(self, idx):
        # Load T past grids
        seq = []
        for i in range(self.seq_len):
            grid = np.load(self.files[idx + i]).astype(np.float32)
            # Resize if needed
            if grid.shape[0] != self.grid_size:
                from PIL import Image as PILImage
                grid = np.array(PILImage.fromarray(grid).resize(
                    (self.grid_size, self.grid_size), PILImage.NEAREST))
            seq.append(torch.tensor(grid).unsqueeze(0))   # [1, H, W]

        input_seq = torch.stack(seq, dim=0)                # [T, 1, H, W]

        # Load target (next grid)
        target = np.load(self.files[idx + self.seq_len]).astype(np.float32)
        if target.shape[0] != self.grid_size:
            from PIL import Image as PILImage
            target = np.array(PILImage.fromarray(target).resize(
                (self.grid_size, self.grid_size), PILImage.NEAREST))
        target = torch.tensor(target).unsqueeze(0)         # [1, H, W]

        return input_seq, target


# ── IoU metric ────────────────────────────────────────────────────────────────

def compute_iou(pred: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5) -> float:
    """
    Compute binary IoU between predicted and target BEV grids.

    pred:   [B, 1, H, W] float in [0,1]
    target: [B, 1, H, W] float in {0,1}

    Returns mean IoU over the batch.
    """
    pred_bin   = (pred > threshold).float()
    target_bin = (target > 0.5).float()

    intersection = (pred_bin * target_bin).sum(dim=[1, 2, 3])
    union        = ((pred_bin + target_bin) > 0).float().sum(dim=[1, 2, 3])

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()
