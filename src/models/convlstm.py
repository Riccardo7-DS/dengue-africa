import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, states):
        h_cur, c_cur = states
        combined = torch.cat([x, h_cur], dim=1)  # (B, C+H, H, W)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, shape):
        H, W = shape
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.hidden_dim, H, W, device=device),
            torch.zeros(batch_size, self.hidden_dim, H, W, device=device),
        )


class DengueConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_dim, tabular_dim, tab_emb_dim, out_channels=1):
        super().__init__()
        # ConvLSTM for raster
        self.convlstm = ConvLSTMCell(input_channels + tab_emb_dim, hidden_dim)

        # MLP for tabular (per time step)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Linear(64, tab_emb_dim),
        )

        # Decoder: conv to pixel predictions
        self.decoder = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x_seq, tab_seq):
        """
        x_seq: (B, T, C, H, W)   raster inputs
        tab_seq: (B, T, F_tab)   tabular inputs (time-varying)
        """
        B, T, C, H, W = x_seq.shape
        h, c = self.convlstm.init_hidden(B, (H, W))

        outputs = []
        for t in range(T):
            x_t = x_seq[:, t]  # (B, C, H, W)
            tab_t = tab_seq[:, t]  # (B, F_tab)

            # Encode tabular → embedding
            tab_emb = self.tabular_mlp(tab_t)  # (B, tab_emb_dim)
            # Broadcast to pixels
            tab_emb_2d = tab_emb[:, :, None, None].expand(-1, -1, H, W)

            # Fuse raster + tabular
            x_fused = torch.cat([x_t, tab_emb_2d], dim=1)

            # ConvLSTM update
            h, c = self.convlstm(x_fused, (h, c))

            # Pixel-level prediction
            out = self.decoder(h)  # (B, 1, H, W)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)  # (B, T, 1, H, W)
        return outputs.squeeze(2)  # (B, T, H, W)