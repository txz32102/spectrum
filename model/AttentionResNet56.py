import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, 1)
        self.conv2 = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size, stride=stride, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(out_channels // 4, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels // 4)
        self.bn3 = nn.BatchNorm1d(out_channels // 4)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1, stride=stride) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        out = F.elu(self.bn1(x))
        out = self.conv1(out)
        out = F.elu(self.bn2(out))
        out = self.conv2(out)
        out = F.elu(self.bn3(out))
        out = self.conv3(out)
        return out + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, encoder_depth=1):
        super(AttentionBlock, self).__init__()
        self.encoder_depth = encoder_depth
        self.trunk_branch = nn.Sequential(*[ResidualBlock(in_channels, in_channels) for _ in range(2)])
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        for i in range(encoder_depth):
            self.encoder.append(nn.Sequential(
                nn.MaxPool1d(2, 2),
                ResidualBlock(in_channels, in_channels)
            ))
            if i < encoder_depth - 1:
                self.skip_connections.append(ResidualBlock(in_channels, in_channels))
                self.decoder.append(nn.Sequential(
                    ResidualBlock(in_channels, in_channels),
                    nn.Upsample(scale_factor=2)
                ))
        
        self.output_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.Conv1d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        trunk_output = self.trunk_branch(x)
        
        encoder_output = x
        skip_outputs = []
        for i, encoder_layer in enumerate(self.encoder):
            encoder_output = encoder_layer(encoder_output)
            if i < self.encoder_depth - 1:
                skip_outputs.append(self.skip_connections[i](encoder_output))
        
        for i, decoder_layer in enumerate(self.decoder):
            encoder_output = decoder_layer(encoder_output)
            encoder_output = encoder_output + skip_outputs[-(i+1)]
        
        mask = self.output_conv(encoder_output)
        
        if mask.size(2) != trunk_output.size(2):
            mask = F.interpolate(mask, size=trunk_output.size(2), mode='linear', align_corners=False)
        
        output = (1 + mask) * trunk_output
        return output

class AnglePredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super(AnglePredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize the last layer with small weights
        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x):
        # Assuming x has shape (batch_size, 2, 1)
        batch_size = x.shape[0]
        
        # Reshape input: (batch_size, 2, 1) -> (batch_size, 2)
        x = x.view(batch_size, -1)
        
        # Pass through the network
        output = self.network(x)
        
        # Reshape output: (batch_size, 2) -> (batch_size, 2, 1)
        output = output.view(batch_size, 2, 1)
        
        return output  # Shape: (batch_size, 2, 1)


class AttentionResNet56(nn.Module):
    def __init__(self, input_shape=(32768, 2), n_channels=64, dropout=0):
        super(AttentionResNet56, self).__init__()
        self.conv1 = nn.Conv1d(input_shape[1], n_channels, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)
        
        self.res_block1 = ResidualBlock(n_channels, n_channels * 4)
        self.attention1 = AttentionBlock(n_channels * 4, n_channels * 4, encoder_depth=3)
        
        self.res_block2 = ResidualBlock(n_channels * 4, n_channels * 8, stride=2)
        self.attention2 = AttentionBlock(n_channels * 8, n_channels * 8, encoder_depth=2)
        
        self.res_block3 = ResidualBlock(n_channels * 8, n_channels * 16, stride=2)
        self.attention3 = AttentionBlock(n_channels * 16, n_channels * 16, encoder_depth=1)
        
        self.res_block4 = ResidualBlock(n_channels * 16, n_channels * 32, stride=2)
        self.res_block5 = ResidualBlock(n_channels * 32, n_channels * 32)
        self.res_block6 = ResidualBlock(n_channels * 32, n_channels * 32)
        
        self.conv2 = nn.Conv1d(n_channels * 32, 16, 9, padding=4)
        self.conv3 = nn.Conv1d(16, 2, 3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(dropout)
        self.angle_predictor = AnglePredictor()

    def forward(self, x):
        x = self.pool(F.elu(self.bn1(self.conv1(x))))
        
        x = self.res_block1(x)
        x = self.attention1(x)
        
        x = self.res_block2(x)
        x = self.attention2(x)
        
        x = self.res_block3(x)
        x = self.attention3(x)
        
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        
        x = F.elu(self.conv2(x))
        x = self.conv3(x)
        x = self.avg_pool(x)
        
        # Ensure x has shape (batch_size, 2, 1) before passing to angle_predictor
        x = x.view(-1, 2, 1)
        
        x = self.angle_predictor(x)
        
        return x


