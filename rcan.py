from torch import nn

NUM_RESIDUAL_GROUPS = 8
NUM_RESIDUAL_BLOCKS = 16
KERNEL_SIZE = 3
REDUCTION_RATIO = 16
NUM_CHANNELS = 64
UPSCALE_FACTOR = 4

class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, num_channels=NUM_CHANNELS, reduction_ratio=REDUCTION_RATIO, kernel_size=KERNEL_SIZE):
        
        super(ResidualChannelAttentionBlock, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels//reduction_ratio, kernel_size=1, stride=1),
            nn.ReLU(),
            # nn.BatchNorm2d(num_channels//reduction_ratio),
            nn.Conv2d(num_channels//reduction_ratio, num_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        block_input = x.clone()

        residual = self.feature_extractor(x) # Feature extraction
        rescale = self.channel_attention(residual) # Rescaling vector
        
        block_output = block_input + (residual * rescale)
        
        return block_output

class ResidualGroup(nn.Module):
    def __init__(self, num_residual_blocks=NUM_RESIDUAL_BLOCKS,
                 num_channels=NUM_CHANNELS, reduction_ratio=REDUCTION_RATIO, kernel_size=KERNEL_SIZE):
        
        super(ResidualGroup, self).__init__()

        self.residual_blocks = nn.Sequential(
            *[ResidualChannelAttentionBlock(num_channels=num_channels, reduction_ratio=reduction_ratio, kernel_size=kernel_size) 
              for _ in range(num_residual_blocks)]
        )

        self.final_conv = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x):
        group_input = x.clone()

        residual = self.residual_blocks(x) # Residual blocks
        residual = self.final_conv(residual) # Final convolution

        group_output = group_input + residual

        return group_output

class ResidualInResidual(nn.Module):
    def __init__(self, num_residual_groups=NUM_RESIDUAL_GROUPS, num_residual_blocks=NUM_RESIDUAL_BLOCKS,
                 num_channels=NUM_CHANNELS, reduction_ratio=REDUCTION_RATIO, kernel_size=KERNEL_SIZE):
        
        super(ResidualInResidual, self).__init__()

        self.residual_groups = nn.Sequential(
            *[ResidualGroup(num_residual_blocks=num_residual_blocks,
                            num_channels=num_channels, reduction_ratio=reduction_ratio, kernel_size=kernel_size) 
              for _ in range(num_residual_groups)]
        )

        self.final_conv = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x):
        shallow_feature = x.clone()
        
        residual = self.residual_groups(x) # Residual groups
        residual = self.final_conv(residual) # Final convolution

        deep_feature = shallow_feature + residual

        return deep_feature

class RCAN(nn.Module):
    def __init__(self, num_residual_groups=NUM_RESIDUAL_GROUPS, num_residual_blocks=NUM_RESIDUAL_BLOCKS,
                 num_channels=NUM_CHANNELS, reduction_ratio=REDUCTION_RATIO, kernel_size=KERNEL_SIZE):
        
        super(RCAN, self).__init__()

        self.shallow_conv = nn.Conv2d(3, num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.residual_in_residual = ResidualInResidual(num_residual_groups=num_residual_groups, num_residual_blocks=num_residual_blocks,
                                                       num_channels=num_channels, reduction_ratio=reduction_ratio, kernel_size=kernel_size)
        self.upscaling_module = nn.PixelShuffle(upscale_factor=UPSCALE_FACTOR)
        self.reconstruction_conv = nn.Conv2d(num_channels // (UPSCALE_FACTOR ** 2), 3, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x):
        shallow_feature = self.shallow_conv(x) # Initial convolution
        deep_feature = self.residual_in_residual(shallow_feature) # Residual in Residual
        upscaled_image = self.upscaling_module(deep_feature) # Upscaling module
        reconstructed_image = self.reconstruction_conv(upscaled_image) # Reconstruction

        return reconstructed_image
