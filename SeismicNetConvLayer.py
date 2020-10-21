
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pool):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size = 1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size = 1),
            ConvBlock(red_3x3, out_3x3, kernel_size = 3, padding = 1)
        )
        
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size = 1),
            ConvBlock(red_5x5, out_5x5, kernel_size = 5, padding = 2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride = 1, padding = 1),
            ConvBlock(in_channels, out_1x1_pool, kernel_size = 1)
        )
        
        
    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )
    
class SeismicNet(nn.Module):
    def __init__(self):
        super(SeismicNet, self).__init__()
        self.conv1 = ConvBlock(3, 192, kernel_size = 2, stride = 2)
        self.inception_1a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception_1b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding = 1)
        
        self.averagepool1 = nn.AvgPool1d(kernel_size= 7, stride= 1)
        self.dropout = nn.Dropout2d(p = 0.15)
        self.fc1 = nn.Linear(3360, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.inception_1a(x)
        x = self.inception_1b(x)
        x = self.maxpool1(x)

        x = self.averagepool1(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x