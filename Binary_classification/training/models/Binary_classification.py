import torch.nn as nn

class Binary_classification(nn.Module):
    def __init__(self, latent, 
                 input_depth, input_height, input_width
                ):
        super(Binary_classification, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),

            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),

            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True)
        )

        FINAL_FLATTEN_SIZE = 16 * 1 * 10 * 10 # 仮の値

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FINAL_FLATTEN_SIZE, latent),
            nn.ReLU(True), 
            nn.Linear(latent, 1),
            nn.Sigmoid()
        )

    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.classifier(x)
    #     return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier[0](x) # Flatten
        x = self.classifier[1](x) # Linear(..., latent) -> ここが欲しい値
        
        latent_out = self.classifier[2](x)    # ReLU
        x = self.classifier[3](latent_out)    # Linear(latent, 1)
        out = self.classifier[4](x)           # Sigmoid

        return out, latent_out


class Binary_classification_2(nn.Module):
    def __init__(self, latent, 
                 input_depth, input_height, input_width
                ):
        super(Binary_classification_2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

        FINAL_FLATTEN_SIZE = 16 * 1 * 10 * 10 # 仮の値

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FINAL_FLATTEN_SIZE, latent),
            nn.ReLU(), 
            nn.Linear(latent, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier[0](x) # Flatten
        x = self.classifier[1](x) # Linear(..., latent) -> ここが欲しい値
        
        latent_out = self.classifier[2](x)    # ReLU
        x = self.classifier[3](latent_out)    # Linear(latent, 1)
        out = self.classifier[4](x)           # Sigmoid

        return out, latent_out


class Binary_classification_3(nn.Module):
    def __init__(self, latent, 
                 input_depth, input_height, input_width
                ):
        super(Binary_classification_3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),

            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),

            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=0),
            nn.GELU()
        )

        FINAL_FLATTEN_SIZE = 16 * 1 * 10 * 10 # 仮の値

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FINAL_FLATTEN_SIZE, latent),
            nn.ReLU(), 
            nn.Linear(latent, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier[0](x) # Flatten
        x = self.classifier[1](x) # Linear(..., latent) -> ここが欲しい値
        
        latent_out = self.classifier[2](x)    # ReLU
        x = self.classifier[3](latent_out)    # Linear(latent, 1)
        out = self.classifier[4](x)           # Sigmoid

        return out, latent_out

class Binary_classification_Bubble(nn.Module):
    def __init__(self, latent):
        super(Binary_classification_Bubble, self).__init__()
        # 入力: (Batch, 1, 30, 100, 100)
        self.features = nn.Sequential(
            # 速度軸(D)は縮小せず、空間軸(H,W)のみ半分にする
            nn.Conv3d(1, 16, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)), # -> (16, 30, 50, 50)
            nn.BatchNorm3d(16),
            nn.GELU(),
            # 速度軸を少し縮小、空間軸も半分に
            nn.Conv3d(16, 32, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)), # -> (32, 15, 25, 25)
            nn.BatchNorm3d(32),
            nn.GELU(),
            # 畳み込みで解像度を調整
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)), # -> (64, 7, 13, 13)
            nn.BatchNorm3d(64),
            nn.GELU(),
            # 最終的な特徴マップのサイズを大まかに絞る
            nn.Conv3d(64, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0)), # -> (16, 3, 6, 6)
            nn.BatchNorm3d(16),
            nn.GELU()
        )
        # 16チャネル * D(3) * H(6) * W(6) = 1728
        FINAL_FLATTEN_SIZE = 16 * 3 * 6 * 6 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FINAL_FLATTEN_SIZE, latent),
            nn.ReLU(), 
            nn.Linear(latent, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.features(x)
        # 各レイヤーの出力を個別に取得する構造に対応
        x_flat = self.classifier[0](x) # Flatten
        latent_in = self.classifier[1](x_flat) # Linear
        latent_out = self.classifier[2](latent_in) # ReLU
        out = self.classifier[3](latent_out) # Linear
        out = self.classifier[4](out) # Sigmoid
        return out, latent_out


class Binary_classification_Bubble2(nn.Module):
    def __init__(self, latent):
        super(Binary_classification_Bubble2, self).__init__()

        # 入力: (Batch, 1, 30, 100, 100)
        self.features = nn.Sequential(
            # 速度軸(D)は縮小せず、空間軸(H,W)のみ半分にする
            nn.Conv3d(1, 16, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)), # -> (16, 30, 50, 50)
            nn.BatchNorm3d(16),
            nn.GELU(),

            # 速度軸を少し縮小、空間軸も半分に
            nn.Conv3d(16, 32, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)), # -> (32, 15, 25, 25)
            nn.BatchNorm3d(32),
            nn.GELU(),

            # 畳み込みで解像度を調整
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)), # -> (64, 7, 13, 13)
            nn.BatchNorm3d(64),
            nn.GELU(),

            nn.AdaptiveAvgPool3d(2),

        )

        # 16チャネル * D(3) * H(6) * W(6) = 1728
        # FINAL_FLATTEN_SIZE = 16 * 3 * 6 * 6 
        FINAL_FLATTEN_SIZE = 64 * 2 * 2 * 2


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FINAL_FLATTEN_SIZE, latent),
            nn.GELU(), 
            nn.Linear(latent, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        
        # 各レイヤーの出力を個別に取得する構造に対応
        x_flat = self.classifier[0](x) # Flatten
        latent_in = self.classifier[1](x_flat) # Linear
        latent_out = self.classifier[2](latent_in) # ReLU
        
        out = self.classifier[3](latent_out) # Linear
        out = self.classifier[4](out) # Sigmoid

        return out, latent_out


class Binary_classification_Bubble3(nn.Module):
    def __init__(self, latent):
        super(Binary_classification_Bubble3, self).__init__()

        # 入力: (Batch, 1, 30, 100, 100)
        self.features = nn.Sequential(
            # 速度軸(D)は縮小せず、空間軸(H,W)のみ半分にする
            nn.Conv3d(1, 16, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)), # -> (16, 30, 50, 50)
            nn.BatchNorm3d(16),
            nn.GELU(),

            # 速度軸を少し縮小、空間軸も半分に
            nn.Conv3d(16, 32, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)), # -> (32, 15, 25, 25)
            nn.BatchNorm3d(32),
            nn.GELU(),

            # 畳み込みで解像度を調整
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)), # -> (64, 7, 13, 13)
            nn.BatchNorm3d(64),
            nn.GELU(),

        )

        # 16チャネル * D(3) * H(6) * W(6) = 1728
        FINAL_FLATTEN_SIZE = 64 * 7 * 13 * 13


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FINAL_FLATTEN_SIZE, latent),
            nn.GELU(), 
            nn.Linear(latent, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        
        # 各レイヤーの出力を個別に取得する構造に対応
        x_flat = self.classifier[0](x) # Flatten
        latent_in = self.classifier[1](x_flat) # Linear
        latent_out = self.classifier[2](latent_in) # ReLU
        
        out = self.classifier[3](latent_out) # Linear
        out = self.classifier[4](out) # Sigmoid

        return out, latent_out



class Binary_classification_Dropout(nn.Module):
    def __init__(self, latent, 
                 # input_depth, input_height, input_width
                ):
        super(Binary_classification_Dropout, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),

            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True)
        )

        FINAL_FLATTEN_SIZE = 16 * 1 * 10 * 10 # 仮の値
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FINAL_FLATTEN_SIZE, latent),
            nn.ReLU(True), 
            nn.Linear(latent, 1), 
            nn.Sigmoid(),
            nn.Dropout(0.5),
        )

    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.classifier(x)
    #     return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier[0](x) # Flatten
        x = self.classifier[1](x) # Linear(..., latent) -> ここが欲しい値
        
        latent_out = self.classifier[2](x)    # ReLU
        x = self.classifier[3](latent_out)    # Linear(latent, 1)
        out = self.classifier[4](x)           # Sigmoid
        
        return out, latent_out