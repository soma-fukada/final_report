import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L

# 1. モデル（ニューラルネットワーク）の定義
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

# 2. LightningModuleの定義（学習ループの中身）
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# 3. データセットの準備と学習の実行
def main():
    # データセットのダウンロードと読み込み
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset, batch_size=32) # num_workersはエラー回避のためデフォルト(0)にします

    # モデルの初期化
    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    # トレーナーの定義（ここでは動作確認用にエポック数を1に制限しています）
    trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    
    # 学習開始
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

if __name__ == "__main__":
    main()