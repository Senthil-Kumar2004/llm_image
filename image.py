from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

dataset = load_dataset("cifar10", split="train")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def preprocess(example):
    example['img'] = transform(example['img'])
    return example

dataset = dataset.map(preprocess)

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = UNet()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

def train(model, dataloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            if isinstance(batch, dict):
                imgs = batch['img'].to(device)
            elif isinstance(batch, list):
                imgs = torch.stack([item['img'] for item in batch]).to(device)
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train(model, dataloader)

torch.save(model.state_dict(), "stable_diffusion_model.pth")

model.load_state_dict(torch.load("stable_diffusion_model.pth"))
model.eval()

def generate_images(model, noise):
    model.eval()
    with torch.no_grad():
        generated_images = model(noise)
    return generated_images

noise = torch.randn(1, 3, 64, 64).to(device)
generated_images = generate_images(model, noise)

import matplotlib.pyplot as plt

generated_images = generated_images.cpu().squeeze(0).permute(1, 2, 0)
plt.imshow(generated_images)
plt.show()
