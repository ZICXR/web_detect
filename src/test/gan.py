import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# =====================
# 超参数
# =====================
batch_size = 64
lr_D = 0.0001
lr_G = 0.0002
latent_dim = 100
epochs = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 数据集 (MNIST)
# =====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# =====================
# 生成器 (Generator)
# =====================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.model(z)
        return x.view(-1, 1, 28, 28)

# =====================
# 判别器 (Discriminator)
# =====================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.model(x)

# 初始化模型
G = Generator().to(device)
D = Discriminator().to(device)

# 损失函数 & 优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr_G)
optimizer_D = optim.Adam(D.parameters(), lr=lr_D)

# =====================
# 训练
# =====================
gen_img = None
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):

        real_imgs = imgs.to(device)
        batch_size_cur = real_imgs.size(0)

        real_labels = torch.ones(batch_size_cur, 1).to(device) * 0.9
        fake_labels = torch.zeros(batch_size_cur, 1).to(device)

        # ---------------------
        # 训练判别器
        # ---------------------
        z = torch.randn(batch_size_cur, latent_dim).to(device)
        fake_imgs = G(z)

        real_loss = criterion(D(real_imgs), real_labels)
        fake_loss = criterion(D(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        # 训练生成器
        # ---------------------
        z = torch.randn(batch_size_cur, latent_dim).to(device)
        fake_imgs = G(z)
        g_loss = criterion(D(fake_imgs), real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if i % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {i} \
                  D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

    
    # 每个epoch生成一张图
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        gen_img = G(z).cpu().squeeze()

plt.imshow(gen_img, cmap="gray")
plt.title(f"Epoch {epoch}")
plt.show()