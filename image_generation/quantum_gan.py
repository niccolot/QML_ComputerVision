import math
import matplotlib.pyplot as plt
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

batch_size=1
image_size=16
n_qubits = 6  # Total number of qubits / N
n_a_qubits = 1  # Number of ancillary qubits / N_A
q_depth = 16 # Depth of the parameterised quantum circuit / D
n_generators = 8  # Number of subgenerators for the patch method / N_G
lrG = 0.3
lrD = 0.01  
num_iter = 200

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

dataset = datasets.MNIST(
    root="data_mnist",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(dataset, batch_size=batch_size)

indexes0 = []

for idx , (image, label) in enumerate(dataset):
    if label == 0:
        indexes0.append(idx)

dataset0 = Subset(dataset, indexes0)
loader0 = DataLoader(dataset0, batch_size=batch_size, drop_last=True)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(image_size**2, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


dev = qml.device("lightning.qubit", wires=n_qubits)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(noise, weights):

    weights = weights.reshape(q_depth, n_qubits)

    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)

    for i in range(q_depth):
        for y in range(n_qubits):
            qml.RY(weights[i][y], wires=y)

        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])

    return qml.probs(wires=list(range(n_qubits)))



def partial_measure(noise, weights):
    probs = quantum_circuit(noise, weights)
    probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    probsgiven0 /= torch.sum(probs)

    probsgiven = probsgiven0 / torch.max(probsgiven0)
    return probsgiven


class PatchQuantumGenerator(nn.Module):

    def __init__(self, n_generators, q_delta=1):

        super().__init__()

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators

    def forward(self, x):
        patch_size = 2 ** (n_qubits - n_a_qubits)

        images = torch.Tensor(x.size(0), 0).to(device)

        for params in self.q_params:

            patches = torch.Tensor(0, patch_size).to(device)
            for elem in x:
                q_out = partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            images = torch.cat((images, patches), 1)

        return images


discriminator = Discriminator().to(device)
generator = PatchQuantumGenerator(n_generators).to(device)

criterion = nn.BCELoss()

optD = optim.SGD(discriminator.parameters(), lr=lrD)
optG = optim.SGD(generator.parameters(), lr=lrG)

real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

fixed_noise = torch.rand(image_size, n_qubits, device=device) * math.pi / 2

counter = 0
d_losses = []
g_losses = []


while True:
    for i, (data, _) in enumerate(loader0):

        data = data.reshape(-1, image_size * image_size)
        real_data = data.to(device)

        noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
        fake_data = generator(noise)

        discriminator.zero_grad()
        outD_real = discriminator(real_data).view(-1)
        outD_fake = discriminator(fake_data.detach()).view(-1)

        errD_real = criterion(outD_real, real_labels)
        errD_fake = criterion(outD_fake, fake_labels)
        errD_real.backward()
        errD_fake.backward()

        errD = (errD_real + errD_fake)/2
        optD.step()

        generator.zero_grad()
        outD_fake = discriminator(fake_data).view(-1)
        errG = criterion(outD_fake, real_labels)
        errG.backward()
        optG.step()

        counter += 1

        d_losses.append(errD.item())
        g_losses.append(errG.item())
        
        if counter % 10 == 0:
            print(f'Iteration: {counter}, D_Loss: {errD:0.3f}, G_Loss: {errG:0.3f}')

        if counter == num_iter:
            break
    if counter == num_iter:
        break

plt.plot(d_losses, label='d_loss')
plt.plot(g_losses, label='g_loss')
plt.legend()


test_images = generator(fixed_noise).view(-1,1,image_size,image_size).cpu().detach()
test_images = torch.squeeze(test_images, dim=1)
test_images = test_images[:8]

fig, axs = plt.subplots(2, 4)
for j, im in enumerate(test_images):
    row = j // 4
    col = j % 4
    image_np = test_images[j].detach().numpy()
    axs[row, col].imshow(image_np, cmap='gray')
plt.show()

