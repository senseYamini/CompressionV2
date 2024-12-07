import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

files = [
    "../../Data/empty1_timedomain.csv",
    "../../Data/jumpAyon_timedomain.csv",
    "../../Data/situpAyon_timedomain.csv",
    "../../Data/standAyon_timedomain.csv",
    "../../Data/walkAyon_timedomain.csv"
]

data = []
labels = []
counter = 0
for file_path in files:
    f = pd.read_csv(file_path, header=None, skiprows=1)
    for i in range(0, len(f), 50):
        data_chunk = f.iloc[i:i+50].to_numpy()
        if len(data_chunk) < 50:  # Check if the chunk is complete
            continue 
        if(np.isnan(data_chunk).any()):
            continue
        data.append(data_chunk)
        labels.append(counter)
    counter += 1
    
for i,chunk in enumerate(data):
    if(chunk.shape[0] != 50 or chunk.shape[1] != 56):
        print(f"inconsistency found for element: {i}")

data_array = np.array(data)
labels_array = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data_array, labels_array, 
                                                    test_size=0.2, random_state=42)

# Check for NaN or Inf in your dataset
print("NaN in data:", np.isnan(X_train).any())
print("Inf in data:", np.isinf(X_train).any())

#transfering training, testing data and labels to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype = torch.long).to(device)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

compression_rate = [450, 225, 112, 45, 22, 11, 7, 5]

for cr in compression_rate:
#defining the autoencoder_10
    class Autoencoder(nn.Module):
        def __init__(self, cr=128):  # Add `cr` as a parameter for the compression ratio
            super(Autoencoder, self).__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, padding=1),  # Output: (48, 50, 56)
                nn.ReLU(),
                nn.MaxPool2d((2, 2), stride=(2, 2), padding=0),  # Output: (48, 25, 28)
                nn.Conv2d(48, 30, kernel_size=3, padding=1),  # Output: (30, 25, 28)
                nn.ReLU(),
                nn.MaxPool2d((2, 2), stride=(2, 2), padding=0),  # Output: (30, 12, 14)
                nn.Conv2d(30, 20, kernel_size=3, padding=1),  # Output: (20, 12, 14)
                nn.ReLU(),
            )

            # Add linear layers
            self.fc1 = nn.Linear(20 * 12 * 14, cr)  # Compression ratio determines bottleneck size
            self.fc2 = nn.Linear(cr, 20 * 12 * 14)

            # Decoder
            self.decoder = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(20, 30, kernel_size=3, padding=1),  # Output: (30, 12, 14)
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),  # Output: (30, 24, 28)
                nn.Conv2d(30, 48, kernel_size=3, padding=1),  # Output: (48, 24, 28)
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),  # Output: (48, 48, 56)
                nn.Conv2d(48, 1, kernel_size=3, padding=1),  # Output: (1, 48, 56)
                nn.Sigmoid(),  # Ensure values are between 0 and 1
                nn.Upsample(size=(50, 56), mode='nearest')  # Output: (1, 50, 56)
            )

        def forward(self, x):
            # Ensure input has a channel dimension
            if len(x.shape) == 3:  # Input shape: [batch_size, 50, 56]
                x = x.unsqueeze(1)  # Add channel dimension -> [batch_size, 1, 50, 56]
            
            # Encoder forward pass
            encoded = self.encoder(x)
            encoded = encoded.view(encoded.size(0), -1)  # Flatten the tensor
            encoded = self.fc1(encoded)  # Apply the linear layer
            
            # Decoder forward pass
            decoded = self.fc2(encoded)  # Apply the reverse linear layer
            decoded = decoded.view(-1, 20, 12, 14)  # Reshape back to the 4D tensor
            decoded = self.decoder(decoded)
            return decoded

        
    autoencoder = Autoencoder().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr = 0.001)

    num_epochs = 100
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    start_time = time.time()
    for epoch in range(num_epochs):
        autoencoder.train()
        running_train_loss = 0.0
        for item, label in train_dataloader:
            # inputs = item.to(device)
            # labels = label.to(device)
            optimizer.zero_grad()
            # item = item.unsqueeze(1)
            outputs = autoencoder(item)
            loss = criterion(outputs, item)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        train_losses.append(running_train_loss / len(train_dataloader))

        # Validation phase
        autoencoder.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                # inputs = inputs.to(device)
                # inputs = item.unsqueeze(1)
                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                running_val_loss += loss.item()
        val_loss = running_val_loss / len(test_dataloader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}')

    end_time = time.time()
    elapsed_time = (end_time - start_time)  # Convert to minutes
    print(f"Training completed in {elapsed_time} seconds")
    print(f'Best Validation Loss at Epoch {best_epoch} with Loss: {best_val_loss:.4f}')