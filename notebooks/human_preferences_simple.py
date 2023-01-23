from typing import List, Tuple
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader

def U(x, coefficients):
    coefficients /= np.linalg.norm(coefficients, ord = 1)
    return sum(coefficients * x) % 2 - 1

class LitModel(pl.LightningModule):

    """ PyTorch Lightning model.
    Outputs the probability that model U(a,b) is in bin i.
    
    Args:
        input_features (int): Number of input features of each of the two inputs.
        output_predictions (int): Number of output prediction bins.
        hidden_dim (int): Number of hidden units in the hidden layer.
        layers (int): Number of hidden layers.
    """

    def __init__(self, input_features, output_predictions, hidden_dim=128, layers = 3):
        self.input_features = input_features
        self.output_predictions = output_predictions
        self.hidden_dim = hidden_dim
        self.layers = layers
        super().__init__()

        self.initial = nn.Sequential(
            nn.Linear(2*self.input_features, self.hidden_dim),
            nn.ReLU()
        )

        self.backbone_block = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_predictions),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.initial(x)
        for i in range(self.layers):
            x = self.backbone_block(x)
        return self.head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.kl_div(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

def create_dataloader(x_list: list, y_list: list):
    tensor_x = torch.Tensor(np.array(x_list)) # transform to torch tensor
    tensor_y = torch.Tensor(np.array(y_list))
    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    return DataLoader(my_dataset, num_workers = 4) # create your dataloader

def create_predict_dataloader(x_list: list):
    tensor_x = torch.Tensor(np.array(x_list)) # transform to torch tensor
    return DataLoader(tensor_x, num_workers = 4) # create your dataloader

def generate_examples(num_examples, num_features, num_bins):
    """Generates examples of human preferences
    If we decide to use a binary loss function, it is sufficient with num_bins = 2.
    """

    # Generate random coefficients
    coefficients = np.random.uniform(-1, 1, num_features)

    # Generate random inputs
    x0 = np.random.uniform(-1, 1, (num_examples, num_features))
    x1 = np.random.uniform(-1, 1, (num_examples, num_features))

    # Compute the utility of each input
    u = np.array([U(x0[i], coefficients) - U(x1[i], coefficients) for i in range(num_examples)])

    # Compute the bin of each input
    bins = np.array([np.digitize(u[i], np.linspace(-1, 1, num_bins-1)) for i in range(num_examples)])

    # Create the input list
    x_list = []
    for i in range(num_examples):
        x_list.append(np.concatenate((x0[i], x1[i])))

    # Create the output list
    y_list = []
    for i in range(num_examples):
        y = np.zeros(num_bins)
        y[bins[i]] = 1
        y_list.append(y)

    return x_list, y_list

def C(alpha: float, x_list: torch.Tensor):
    loader = DataLoader(torch.Tensor(x_list))
    predictions = trainer.predict(model,loader)
    p = []
    for prediction in predictions:
        prediction = torch.flatten(prediction)
        p.append(torch.where(prediction > alpha, torch.ones_like(prediction), torch.zeros_like(prediction)))
    return torch.stack(p)

num_examples = 1000
num_features = 3
num_bins = 20
x_list, y_list = generate_examples(num_examples = num_examples, num_features = num_features, num_bins = num_bins)
train_loader = create_dataloader(x_list, y_list)
predict_loader = create_predict_dataloader(x_list)
trainer = pl.Trainer(max_epochs=5)
model = LitModel(input_features=num_features, output_predictions=num_bins)

trainer.fit(model, train_dataloaders=train_loader)

a = C(0.5, x_list)
