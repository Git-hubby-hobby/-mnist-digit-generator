
import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10, img_dim=28*28):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        input = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(input)
        return img.view(img.size(0), 1, 28, 28)


@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    model.eval()
    return model

st.title("🧠 Handwritten Digit Generator (0–9)")
digit = st.number_input("Select a digit (0–9)", min_value=0, max_value=9, step=1)

if st.button("Generate 5 Images"):
    G = load_model()
    noise = torch.randn(5, 100)
    labels = torch.tensor([digit]*5)
    with torch.no_grad():
        generated = G(noise, labels).detach()
    grid = make_grid(generated, nrow=5, normalize=True)
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(grid, (1,2,0)))
    ax.axis("off")
    st.pyplot(fig)
