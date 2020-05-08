# TKO Lab

## Fraud Detection

This is the CML port of the prototype which is part of the [Deep Learning for Anomaly Detection](https://ff12.fastforwardlabs.com/) report from Cloudera Fast Forward Labs.

## Auto Deploy

To build all the project artifacts, run the `0_cdsw-build.py` file in a Python3 session.

## Model Definition

This fraud detection model is defined as an AutoEncoder with [PyTorch](https://pytorch.org/). Here is the definition code:

```
class autoencoder(nn.Module):
    def __init__(self,num_input):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_input, 15),
            nn.ReLU(True),
            nn.Linear(15, 7))
        self.decoder = nn.Sequential(
            nn.Linear(7, 15),
            nn.ReLU(True),
            nn.Linear(15, num_input),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

## Model Training

A trained model comes with the project, and can be directly used if you only need to present a demo. If you wish to retrain the model, use `1_fraud-model-train.ipynb`.

## Model View

If you wish to get a quick view of the trained model, use `3_fraud-model-view.ipynb`.
