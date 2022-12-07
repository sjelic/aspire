import torch
from torch import nn


class RapidEClassifier(nn.Module):
    def __init__(self, number_of_classes=2, dropout_rate=0.5, name=None, type='Classifier'):
        super(RapidEClassifier, self).__init__()
        self.number_of_classes = number_of_classes
        self.dropout_rate = dropout_rate
        self.features = ["Scatter", "Spectrum", "Lifetime 1", "Lifetime 2", "Size"]
        self.name = name
        self.type = type

        self.scatterConv1 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(1, 10, 5), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )

        self.batchNormScatter = nn.BatchNorm2d(10)

        self.scatterConv2 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(10, 20, 3), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )

        self.spectrumnConv1 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(1, 50, 5), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )

        self.batchNormSpectrum = nn.BatchNorm2d(50)

        self.spectrumnConv2 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(50, 100, 3), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )

        self.lifetimeConv1 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, 70, (1, 7)), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )

        self.batchNormLifetime1 = nn.BatchNorm2d(70)

        self.lifetimeConv2 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(70, 140, (1, 5)), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )

        self.batchNormLifetime2 = nn.BatchNorm2d(140)

        self.lifetimeConv3 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(140, 200, 3), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )

        # FC layers

        self.batchNormFCScatter = nn.BatchNorm1d(3000)
        self.batchNormFCSpectrum = nn.BatchNorm1d(800)
        self.batchNormFCLifetime = nn.BatchNorm1d(400)

        self.FCScatter = nn.Sequential(
            nn.Linear(3000, 50), nn.ReLU(), nn.Dropout2d(dropout_rate),
        )
        self.FCSpectrum = nn.Sequential(
            nn.Linear(800, 50), nn.ReLU(), nn.Dropout2d(dropout_rate),
        )
        self.FCLifetime1 = nn.Sequential(
            nn.Linear(400, 50), nn.ReLU(), nn.Dropout2d(dropout_rate),

        )

        self.FCLifetime2 = nn.Sequential(
            nn.ReLU(), nn.Dropout2d(dropout_rate)
        )
        self.FCSize = nn.Sequential(
            nn.ReLU(), nn.Dropout2d(dropout_rate)
        )

        self.batchNormFinal = nn.BatchNorm1d(155)

        self.FCFinal = nn.Sequential(
            nn.Linear(155, number_of_classes), nn.ReLU(),
            nn.Softmax(dim=1)

        )


    

    def forward(self, sample):  # red: spec, scat, life1, life2, size
        scatter = self.scatterConv1(sample['Scatter'])
        scatter = self.batchNormScatter(scatter)
        scatter = self.scatterConv2(scatter)

        spectrum = self.spectrumnConv1(sample['Spectrum'])
        spectrum = self.batchNormSpectrum(spectrum)
        spectrum = self.spectrumnConv2(spectrum)

        lifetime1 = self.lifetimeConv1(sample['Lifetime 1'])
        lifetime1 = self.batchNormLifetime1(lifetime1)
        lifetime1 = self.lifetimeConv2(lifetime1)
        lifetime1 = self.batchNormLifetime2(lifetime1)
        lifetime1 = self.lifetimeConv3(lifetime1)

        scatter = scatter.view(-1, 3000)

        spectrum = spectrum.view(-1, 800)

        lifetime1 = lifetime1.view(-1, 400)

        scatter = self.FCScatter(scatter)
        spectrum = self.FCSpectrum(spectrum)
        lifetime1 = self.FCLifetime1(lifetime1)
        lifetime2 = self.FCLifetime2(sample['Lifetime 2'])
        size = self.FCSize(sample['Size'])

        features = torch.cat((scatter, spectrum, lifetime1, lifetime2, size), dim=1)
        output = self.FCFinal(features)

        return output
