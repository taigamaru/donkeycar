import pytorch_lightning as pl
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
from donkeycar.parts.pytorch.torch_data import get_default_transform

from torchmetrics import MeanSquaredError

class default_n_linear(nn.Module):
    def __init__(self):
        super(default_n_linear, self).__init__()
        drop = 0.2
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2,padding=0),
            nn.ReLU(),
            nn.Dropout(drop))
        self.layer2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=5, stride=2,padding=0),
            nn.ReLU(),
            nn.Dropout(drop))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2,padding=0),
            nn.ReLU(),
            nn.Dropout(drop))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=0),
            nn.ReLU(),
            nn.Dropout(drop))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=0),
            nn.ReLU(),
            nn.Dropout(drop))
        self.fc1 = nn.Linear(21*21*64, 50)
        self.dropout1 = nn.Dropout(p=drop)
        self.fc2 = nn.Linear(50, 2)
        self.dropout2 = nn.Dropout(p=drop)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1) #Flatten\
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        #x = F.relu(self.fc2(x))
        #x = self.dropout2(x)
        x = self.fc2(x)
        return x

def load_linear(num_classes=2):
    model = default_n_linear()
    return model
    
class Linear(pl.LightningModule):
    def __init__(self, input_shape=(128, 3, 224, 224), output_size=(2,)):
        super().__init__()

        # Used by PyTorch Lightning to print an example model summary
        self.example_input_array = torch.rand(input_shape)

        # Metrics
        self.train_mse = MeanSquaredError()
        self.valid_mse = MeanSquaredError()

        #self.model = load_resnet18(num_classes=output_size[0])
        self.model = load_linear(num_classes=output_size[0])
        

        self.inference_transform = get_default_transform(for_inference=True)

        # Keep track of the loss history. This is useful for writing tests
        self.loss_history = []

        
    def forward(self, x):
        # Forward defines the prediction/inference action
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        #loss = F.l1_loss(logits, y)
        loss = F.mse_loss(logits,y)
        self.loss_history.append(loss)
        self.log("train_loss", loss)

        # Log Metrics
        self.train_mse(logits, y)
        self.log("train_mse", self.train_mse, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        #loss = F.l1_loss(logits,y)
        loss = F.mse_loss(logits, y)

        self.log("val_loss", loss)

        # Log Metrics
        self.valid_mse(logits, y)
        self.log("valid_mse", self.valid_mse, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        #optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0005)
        optimizer = optim.Adam(self.model.parameters())
        return optimizer

    def run(self, img_arr: np.ndarray, other_arr: np.ndarray = None):
        """
        Donkeycar parts interface to run the part in the loop.

        :param img_arr:     uint8 [0,255] numpy array with image data
        :param other_arr:   numpy array of additional data to be used in the
                            pilot, like IMU array for the IMU model or a
                            state vector in the Behavioural model
        :return:            tuple of (angle, throttle)
        """
        from PIL import Image

        pil_image = Image.fromarray(img_arr)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        tensor_image = self.inference_transform(pil_image)
        tensor_image = tensor_image.unsqueeze(0)
        tensor_image = tensor_image.to(device)
        
        #mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        #std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        #tensor_image = transforms.functional.to_tensor(pil_image).to(device)
        #tensor_image.sub_(mean[:, None, None]).div_(std[:, None, None])
        #tensor_image = tensor_image[None, ...]
        #tensor_image =  tensor_image.half()

        
        # Result is (1, 2)
        #result = self.forward(tensor_image)
        result = self.forward(tensor_image).detach().cpu().numpy().flatten()

        # Resize to (2,)
        result = result.reshape(-1)

        # Convert from being normalized between [0, 1] to being between [-1, 1]
        result = result * 2 - 1
        print("ResNet18 result[angle throttle]: {}".format(result))
        return result
