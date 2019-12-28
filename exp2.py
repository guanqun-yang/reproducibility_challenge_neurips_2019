import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import argparse
import pickle

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

NUM_CLASSES = 10

LR = 1e-4
BATCH_SIZE = 64
MAX_EPOCH = 80
PRINT_FREQ = 500
DATA_PATH = "./data/CIFAR10/"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# utility function 
def make_prediction(score):
    return torch.topk(score.detach(), k=1, dim=1).indices.cpu().squeeze().numpy()

class GradientReverse(torch.autograd.Function):
    scale = 1.0
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()
    

def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)

class PAR(nn.Module):
    def __init__(self, inchannel, scale):
        super(PAR, self).__init__()
        self.scale = scale

        self.conv = nn.Conv2d(inchannel, NUM_CLASSES, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = grad_reverse(x, scale=self.scale)
        x = self.conv(x)

        return x

# penalizing higher conv layer output (second conv layer)
class PARH(nn.Module):
    def __init__(self, inchannel, scale=1.0):
        super(PARH, self).__init__()
        self.scale = scale

        # (64, 7, 7)
        self.conv = nn.Conv2d(inchannel, NUM_CLASSES, kernel_size=1, stride=1, padding=0)
        # (NUM_CLASSES, 7, 7)
    
    def forward(self, x):
        x = grad_reverse(x, scale=self.scale)
        x = self.conv(x)

        return x

# more powerful 
class PARM(nn.Module):
    def __init__(self, inchannel, input_size, scale=1.0):
        super(PARM, self).__init__()
        self.scale = scale
        self.input_size = input_size

        self.rep = nn.Sequential(nn.Conv2d(inchannel, 100, kernel_size=3, stride=1, padding=1),
                                 nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(100, 50),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(50, NUM_CLASSES))

    
    def forward(self, x):
        x = grad_reverse(x, scale=self.scale)
        x = self.rep(x)
        x = torch.flatten(x, start_dim=-2, end_dim=-1).permute(0, 2, 1)
        x = self.classifier(x)
        output = x.permute(0, 2, 1).view(-1, NUM_CLASSES, self.input_size, self.input_size)
        return output


# broader penalizing region
class PARB(nn.Module):
    def __init__(self, inchannel, scale=1.0):
        super(PARB, self).__init__()
        self.scale = scale

        self.conv = nn.Conv2d(inchannel, NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    
    def forward(self, x):
        x = grad_reverse(x, scale=self.scale)
        x = self.conv(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResidualBlock, self).__init__()

        self.pad_size = None
        stride = None
        shortcut = None
        if inchannel * 2 == outchannel:
            stride = 2
            self.pad_size = inchannel // 2
            # note the channels have to be padded with 0's to match the conv result
            shortcut = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        elif inchannel == outchannel:
            stride = 1
            shortcut = None
        else:
            raise ValueError("invalid inchannel and outchannel pair")

        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        if self.right == None:
            residual = x
        else:
            # note the channel has to be padded with 0's
            residual = F.pad(self.right(x), pad=(0, 0, 0, 0, self.pad_size, self.pad_size))
    
        out += residual
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, par_type, scale=1.0):
        super(ResNet, self).__init__()
        self.par_type = par_type

        # (3, 32, 32)
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(num_features=16),
                                    nn.ReLU(inplace=True))
        # (16, 32, 32)
        self.layer2 = self._make_layer(16, 16, 5)
        # (16, 32, 32)
        self.layer3 = self._make_layer(16, 32, 5)
        # (32, 16, 16)
        self.layer4 = self._make_layer(32, 64, 5)
        # (64, 8, 8)
    
        self.output_layer = nn.Sequential(nn.BatchNorm2d(num_features=64),
                                          nn.ReLU(inplace=True),
                                          nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                          nn.Flatten(start_dim=1, end_dim=-1),
                                          nn.Linear(64, NUM_CLASSES))
        if par_type == "PAR":
            self.adv = PAR(inchannel=16, scale=scale)
        elif par_type == "PARM":
            self.adv = PARM(inchannel=16, input_size=32, scale=scale)
        elif par_type == "PARB":
            self.adv = PARB(inchannel=16, scale=scale)
        elif par_type == "PARH":
            self.adv = PARH(inchannel=16, scale=scale)
        else:
            self.adv = None
        

    def _make_layer(self, inchannel, outchannel, block_num):
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel))

        for _ in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer1(x)
        x_adv1 = x.clone()
        x = self.layer2(x)
        x_adv2 = x.clone()
        x = self.layer3(x)
        x = self.layer4(x)

        output = self.output_layer(x)
        # adv output
        if self.par_type in ["PAR", "PARB", "PARM"]:
            adv_output = self.adv(x_adv1)
        elif self.par_type == "PARH":
            adv_output = self.adv(x_adv2)
        elif self.par_type == "vanilla":
            adv_output = None
        else:
            raise ValueError("incorret adversarial training mode")
        
        return output, adv_output

def main(args):
    input_size = 32

    # 0 - testData.npy, 1 - testData_greyscale.npy, 2 - testData_negative.npy, 3 - testData_radiokernel.npy, 4 - testData_randomkernel.npy
    val_data_path = None
    if args.test_case is 0:
        val_data_path = "testData.npy"
    elif args.test_case is 1:
        val_data_path = "testData_greyscale.npy"
    elif args.test_case is 2:
        val_data_path = "testData_negative.npy"
    elif args.test_case is 3:
        val_data_path = "testData_radiokernel.npy"
    elif args.test_case is 4:
        val_data_path = "testData_randomkernel.npy"
    else:
        raise ValueError("unsupported test data")
    
    X_train = torch.tensor(np.load(os.path.join(DATA_PATH, "trainData.npy"))).permute((0, 3, 1, 2))
    y_train = torch.tensor(np.load(os.path.join(DATA_PATH, "trainLabel.npy")))
    X_val = torch.tensor(np.load(os.path.join(DATA_PATH, val_data_path))).permute((0, 3, 1, 2))
    y_val = torch.tensor(np.load(os.path.join(DATA_PATH, "testLabel.npy")))

    train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, pin_memory=True)
    val_dataloader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, pin_memory=True)

    num_train_sample = X_train.size(0)
    # model
    model = ResNet(par_type=args.par_type, scale=args.adv_strength).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    # training loop
    print("training loop...")
    train_loss_list = list()
    train_acc_list = list()
    val_loss_list = list()
    val_acc_list = list()
    percent = 0.0
    for epoch in range(MAX_EPOCH):
        print("epoch %d / %d" % (epoch+1, MAX_EPOCH))
        model.train()
        # training loop
        temp_loss_list = list()
        for i, (X_train, y_train) in enumerate(train_dataloader):
            X_train = X_train.type(torch.float32).to(device)
            y_train = y_train.type(torch.int64).to(device)

            optimizer.zero_grad()

            score, adv_score = model(X_train)
            # vanilla model
            if adv_score is None:
                loss = criterion(input=score, target=y_train)
            # PAR and its variants
            else:
                # apply adversarial training at 250/400 = 0.625 of MAX_EPOCH
                if epoch <= 0.625 * MAX_EPOCH:
                    loss = criterion(input=score, target=y_train)
                else:
                    loss = criterion(input=score, target=y_train) + \
                           criterion(input=adv_score.permute(0, 2, 3, 1).reshape(-1, 10), target=y_train.repeat(input_size * input_size))
            loss.backward()

            optimizer.step()

            temp_loss_list.append(loss.detach().cpu().numpy())
            percent += BATCH_SIZE / (num_train_sample * MAX_EPOCH) * 100
            if (i + 1) % PRINT_FREQ == 0:
                print("\tloss: %.5f, progress: %.2f%%" % (temp_loss_list[-1], percent))
        
        temp_loss_list = list()
        y_train_list = list()
        y_train_pred_list = list()
        for X_train, y_train in train_dataloader:
            X_train = X_train.type(torch.float32).to(device)
            y_train = y_train.type(torch.int64).to(device)

            score, _ = model(X_train)
            loss = criterion(input=score, target=y_train)

            temp_loss_list.append(loss.detach().cpu().numpy())
            y_train_list.extend(y_train.detach().cpu().numpy())
            y_train_pred_list.extend(make_prediction(score))
        
        train_acc_list.append(accuracy_score(y_true=y_train_list, y_pred=y_train_pred_list))
        train_loss_list.append(np.average(temp_loss_list))

        # validation
        model.eval()
        
        temp_loss_list = list()
        y_val_list = list()
        y_val_pred_list = list()
        for X_val, y_val in val_dataloader:
            X_val = X_val.type(torch.float32).to(device)
            y_val = y_val.type(torch.int64).to(device)

            score, _ = model(X_val)
            loss = criterion(input=score, target=y_val)

            temp_loss_list.append(loss.detach().cpu().numpy())
            y_val_list.extend(y_val.detach().cpu().numpy())
            y_val_pred_list.extend(make_prediction(score))
        
        val_acc_list.append(accuracy_score(y_true=y_val_list, y_pred=y_val_pred_list))
        val_loss_list.append(np.average(temp_loss_list))

        print("\ttrain loss: %.5f, train acc: %.5f" % (train_loss_list[-1], train_acc_list[-1]))
        print("\tval loss: %.5f, val acc: %.5f" % (val_loss_list[-1], val_acc_list[-1]))

    # test
    model.eval()
    y_test_list = []
    y_pred_list = []
    for X_test, y_test in val_dataloader:
        X_test = X_test.type(torch.float32).to(device)
        # label
        y_test_list.extend(y_test.detach().cpu().numpy())

        # prediction
        score, _ = model(X_test)
        y_pred_list.extend(make_prediction(score))
    
    test_accuracy = accuracy_score(y_true=y_test_list, y_pred=y_pred_list)
    test_result = classification_report(y_true=y_test_list, y_pred=y_pred_list)
    filename = "EXP2_" + args.par_type + "_" + str(args.test_case)
    print(test_accuracy)
    print(test_result)
    result_dict = {"train_loss": train_loss_list, 
                   "val_loss": val_loss_list,
                   "train_acc": train_acc_list,
                   "val_acc": val_acc_list,
                   "test_acc": test_accuracy,
                   "report": test_result}
    with open(filename + ".pickle", "wb") as file_object:
        pickle.dump(result_dict, file_object)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 0 - testData.npy, 1 - testData_greyscale.npy, 2 - testData_negative.npy, 3 - testData_radiokernel.npy, 4 - testData_randomkernel.npy 
    parser.add_argument("--test_case", type=int, default=0)
    # PAR, PARH (higher), PARB (broader), PARM (more complex)
    parser.add_argument("--par_type", type=str, default="PAR")
    # adversarial strength (lambda)
    parser.add_argument("--adv_strength", type=float, default=1.0)
    
    args = parser.parse_args()
    print("parameter")
    for param, value in vars(args).items():
        print("\t{}: {}".format(param, str(value)))
    
    # train, val, and test 
    main(args=args)