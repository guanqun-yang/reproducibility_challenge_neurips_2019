import torch
import numpy as np
import matplotlib.pyplot as plt

import argparse
import pickle
import gzip

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
MAX_EPOCH = 100
PRINT_FREQ = 500

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def loadMultiDomainMNISTData(testCase=1, mode="independent"):
    '''
    :param testCase:
            0 for original distribution as testing
            1 for random kernel as testing
            2 for radial kernel as testing
    :return:
    '''
    # local functions
    def fft(img):
        return np.fft.fft2(img)
    def fftshift(img):
        return np.fft.fftshift(fft(img))
    def ifft(img):
        return np.fft.ifft2(img)
    def ifftshift(img):
        return ifft(np.fft.ifftshift(img))

    def distance(i,j,w,h,r):
        dis=np.sqrt((i-14)**2+(j-14)**2)
        if dis<r:
            return 0.0
        else:
            return 1.0

    def addingPattern(r, mask):
        fftshift_img_r=fftshift(r)
        fftshift_result_r = fftshift_img_r * mask
        result_r = ifftshift(fftshift_result_r)
        mr=np.abs(result_r)
        return mr

    def mask_radial_MM(isGray=True):  
        mask = np.zeros((28,28)) 
        for i in range(28):
            for j in range(28):
                mask[i,j]=distance(i,j,28,28,r=3.5)
        return mask

    def mask_random_MM(p=0.5, isGray=True):  
        mask=np.random.binomial(1,1-p,(28,28))
        return mask

    # the paramter l is a placeholder here
    def addMultiDomainPatternIndependent(r, l, testCase, testingFlag=False, randomMask=None, radioMask=None):
        if testingFlag:
            if testCase == 0:
                return r
            elif testCase == 1:
                return addingPattern(r, randomMask)
            else:
                return addingPattern(r, radioMask)
        else:
            if np.random.rand() < 0.5:
                k = 1
            else:
                k = 2
            return addMultiDomainPatternIndependent(r, None, int(testCase+k)%3, testingFlag=True, randomMask=randomMask, radioMask=radioMask)
    
    def addMultiDomainPatternDependent(r, l, testCase, testingFlag=False, randomMask=None, radioMask=None):
        if testingFlag:
            if testCase == 0:
                return r
            elif testCase == 1:
                return addingPattern(r, randomMask)
            else:
                return addingPattern(r, radioMask)
        else:
            if l < 5: 
                k = 1
            else:
                k = 2
            return addMultiDomainPatternDependent(r, None, int(testCase+k)%3, testingFlag=True, randomMask=randomMask, radioMask=radioMask)

    # main
    np.random.seed(1)
    with gzip.open("./data/MNIST/mnist.pkl.gz", "rb") as file_object:
        training_data, validation_data, test_data = pickle.load(file_object, encoding="latin1")

    randomMask = mask_random_MM()
    radioMask = mask_radial_MM()

    _Xtrain=np.zeros((training_data[0].shape[0],28*28))
    _Xvalidation = np.zeros((validation_data[0].shape[0],28*28))
    _Xtest=np.zeros((test_data[0].shape[0],28*28))

    # change the behavior of addMultiDomainPattern function based on mode
    if mode == "independent":
        addMultiDomainPattern = addMultiDomainPatternIndependent
    elif mode == "dependent":
        addMultiDomainPattern = addMultiDomainPatternDependent
    else:
        raise ValueError("incorrect mode!")
        
    for i in range(training_data[0].shape[0]):
        r = training_data[0][i]
        r=r.reshape(28,28)
        img = addMultiDomainPattern(r, training_data[1][i], testCase, randomMask=randomMask, radioMask=radioMask)
        _Xtrain[i]=img.reshape(1,28*28)

    for i in range(validation_data[0].shape[0]):
        r = validation_data[0][i]
        r=r.reshape(28,28)
        img = addMultiDomainPattern(r, training_data[1][i], testCase, randomMask=randomMask, radioMask=radioMask)
        _Xvalidation[i]=img.reshape(1,28*28)


    for i in range(test_data[0].shape[0]):
        r = test_data[0][i]
        r=r.reshape(28,28)
        img = addMultiDomainPattern(r, training_data[1][i], testCase, testingFlag=True,randomMask=randomMask, radioMask=radioMask)
        _Xtest[i]=img.reshape(1,28*28)

    indices = np.random.permutation(_Xtrain.shape[0])
    _Xtrain = _Xtrain[indices, :]
    training_label = training_data[1][indices]

    return torch.tensor(_Xtrain.reshape(-1, 1, 28, 28)), torch.tensor(training_label),\
           torch.tensor(_Xvalidation.reshape(-1, 1, 28, 28)), torch.tensor(validation_data[1]),\
           torch.tensor(_Xtest.reshape(-1, 1, 28, 28)), torch.tensor(test_data[1])

# utility function 
def make_prediction(score):
    return torch.topk(score.detach(), k=1, dim=1).indices.cpu().squeeze().numpy()

# model
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
    def __init__(self, inchannel, scale=1.0):
        super(PAR, self).__init__()
        self.scale = scale

        # (32, 14, 14)
        self.conv = nn.Conv2d(inchannel, NUM_CLASSES, kernel_size=1, stride=1, padding=0)
        # (NUM_CLASSES, 14, 14), each channel corresponds to NUM_CLASSES scores

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

class TwoLayerCNN(nn.Module):
    def __init__(self, par_type, scale=1.0):
        super(TwoLayerCNN, self).__init__()
        self.par_type = par_type

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.linear1 = nn.Linear(64*7*7, 1024)
        self.linear2 = nn.Linear(1024, NUM_CLASSES)

        if par_type == "PAR":
            self.adv = PAR(inchannel=32, scale=scale)
        elif par_type == "PARM":
            self.adv = PARM(inchannel=32, input_size=14, scale=scale)
        elif par_type == "PARB":
            self.adv = PARB(inchannel=32, scale=scale)
        elif par_type == "PARH":
            self.adv = PARH(inchannel=64, scale=scale)
        else:
            self.adv = None

    def forward(self, x):
        # (1, 28, 28)
        x = self.pool1(F.relu(self.conv1(x)))
        x_adv1 = x.clone()
        # (32, 14. 14)
        x = self.pool2(F.relu(self.conv2(x)))
        x_adv2 = x.clone()
        # (64, 7, 7)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        # 64*7*7
        x = self.linear1(x)
        # l2 normalization + dropout
        x = F.normalize(x, dim=0, p=2)
        x = F.dropout(x, p=0.5)
        # output
        output = self.linear2(x)
        adv_output = None

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
    # change some parameter
    input_size = 7 if args.par_type == "PARH" else 14

    # data
    # the returned values are all tensors
    print("loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = loadMultiDomainMNISTData(testCase=args.test_case, mode=args.superficial_mode)
    train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    num_train_sample = X_train.shape[0]
    # model
    model = TwoLayerCNN(par_type=args.par_type, scale=args.adv_strength).to(device)
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
                if epoch <= 0.5 * MAX_EPOCH:
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
    for X_test, y_test in test_dataloader:
        X_test = X_test.type(torch.float32).to(device)
        # label
        y_test_list.extend(y_test.detach().cpu().numpy())

        # prediction
        score, _ = model(X_test)
        y_pred_list.extend(make_prediction(score))

    test_accuracy = accuracy_score(y_true=y_test_list, y_pred=y_pred_list)
    test_result = classification_report(y_true=y_test_list, y_pred=y_pred_list)
    filename = "EXP1_" + args.par_type + "_" + str(args.test_case) + "_" + args.superficial_mode
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
    # 0 - original distribution, 1 - random kernel, 2 - radial kernel
    parser.add_argument("--test_case", type=int, default=0)
    # PAR, PARH (higher), PARB (broader), PARM (more complex)
    parser.add_argument("--par_type", type=str, default="PAR")
    # independent, dependent
    parser.add_argument("--superficial_mode", type=str, default="independent")
    # adversarial strength (lambda)
    parser.add_argument("--adv_strength", type=float, default=1.0)
    
    args = parser.parse_args()
    print("parameter")
    for param, value in vars(args).items():
        print("\t{}: {}".format(param, str(value)))
    
    # train, val, and test 
    main(args=args)
