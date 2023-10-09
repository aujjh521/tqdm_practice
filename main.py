#general
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#scikt learn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#joblib
from tqdm import tqdm
import time
from joblib import Parallel, delayed

#pytorch related
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

#load data
print(f'{"="*20} start load data {"="*20}')
train_data_path = 'OVL_data_for_ML_test_medium_new.xlsx'
test_data_path = 'OVL_data_for_ML_test_medium_exam_new.xlsx'

train_dataset = pd.read_excel(train_data_path)
test_dataset = pd.read_excel(test_data_path)
print(f'train / test dataset are loaded, shape is {train_dataset.shape} / {test_dataset.shape}')

#column define
print(f'{"="*20} start column define {"="*20}')
feature_col = ['PART', 'CUREQP', 'PRE1EQP', 'PRE2EQP', 'RETICLE','PRERETICLE']
val_col = ['Tx_Rn', 'Ty_Rn']
X = train_dataset[feature_col]
y = train_dataset[val_col]

#preprocessing
#label encode & one hot encode
print(f'{"="*20} start label encode & one hot encode {"="*20}')


le_list = []
for col in feature_col:
  le = LabelEncoder()
  X[col] = le.fit_transform(X[col])
  print(f'label encodinf done, {le.classes_}')
  le_list.append(le)
print(f'after label encoding, X is:\n{X.head()}')

ohe = OneHotEncoder()
X = ohe.fit_transform(X).toarray()
print(f'one hot encoder: {ohe.categories_}')
print(f'after one hot encoding, X is:\n{X}')

#get all permutations
print(f'{"="*20} start get all permutations {"="*20}')
index = list(range(len(X)))
permut = itertools.permutations(index,r=2)

#make y to numpy array
y = y.values

X_bias = np.empty((0,X.shape[1]))
y_bias = np.empty((0,y.shape[1]))
print(f'X:{type(X)},y:{type(y)}')
for pair in permut:
  temp_X = X[pair[0]] - X[pair[1]]
  temp_y = y[pair[0]] - y[pair[1]]
  X_bias = np.append(X_bias,temp_X)
  y_bias = np.append(y_bias,temp_y)

X_bias = X_bias.reshape(-1,X.shape[1])
y_bias = y_bias.reshape(-1,y.shape[1])
print(f'資料膨脹完畢')

#train / test split (注意這邊是把X_bias,y_bias拿去拆train test)
print(f'{"="*20} start train / test split {"="*20}')
X_train , X_test , y_train , y_test = train_test_split(X_bias,y_bias , test_size=0.3 , random_state=40)
print(f'train/test split done, X_train , X_test , y_train , y_test size:\n{X_train.shape , X_test.shape , y_train.shape , y_test.shape}')
print(y_train[0:5])

#Standardization (y train 做完scaler之後套給 y test)
print(f'{"="*20} start Standardization {"="*20}')
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
print(f'after standardization, y_train is:\n{y_train[15:25]}')
y_test = scaler.transform(y_test)




#create dataset / dataloader
class FakeDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#defineNN model
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.main = nn.Sequential(
        nn.Linear(n_in, 32),
        nn.ReLU(),

        nn.Linear(32, 64),
        nn.ReLU(),

        nn.Linear(64,n_out)
    )

  def forward(self,x):
    return self.main(x)

def train(dataloader, model, loss_fn, optimizer):
    '每個batch跑完update一次loss'
    size = len(dataloader.dataset)
    model.train()
    #print('switch to .train() mode')
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # losses['training_MAE'].append(loss.item())

def test(dataloader, model, loss_fn):
    '整個epoch跑完update一次loss,每次update的內容是每個batch相加總和再除以batch的數量'
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    #print('switch to .eval() mode')
    test_loss = 0
    with torch.no_grad():
      for X, y in dataloader:
          # Compute prediction error
          pred = model(X)
          test_loss = test_loss + loss_fn(pred, y).item()
    test_loss /= num_batches
    # losses['testing_MAE'].append(test_loss)
    #print(f"Avg loss: {test_loss:>8f}")

def trainNN(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, **kwargs):

  print(f'start train with params {kwargs}')

  #get hyperparameters
  epoch_number = kwargs['epoch_number']
  batch_size = kwargs['batch_size']
  lr = kwargs['lr']

  dataset = FakeDataset(X_train_tensor, y_train_tensor)
  dataloader = DataLoader(dataset, batch_size=batch_size,
                shuffle=True)

  dataset_tset = FakeDataset(X_test_tensor, y_test_tensor)
  dataloader_test = DataLoader(dataset_tset, batch_size=batch_size,
                  shuffle=True)
  
  NN = Net()
  # print(NN)
  loss_func = nn.L1Loss()
  optim = torch.optim.SGD(NN.parameters(), lr=lr)
  # print(optim)

  for epoc in range(epoch_number):
    train(dataloader, NN, loss_func, optim)
    test(dataloader_test, NN, loss_func)



if __name__ == '__main__':

    #create dataset
    X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
    y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
    y_test_tensor = torch.from_numpy(y_test.astype(np.float32))

    #model parameter
    n_in = X_train_tensor.shape[1]
    n_out = y_train_tensor.shape[1]

    #define hyperparameters
    NN_hyperparams = []
    epoch_number_list = np.linspace(50,50,10).tolist()
    epoch_number_list = [int(i) for i in epoch_number_list]
    batch_size_list = np.linspace(5,20,10).tolist()
    batch_size_list = [int(i) for i in batch_size_list]
    lr_list = np.linspace(0.001,0.01,10).tolist()


    for epoch_number, batch_size, lr in zip(epoch_number_list, batch_size_list, lr_list):
        params = {"epoch_number":epoch_number,
                "batch_size": batch_size,
                "lr": lr}
        NN_hyperparams.append(params)

    res = Parallel(n_jobs=2)(delayed(trainNN)(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, **params) for params in tqdm(NN_hyperparams))
