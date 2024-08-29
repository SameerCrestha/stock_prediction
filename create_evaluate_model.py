import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
import joblib
torch.manual_seed(42)
from models import NETWORK_SINGLE_LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Gives data for training, validation and testing the model
def get_train_val_test_data():
    rolling_dataset=pd.read_csv('assets/rolling_window_data.csv')
    from helper.concatenated_windows_to_3d_array import concatenated_windows_to_3d_array as cwta
    dataset_array=cwta(rolling_dataset,120)

    from helper.train_val_test_split_3d import train_val_test_split_3d as tvts
    train_set, val_set, test_set = tvts(dataset_array, train_ratio=0.6, val_ratio=0,shuffle=False)
    print("Train set shape: ", train_set.shape)
    print("Validation set shape: ", val_set.shape)
    print("test set shape: ", test_set.shape)

    inp_seq_length=100
    X_train=train_set[:,:inp_seq_length,:]
    y_train=train_set[:,inp_seq_length:,rolling_dataset.columns.tolist().index('Close')]
    X_validate=val_set[:,:inp_seq_length,:]
    y_validate=val_set[:,inp_seq_length:,rolling_dataset.columns.tolist().index('Close')]
    X_test=test_set[:,:inp_seq_length,:]
    y_test=test_set[:,inp_seq_length:,rolling_dataset.columns.tolist().index('Close')]

    X_train = Variable(torch.Tensor(X_train))
    y_train = Variable(torch.Tensor(y_train))
    X_validate = Variable(torch.Tensor(X_validate))
    y_validate = Variable(torch.Tensor(y_validate))
    X_test = Variable(torch.Tensor(X_test))
    y_test = Variable(torch.Tensor(y_test))
    return X_train,y_train,X_validate,y_validate,X_test,y_test

def create_model(params,X_train,y_train):
    #Initializing model with parameters
    num_epochs=params['num_epochs']
    learning_rate=params['learning_rate']
    input_size=X_train.shape[2]
    num_classes=20
    model = NETWORK_SINGLE_LSTM(num_classes,input_size, params['hidden_size'],params['num_layers'],device=device)
    model.to(device)

    # DEFINE OPTIMIZER
    criterion = torch.nn.L1Loss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Lists to store training losses for plotting the training curve
    train_losses = []
    #batch gradient descent
    # TRAIN THE MODEL:
    model.train()
    print("Training in progress...")
    for epoch in range(1,num_epochs+1):
        
        outputs = model(X_train.to(device))
        optimizer.zero_grad()
        # obtain the loss function
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        # if epoch % 10 == 0:
        #     print("Epoch: %d, training loss: %1.5f" % (epoch, loss.item()))
    model.eval()
    print("Training completed.")
    return model

def evaluate_model(model,X_test,y_test):
    rolling_dataset=pd.read_csv('assets/rolling_window_data.csv')
    scaler=joblib.load('assets/scaler.gz')
    scaler.scale_=scaler.scale_[rolling_dataset.columns.get_loc('Close')]
    scaler.mean_=scaler.mean_[rolling_dataset.columns.get_loc('Close')]
    model.eval()
    y_pred=model(X_test.to(device))
    y_pred=y_pred.detach().numpy()*(1/scaler.scale_)+scaler.mean_
    y_test=y_test*(1/scaler.scale_)+scaler.mean_
    from sklearn.metrics import mean_absolute_error
    mae=mean_absolute_error(y_test,y_pred)
    return mae