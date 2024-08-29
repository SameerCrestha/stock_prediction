import torch.nn as nn
import torch
torch.manual_seed(42)
class NETWORK_SINGLE_LSTM(nn.Module):

    def __init__(self,num_classes,input_size,hidden_size,num_layers,device='cpu'):
        super(NETWORK_SINGLE_LSTM,self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm=nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,dropout=0.3 if num_layers>1 else 0).to(device)
        self.fc=nn.Linear(hidden_size,num_classes)
        self.device=device
    
    def forward(self,x):
        output_lstm, (h_n,c_n) = self.lstm(x)
        input_fc=output_lstm[:,-1,:]
        output_fc=self.fc(input_fc).to(self.device)
        return output_fc