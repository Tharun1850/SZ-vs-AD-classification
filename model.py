import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from torch.nn import Dropout



class LSTM(nn.Module):
        def __init__(self, config, hidden_dim, num_layers, output_dim, dropout_p=0.5):
            super(LSTM, self).__init__()
            self.config = config
            self.hidden_dim = config.hidden_dim
            self.num_layers = config.num_layers
            self.dropout_p = dropout_p
            self.lstm = nn.LSTM (self.hidden_dim, self.num_layers,batch_first=True, dropout=dropout_p)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.dropout = Dropout(self.dropout_p)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_(True)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_(True)
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.dropout(out)
            out = self.fc(out)
            out = torch.softmax(out, dim=1)
            return out
'''
# Set the hyperparameter values to search over
def params_dict(self,config):
    self.config = config
    params = {'lr': [config.lr],
                'max_epochs': [config.max_epochs],
                'optimizer': [config.optimizer]
             }
    return params
        # Create the NeuralNetClassifier
    
def nnc(self,config):
    net = NeuralNetClassifier(LSTM(hidden_dim=config.hidden_dim,
                                    output_dim=config.output_dim,
                                    num_layers=config.num_layers),
                              max_epochs=params_dict["max_epochs"],
                              lr=params_dict["lr"],
                              optimizer = params_dict['optimizer'])          
            
'''


