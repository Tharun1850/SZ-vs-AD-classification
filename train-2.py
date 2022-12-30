from utils import split_data, preprocess, load_data
#from model import LSTM
from skorch import NeuralNetClassifier
from predict import predict_data
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
#from main import config
import pickle
from torch.nn import Dropout
from sklearn.metrics import confusion_matrix
import model


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
        
'''# 

#preprocess the data
dataset = preprocess(dataset)
#split into X_train, y_train, X_val, y_val 
X_train, y_train, X_val, y_val = utils.split_data(dataset)'''

class trainer:
    def __init__(self, config, input_dim,hidden_dim, num_layers, output_dim, lr, max_epochs, optimizer, X_train, y_train, X_val, y_val):
        self.config = config
        print("reached trianer")
        
    def train(self,config, X_train,y_train):
        self.config = config
       # self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        
        params = {'lr': [config.lr],
                'max_epochs': [config.max_epochs],
                'optimizer': [config.optimizer]
             }
       
        Net = NeuralNetClassifier(LSTM(config, config.hidden_dim,
                                       config.output_dim,
                                       config.num_layers),
                              max_epochs=params["max_epochs"],
                              lr=params["lr"],
                              optimizer = params['optimizer'])
        
        # Create the grid search object
        grid_search = GridSearchCV(Net, params, refit=True, verbose=3, cv=10, error_score='raise', n_jobs=-1, scoring="accuracy")
        print("after gs")
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        grid_search.predict_data(X_train.float())
        print("after gs")
        
        best_parameters = grid_search.best_estimator_.get_params()
        #save best model
        '''
        with open('best_model'+str(config.r+ config.s)+'.pkl', 'wb') as f:
            pickle.dump(best_model, f)

        with open('best_model'+str(config.r+ config.s)+'.pkl', 'rb') as f:
             best_model = pickle.load(f)'''

        with open('best_model10.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        with open('best_model10.pkl', 'rb') as f:
            best_model = pickle.load(f)
        print(best_model.params)


        predict_data(X_train.float(), best_model)
        predict_data(X_val.float(), best_model)
        print("passed train module")
 