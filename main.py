import argparse
from train import trainer
from utils import load_data, preprocess,split_data
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

def main(config):
    if config.mode == 'train':
        train_data = load_data(config, config.mode)
        X,y = preprocess(train_data)
        x_train, x_val, y_train, y_val = split_data(X, y) 
        training =trainer(config,config.input_dim, config.hidden_dim, config.num_layers, config.output_dim, config.lr, config.max_epochs, config.optimizer, x_train, y_train, x_val, y_val)
        training.train(config,x_train, y_train, x_val, y_val)

    else:
        test_data = load_data(config, config.mode)
        X,y = preprocess(test_data)
        
        '''with open('best_model10.pkl', 'rb') as f:
            best_model = pickle.load(f)'''
        with open('best_model'+str(config.r)+str(config.s)+'.pkl', 'rb') as f:
             best_model = pickle.load(f)
                
        predictions = best_model.predict(X.float())
        predictions = (predictions > 0.5).astype(int)
        ground_truth = (y > 0.5).to(torch.int32)
        print ('Accuracy Score on '+config.mode+" : ",accuracy_score(y, predictions))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Classification')
    parser.add_argument('--input_dim', type=int, default=53, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--lr', type=float, default=0.1, choices= [0.001,0.005, 0.01, 0.1],help='Learning rate')
    parser.add_argument('--r', type=int, default=0, choices= [0,1,2,3,4])
    parser.add_argument('--s', type=int, default=1, choices= [0,1,2,3,4])  
    parser.add_argument('--max_epochs', type=int, default=120, help='Number of epochs' ,choices= [80,100,120])
    parser.add_argument('--optimizer', type=str, default='Optim.Adam', help='Optimizer')
    parser.add_argument('--mode', type=str, default='train', help='Mode', choices= ['train','test'])
    parser.add_argument('--model_path', type=str, default='', help='Path to saved model')
    parser.add_argument('--random_seed', type=int, default=42, help='setting a random seed')

    
    
    config = parser.parse_args()

main(config)




