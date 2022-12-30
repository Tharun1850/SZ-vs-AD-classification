import argparse
from train import trainer
from utils import load_data, preprocess,split_data

def main(config):
    if config.mode == 'train':
        train_data = load_data(config.mode, config.random_seed)
        X,y = preprocess(train_data)
        x_train, x_val, y_train, y_val = split_data(X,y) 
        training =trainer(config,config.input_dim, config.hidden_dim, config.num_layers, config.output_dim, config.lr,config.max_epochs, config.optimizer, x_train, y_train, x_val, y_val)
        training.train(config,x_train, y_train)

    else:
        test_data = load_data(config.mode, config.random_seed)
        X,y = preprocess_data(test_data)
        
        training.inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Classification')
    parser.add_argument('--input_dim', type=int, default=53, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--lr', type=float, default=0.00, choices= [0.001,0.005, 0.01, 0.1],help='Learning rate')
    parser.add_argument('--r', type=int, default=0, choices= [0,1,2,3,4])
    parser.add_argument('--s', type=int, default=1, choices= [0,1,2,3,4])  
    parser.add_argument('--max_epochs', type=int, default=80, help='Number of epochs' ,choices= [80,100,120])
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--mode', type=str, default='train', help='Mode', choices= ['train','test'])
    parser.add_argument('--model_path', type=str, default='', help='Path to saved model')
    parser.add_argument('--random_seed', type=int, default=42, help='setting a random seed')

    
    
    config = parser.parse_args()

main(config)



'''args = parser.parse_args()

input_dim = args.input_dim
hidden_dim = args.hidden_dim
num_layers = args.num_layers
output_dim = args.output_dim
lr = args.lr
max_epochs = args.max_epochs
optimizer = args.optimizer
r = args.r
s = args.s '''



