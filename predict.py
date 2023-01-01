import argparse
import torch
import train
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import preprocess, load_data

acc_trn=[]
sens_trn=[]
spec_trn=[]
acc_val=[]
sens_val=[]
spec_val=[]

def predict_data(x, y, mode, best_model, config):
    print("reached predicitons")

    predictions = best_model.predict(x.float())
    print ('Accuracy Score on'+mode+" : ", accuracy_score(y, predictions))
    predictions = (predictions > 0.5).astype(int)
    ground_truth = (y > 0.5).to(torch.int32)
    cm = confusion_matrix(ground_truth, predictions)
    tp = cm[1, 1]
    fp = cm[0, 1]
    tn = cm[0, 0]
    fn = cm[1, 0]

    print(f"Sensitivity on" +mode+" : ",{(tp / (tp + fn))})
    print(f"Specificity on" +mode+" : ",{(tn / (tn + fp))})

    if mode == 'train':
        acc_trn.append(accuracy_score(y, predictions))
        sens_trn.append(tp / (tp + fn))
        spec_trn.append(tn / (tn + fp))
    else:
        acc_val.append(accuracy_score(y, predictions))
        sens_val.append(tp / (tp + fn))
        spec_val.append(tn / (tn + fp))
        

'''
# Load the trained model
with open('best_model'+str(config.r)+ str(config.s)+'.pkl', 'rb') as f:
        best_model = pickle.load(f)
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Classification')
    parser.add_argument('--input_dim', type=int, default=53, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--lr', type=float, default=0.1, choices= [0.001,0.005, 0.01, 0.1],help='Learning rate')
    parser.add_argument('--r', type=int, default=0, choices= [0,1,2,3,4])
    parser.add_argument('--s', type=int, default=1, choices= [0,1,2,3,4])  
    parser.add_argument('--max_epochs', type=int, default=20, help='Number of epochs' ,choices= [80,100,120])
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--mode', type=str, default='train', help='Mode', choices= ['train','test'])
    parser.add_argument('--model_path', type=str, default='', help='Path to saved model')
    parser.add_argument('--random_seed', type=int, default=42, help='setting a random seed')


    config = parser.parse_args()


    
        

    



