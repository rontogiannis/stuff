import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns

from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
seq_field = Field(lower=True, include_lengths=True, batch_first=True)

fields = [('seq', seq_field), ('neut', label_field)]
device = "cuda" if torch.cuda.is_available() else "cpu"

train, valid, test = TabularDataset.splits(path="", train='train.csv', validation='valid.csv', test='test.csv', format='CSV', fields=fields, skip_header=True)

train_iter = BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.seq), device=device, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=32, sort_key=lambda x: len(x.seq), device=device, sort=True, sort_within_batch=True)
test_iter = BucketIterator(test, batch_size=32, sort_key=lambda x: len(x.seq), device=device, sort=True, sort_within_batch=True)

seq_field.build_vocab(train, min_freq=1)

class LSTM(nn.Module):
    def __init__(self, dimension=256):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(seq_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.1)

        self.linear_relu_stack = nn.Sequential(
                nn.Linear(2*dimension, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
                nn.ReLU(),
                nn.Linear(10, 1)
        )

    def forward(self, seq, seq_len):
        seq_emb = self.embedding(seq)

        packed_input = pack_padded_sequence(seq_emb, seq_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), seq_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        seq_fea = self.drop(out_reduced)

        seq_fea = self.linear_relu_stack(seq_fea)
        
        seq_fea = torch.squeeze(seq_fea, 1)
        seq_out = torch.sigmoid(seq_fea)

        return seq_out
        
def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

# training
def train(model,
          optimizer,
          criterion = nn.BCELoss(),
          train_loader = train_iter,
          valid_loader = valid_iter,
          num_epochs = 5,
          eval_every = len(train_iter) // 2,
          file_path = "output",
          best_valid_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    
    for epoch in range(num_epochs):
        for ((seq, seq_len), neut), _ in train_loader:           
            seq = seq.to(device)
            neut = neut.to(device)
            output = model(seq, seq_len)

            loss = criterion(output, neut)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                  # validation loop
                  for ((seq, seq_len), neut), _ in valid_loader:
                      seq = seq.to(device)
                      neut = neut.to(device)
                      output = model(seq, seq_len)

                      loss = criterion(output, neut)
                      valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


lr = 0.0025

model = LSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

print('Commencing Training')
train(model=model, optimizer=optimizer, num_epochs=30)

destination_folder = "output"

train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt')
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Evaluation Function

def evaluate(model, test_loader, version='title', threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    
    with torch.no_grad():
        for ((seq, seq_len), neut), _ in test_loader:           
            seq = seq.to(device)
            neut = neut.to(device)
            output = model(seq, seq_len)
            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(neut.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['True', 'False'])
    ax.yaxis.set_ticklabels(['True', 'False'])
    
    
best_model = LSTM().to(device)
optimizer = optim.Adam(best_model.parameters(), lr=lr)

load_checkpoint(destination_folder + '/model.pt', best_model, optimizer)
evaluate(best_model, test_iter)

plt.show()