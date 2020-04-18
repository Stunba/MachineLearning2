import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def _train(model, iterator, optimizer, criterion, logs_writer, epoch):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for i, batch in enumerate(iterator):
        optimizer.zero_grad()
                
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        logs_writer.add_scalar('Itearation Loss/train', loss, epoch*len(iterator) + i)
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train(model, train_iterator, val_iterator, optimizer, criterion, logs_writer, num_epochs):
    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = _train(model, train_iterator, optimizer, criterion, logs_writer, epoch)
        valid_loss, valid_acc = evaluate(model, val_iterator, criterion)

        logs_writer.add_scalar('Accuracy/train', train_acc, epoch)
        logs_writer.add_scalar('Accuracy/validation', valid_acc, epoch)
        logs_writer.add_scalar('Loss/train', train_loss, epoch)
        logs_writer.add_scalar('Loss/validation', valid_loss, epoch)