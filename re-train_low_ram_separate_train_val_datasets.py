import sys
import getopt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.optim import Adam
from Bio import SeqIO
import numpy as np
import time
from helper import *
import click
import logging
import psutil
from torch.utils.data import random_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


# Custom Dataset class to load data on-the-fly
class FastaDataset(Dataset):
        def __init__(self, inputset, labelset, sequence_length=9000):
            self.inputset = list(SeqIO.parse(inputset, "fasta"))  # Convert to list to support indexing
            self.labels = pd.read_csv(labelset).to_numpy()
            self.sequence_length = sequence_length

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            # Load and preprocess sequence and label
            seq = str(self.inputset[idx].seq)
            add_len = max(0, self.sequence_length - len(seq))  # Adjust length
            encoded_seq = generate_long_sequences(seq + "0" * add_len)[:self.sequence_length]
            label = encodeLabel(self.labels[idx][1])
            return torch.tensor(encoded_seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Helper functions
def encodeLabel(num):
    """One-hot encode the label."""
    encoded_l = np.zeros(2)  # Adjusted for binary classification
    encoded_l[num] = 1
    return encoded_l

# Command-line interface using click
@click.command()

@click.option(
    "-i",
    "--input", 
    required=True, 
    type=click.Path(exists=True), 
    help="Path to input FASTA file."
)

@click.option(
    "-l", 
    "--labels", 
    required=True, 
    type=click.Path(exists=True), 
    help="Path to label CSV file."
)

@click.option(
    "-vi",
    "--validation_input", 
    required=True, 
    type=click.Path(exists=True), 
    help="Path to validation data FASTA file."
)

@click.option(
    "-vl", 
    "--validation_labels", 
    required=True, 
    type=click.Path(exists=True), 
    help="Path to validation true labels CSV file."
)

@click.option(
    "-n", 
    "--new_model", 
    required=True, 
    type=click.Path(), 
    help="Path to save the trained model."
)

@click.option(
    "--pre_model",
    "-p",
    help="path to the existing model",
    type=click.Path(),
    required=True,
)

@click.option(
    "-o", 
    "--output", 
    required=True, 
    type=click.Path(), 
    help="Path to save the results log."
)

@click.option(
    "--batch_size",
    "-b",
    help="batch size",
    type=int,
    default=512,
    show_default=True,
    required=False,
)
@click.option(
    "--epochs",
    "-e",
    help="number of epoches",
    type=int,
    default=50,
    show_default=True,
    required=False,
)
@click.option(
    "--learning_rate",
    "-lr",
    help="learning rate",
    type=float,
    default=0.0001,
    show_default=True,
)
@click.help_option("--help", "-h", help="Show this message and exit")

def main(input, labels, validation_input, validation_labels, new_model,pre_model,  output, batch_size, epochs, learning_rate):

    newModelPath = new_model
    preModelPath = pre_model
    inputset = input
    labelset = labels
    validationinputset = validation_input
    validationlabelset = validation_labels
    resultPath = output
    
    # Constants
    INIT_LR = learning_rate
    BATCH_SIZE = int(batch_size)
    EPOCHS = epochs
    SEQUENCE_LENGTH = 9000

    logger = logging.getLogger(f"amaise")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)
    logger.addHandler(consoleHeader)

    fileHandler = logging.FileHandler(f"{resultPath}.log")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    logger.info(f"Pre-Model path: {preModelPath}")
    logger.info(f"Model path: {newModelPath}")
    logger.info(f"Input path: {inputset}")
    logger.info(f"Labels path: {labelset}")
    logger.info(f"Validation Input path: {validationinputset}")
    logger.info(f"Validation Labels path: {validationlabelset}")
    logger.info(f"Results path: {resultPath}")

    logger.info(f"Learning rate: {INIT_LR}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"# Epoches: {EPOCHS}")



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Dataset and DataLoader
    # train_dataset = FastaDataset(inputset, labelset)
    # trainDataLoader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=20)

    # Load and split dataset
    trainDataSet = FastaDataset(inputset, labelset)
    valDataSet = FastaDataset(validationinputset, validationlabelset)

    trainDataLoader = DataLoader(trainDataSet, shuffle=True, batch_size=BATCH_SIZE, num_workers=10)
    valDataLoader = DataLoader(valDataSet, shuffle=False, batch_size=BATCH_SIZE, num_workers=10)


    logger.info("initializing the TCN model...")
    model = nn.DataParallel(TCN())
    # load the pretrained model
    model.load_state_dict(torch.load(preModelPath, device))

    #  freeze layers
    for param in model.module.c_in1.parameters():
        param.requires_grad = False  # Freeze first conv layer
    for param in model.module.c_in2.parameters():
        param.requires_grad = False  # Freeze second conv layer
    for param in model.module.c_in3.parameters():
        param.requires_grad = False   # Fine-tune third conv layer
    for param in model.module.c_in4.parameters():
        param.requires_grad = True   # Fine-tune fourth conv layer
    for param in model.module.fc.parameters():
        param.requires_grad = True   # Fine-tune fully connected layer


    model = model.to(device)

    # Model setup
    opt = Adam(model.parameters(), lr=INIT_LR)
    lossFn = nn.CrossEntropyLoss()
    # Initialize loss function with class weights
   # lossFn = nn.CrossEntropyLoss(weight=class_weights)


    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    epoch_list = []

    # Training loop
    logger.info("Training the network...")
    startTime = time.time()
    

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct_train_predictions = 0

        for step, (x, y) in enumerate(trainDataLoader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = lossFn(pred, y)
            total_loss += loss.item()
            
            _, predicted_labels = torch.max(pred, 1)
            _, true_labels = torch.max(y, 1)
            correct_train_predictions += (predicted_labels == true_labels).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        train_accuracy = correct_train_predictions / len(trainDataLoader.dataset)
        avg_train_loss = total_loss / len(trainDataLoader)

        # Validation loop
        model.eval()
        total_val_loss, correct_val_predictions = 0.0, 0

        with torch.no_grad():
            for x_val, y_val in valDataLoader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                pred_val = model(x_val)
                val_loss = lossFn(pred_val, y_val)
                total_val_loss += val_loss.item()

                _, predicted_val_labels = torch.max(pred_val, 1)
                _, true_val_labels = torch.max(y_val, 1)
                correct_val_predictions += (predicted_val_labels == true_val_labels).sum().item()

        val_accuracy = correct_val_predictions / len(valDataLoader.dataset)
        avg_val_loss = total_val_loss / len(valDataLoader)

        train_losses.append(avg_train_loss)
        validation_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(val_accuracy)
        epoch_list.append(epoch + 1)

        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {avg_train_loss}, Training Accuracy: {train_accuracy}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}")

    # Save model
    torch.save(model.state_dict(), newModelPath)

    endTime = time.time()
    memory = psutil.Process().memory_info()

    logger.info("Total time taken to train the model: {:.2f} min".format((endTime - startTime) / 60))

    logger.info(f"Memory usage: {memory}")

    # Plot training and validation losses vs. epochs
    plt.plot(epoch_list, train_losses, label="Training Loss")
    plt.plot(epoch_list, validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(f"{resultPath}_losses.png", dpi=300, bbox_inches="tight")
    plt.clf()  # Clear the plot for the next figure

    # Plot training and validation accuracies vs. epochs
    plt.plot(epoch_list, train_accuracies, label="Training Accuracy")
    plt.plot(epoch_list, validation_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(f"{resultPath}_accuracies.png", dpi=300, bbox_inches="tight")
    plt.clf()  # Clear the plot for any further plotting

if __name__ == "__main__":
    main()
