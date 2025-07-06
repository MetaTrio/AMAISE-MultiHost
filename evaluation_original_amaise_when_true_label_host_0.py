from helper import *
import pandas as pd
import logging
import click
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix


@click.command()
@click.option(
    "--pred",
    "-p",
    help="path to predicted labels file (txt format)",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--true",
    "-t",
    help="path to true labels file (csv format)",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--output",
    "-o",
    help="path to output report file",
    type=click.Path(exists=False),
    required=True,
)
@click.help_option("--help", "-h", help="Show this message and exit")
def main(pred, true, output):

    # Set up logging
    logger = logging.getLogger(f"amaise")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)
    logger.addHandler(consoleHeader)

    # Read the predicted labels from txt file 
    pred_df = pd.read_csv(pred, sep=", ", header=0)
    
    # Rename column for easier access
    pred_df.rename(columns={"classification label (0 for microbe": "pred_label"}, inplace=True)
    pred_df.rename(columns={"1 for host)": "length"}, inplace=True)
    pred_df = pred_df.iloc[:, :-1]
    
    # Read the true labels from CSV file with two columns: 'id' and 'y_true'
    true_df = pd.read_csv(true, usecols=['id', 'y_true'])

    # Merge true and predicted labels on 'id'
    merged_df = pd.merge(true_df, pred_df[['id', 'pred_label']], on='id', how='inner')

    # if pred host = 1 and pred microbe = 0 (This happens in orginal AMAISE) but true host(in true label file)
    # is 0 and microbial is 1 convert true host to be 1 and microbial to be 0
    merged_df['y_true_flipped'] = 1 - merged_df['y_true']

    # Extract the flipped true and predicted labels as lists
    true = merged_df['y_true_flipped'].tolist()
    pred = merged_df['pred_label'].tolist()

    # Generate the classification report for binary classification
    # ["Microbial","Host"] = [0, 1]
    report = classification_report(true, pred, target_names=["Microbial","Host"], output_dict=False)

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(true, pred)

    # Format confusion matrix as text
    conf_matrix_df = pd.DataFrame(conf_matrix, index=["True Microbial", "True Host"], columns=["Predicted Microbial", "Predicted Host"])

    print(report,"\n")
    print("Confusion Matrix (Microbial vs. Host):")
    print(conf_matrix_df)

    # Output the report to a text file
    with open(output, 'w') as f:
        f.write(f"Binary Classification Report (Host vs. Microbial):\n\n")
        f.write(report)

        f.write("\nConfusion Matrix (Microbial vs. Host):\n")
        f.write(conf_matrix_df.to_string())  

    logger.info(f"Classification report and confusion matrix saved to: {output}")


if __name__ == "__main__":
    main()
