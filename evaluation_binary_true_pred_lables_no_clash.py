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

    # Read the true labels from CSV file with two columns: 'id' and 'y_true'
    true_df = pd.read_csv(true, usecols=['id', 'y_true'])
    
    # Convert any label > 1 to 1
    true_df['y_true'] = true_df['y_true'].apply(lambda x: 1 if x > 1 else x)
    #  size of the true_df
    print("true",true_df.shape)

    # Read the predicted labels from txt file 
    pred_df = pd.read_csv(pred, sep=",", names=['id', 'pred_label', 'length','pred_prob'], header=0)
    # Keep only the first occurrence for each ID
    pred_df = pred_df.drop_duplicates(subset='id', keep='first')


    # Merge true and predicted labels on 'id'
    merged_df = pred_df.merge(true_df[['id', 'y_true']], on='id', how='left')
    merged_df = merged_df.dropna(subset=['y_true']) 
    merged_df['pred_label'] = merged_df['pred_prob'].apply(lambda x: 1 if x >= 0.95 else 0)
    print("mreged", merged_df.shape) 
    
    # Extract lists
    true = merged_df['y_true'].tolist()
    pred = merged_df['pred_label'].tolist()

    # Generate the classification report for binary classification
    # ["Host", "Microbial"] = [0, 1]
    report = classification_report(true, pred, target_names=["Host", "Microbial"], output_dict=False)
    print(report,"\n")

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(true, pred)

    conf_matrix_df = pd.DataFrame(conf_matrix, index=["True Host", "True Microbial"], columns=["Predicted Host", "Predicted Microbial"])


    print("Confusion Matrix (Host vs. Microbial):")
    print(conf_matrix_df)

    print()

    # Normalized confusion matrix 
    conf_matrix_normalized = confusion_matrix(true, pred, normalize='true')

    conf_matrix_norm_df = pd.DataFrame(
        np.round(conf_matrix_normalized, 3),
        index=["True Host", "True Microbial"],
        columns=["Predicted Host", "Predicted Microbial"]
    )

    print("Confusion Matrix (Percentages)")
    print(conf_matrix_norm_df)
    print()

    # Output the report to a text file
    with open(output, 'w') as f:
        f.write(f"Binary Classification Report (Host vs. Microbial):\n\n")
        f.write(report)

        f.write("\nConfusion Matrix (Host vs. Microbial):\n")
        f.write(conf_matrix_df.to_string())  

        f.write("\n\nNormalized Confusion Matrix (Host vs. Microbial):\n")
        f.write(conf_matrix_norm_df.to_string())


    logger.info(f"Classification report and confusion matrix saved to: {output}")


if __name__ == "__main__":
    main()
