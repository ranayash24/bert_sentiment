import config
import pandas as pd
from sklearn import model_selection
import dataset
import torch

def run():
    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")
    dfx.sentiment = dfx.sentiment.apply(
        lambda x:1 if x == "positive" else 0
    )
    df_train , df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.1,
        random_state=42,
        stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop = True)
    df_valid = df_valid.reset_index(drop = True)

    train_dataset = dataset.BERTDataset(
        review= df_train.review.values,
        target= df_train.target.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE,
        num_workers = 4
    )