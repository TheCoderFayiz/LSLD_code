from src.data_loader import load_data
from src.dataset import CustomDataset
from src.models import AttentionLogisticRegression
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import get_device, seed_all

import torch
from torch.utils.data import DataLoader

def main():
    train_df, test_df = load_data()

    # Device
    device = get_device()

    # Create datasets and dataloaders
    dataset_train = CustomDataset(train_df.drop("Q_overall", axis=1), train_df["Q_overall"], device)
    dataset_test = CustomDataset(test_df.drop("Q_overall", axis=1), test_df["Q_overall"], device)

    dataloader_train = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    # Initialize model
    model = AttentionLogisticRegression(embedding_dim=384, num_features=5).to(device)


    eval_history = train_model(
        model,
        dataloader_train,
        dataloader_test,
        test_df,
        device,
        num_epochs=1000,      
        eval_every=25,
        eval_start_epoch=349  
    )

if __name__ == "__main__":
    main()
