import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, features, labels, device):
        self.features = features
        self.labels = labels
        self.device = device
        self.processed_data = self.preprocess_data()

    def preprocess_data(self):
        data = []
        for idx in range(len(self.features)):
            row = self.features.iloc[idx]
            target_embedding = torch.tensor(row['text_embedding'], dtype=torch.float32, device=self.device)
            other_features = torch.tensor([
                row['rater_gender_encoded'],
                row['rater_race_encoded'],
                row['rater_age_encoded'],
                row['rater_education_encoded'],
                row['rater_locale_encoded']
            ], dtype=torch.float32, device=self.device)
            unique_id = row['item_id']
            label = torch.tensor(self.labels.iloc[idx], dtype=torch.long, device=self.device)
            avg_label = torch.tensor(row['Q_average'], dtype=torch.float, device=self.device)
            data.append((target_embedding, other_features, label, unique_id, avg_label))
        return data

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.processed_data[idx]
