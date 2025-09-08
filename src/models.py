import torch
import torch.nn as nn

class AttentionLogisticRegression(nn.Module):
    def __init__(self, embedding_dim, num_features, dropout_rate=0):
        super().__init__()
        self.fc = nn.Linear(embedding_dim + 50, 20)
        self.lin = nn.Linear(20,1)
        self.dropout = nn.Dropout(dropout_rate)

        self.embedding1 = nn.Embedding(2, 10)  # gender
        self.embedding2 = nn.Embedding(6, 10)  # race
        self.embedding3 = nn.Embedding(3, 10)  # age
        self.embedding4 = nn.Embedding(3, 10)  # education
        self.embedding5 = nn.Embedding(2, 10)  # locale

        self.relu = nn.ReLU()

    def forward(self, target_embedding, other_features):
        g_embf = self.embedding1(other_features[:,0].int())
        r_embf = self.embedding2(other_features[:,1].int())
        a_embf = self.embedding3(other_features[:,2].int())
        e_embf = self.embedding4(other_features[:,3].int())
        l_embf = self.embedding5(other_features[:,4].int())

        combined_demo = torch.cat((g_embf,r_embf,a_embf,e_embf,l_embf), dim=-1)
        combined_feature_vector = torch.cat((target_embedding, combined_demo), dim=1)

        temp = self.fc(combined_feature_vector)
        temp = self.relu(temp)
        output = torch.sigmoid(self.lin(temp))
        return output
