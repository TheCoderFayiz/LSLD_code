import torch
import torch.nn.functional as F

def custom_function(x, avg_label):
    return torch.clamp(0.5 * (1 + torch.tanh(10000*(x-avg_label))),0.0001,0.9999)

def custom_loss(model, outputs, label, avg_label, index, data_train):
    weights = torch.ones_like(label, dtype=torch.float)

    avg_label = avg_label.unsqueeze(1)
    output_tensor = custom_function(outputs, avg_label)

    unique_indices = torch.unique(index)
    kl_value = 0
    for idx in torch.unique(torch.tensor(data_train['item_id'])):
        mask = (index == idx)
        out_value = torch.mean(output_tensor[mask])
        avg_value = torch.mean(avg_label[mask])
        kl_value += avg_value * (-torch.log(out_value)) + (1-avg_value)*(-torch.log(1-out_value))

    kl_value = kl_value / torch.tensor(6000, device=label.device)

    loss_per_sample = F.binary_cross_entropy(outputs, label.unsqueeze(-1).float(), reduction='none')
    weighted_loss = loss_per_sample.mean()

    l2_lambda = 1.5e-4
    l2_reg = (
        torch.sum(model.embedding1.weight ** 2) +
        torch.sum(model.embedding2.weight ** 2) +
        torch.sum(model.embedding3.weight ** 2) +
        torch.sum(model.embedding4.weight ** 2) +
        torch.sum(model.embedding5.weight ** 2)
    )

    return kl_value + weighted_loss + l2_lambda*l2_reg