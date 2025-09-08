import torch
import torch.optim as optim
from .loss import custom_loss
from .evaluate import evaluate_model

def train_model(
    model,
    dataloader_train,
    dataloader_test,
    test_df,
    device,
    num_epochs: int = 1000,
    eval_every: int = 25,
    eval_start_epoch: int = 349,
):
    """
    Train loop that runs periodic evaluation like the original notebook:
      - Evaluate at epoch == 0
      - Then evaluate every `eval_every` epochs only when epoch > eval_start_epoch
        (this matches: epoch==0 or ((epoch+1) % 25 == 0 and epoch>350))
    Returns:
      - eval_history: list of dicts with evaluation results per evaluation call
    """
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    eval_history = []
    best_loss = float("inf")
    best_epoch = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for target_embedding, other_features, label, unique_id, avg_label in dataloader_train:
            optimizer.zero_grad()
            outputs = model(target_embedding, other_features)
            loss = custom_loss(
                model,
                outputs,
                label,
                avg_label,
                unique_id,
                dataloader_train.dataset.features  # pass features DataFrame for KL term
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_epoch_loss = running_loss / max(len(dataloader_train), 1)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}")

        # Keep track of best loss (optional)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch + 1

        # Evaluation condition exactly like your notebook
        do_eval = (epoch == 0) or ( ((epoch + 1) % eval_every == 0) and (epoch > eval_start_epoch) )
        if do_eval:
            print(f"--> Running evaluation at epoch {epoch+1} ...")
            try:
                results = evaluate_model(model, dataloader_test, test_df, device)
                results["epoch"] = epoch + 1
                eval_history.append(results)
            except Exception as e:
                print(f"Evaluation failed at epoch {epoch+1}: {e}")

    print(f"Training finished. Best epoch by loss: {best_epoch} with loss {best_loss:.6f}")
    return eval_history
