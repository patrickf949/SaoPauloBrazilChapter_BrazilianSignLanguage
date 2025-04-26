import torch
from sklearn.metrics import accuracy_score

def evaluate(
    model,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    top_k: int = 5,
):
    model.eval()
    y_true = []
    y_pred_top1 = []
    correct_topk = 0
    total = 0

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            features, labels = batch
            y = labels.squeeze(0).to(device)
            features = features.to(device)

            output = model(features)

            # Top-1 prediction
            top1 = torch.argmax(output, dim=1)
            y_pred_top1.append(top1.item())
            y_true.append(y.item())

            # Top-k prediction
            topk = torch.topk(output, k=top_k, dim=1).indices
            correct_topk += sum([y.item() in topk[i] for i in range(topk.size(0))])
            total += output.size(0)

    acc_top1 = accuracy_score(y_true, y_pred_top1)
    print(f"Top-1 Accuracy: {acc_top1:.4f}")
    acc_topk = correct_topk / total if total > 0 else 0

    print(f"Top-{top_k} Accuracy: {acc_topk:.4f}")

    return acc_top1, acc_topk
