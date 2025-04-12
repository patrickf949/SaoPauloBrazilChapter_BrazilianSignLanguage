from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score
from models.landmark.dataset.landmark_dataset import LandmarkDataset


def prepare_data(dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    features = []
    labels = []

    for idx, batch in enumerate(loader):
        feature, label = batch

        if feature.ndim == 2:  # multi-frame → flatten
            feature = feature.flatten()

        features.append(feature)
        labels.append(label)

    return torch.stack(features), torch.tensor(labels)


class KNNClassifier:
    def __init__(self, k: int):
        self.k = k
        self.train_features = None
        self.train_labels = None

    def fit(self, features: torch.FloatTensor, labels: torch.FloatTensor):
        self.train_features = features
        self.train_labels = labels

    def predict(self, input: torch.FloatTensor) -> int:
        # Compute Euclidean distances to all training samples
        distances = torch.norm(self.train_features - input, dim=1)
        knn_indices = torch.topk(distances, self.k, largest=False).indices
        knn_labels = self.train_labels[knn_indices]
        # Return the most frequent label
        predicted_label = torch.mode(knn_labels).values.item()
        return predicted_label


def evaluate_knn_train_test(train_dataset, test_dataset, k_values=[1, 3]):
    X_train, y_train = prepare_data(train_dataset)
    X_test, y_test = prepare_data(test_dataset)

    results = {}
    for k in k_values:
        clf = KNNClassifier(k)
        clf.fit(X_train, y_train)

        y_pred = torch.tensor([clf.predict(x) for x in X_test])
        acc = accuracy_score(y_test, y_pred)
        results[k] = acc

    return results


if __name__ == "__main__":
    train_dataset = LandmarkDataset(
        "/home/ana/Projects/Omdena/HealthSignLangBrazil/SaoPauloBrazilChapter_BrazilianSignLanguage/code/models/landmark/dataset/configs/dataset.yaml",
        "train",
    )
    val_dataset = LandmarkDataset(
        "/home/ana/Projects/Omdena/HealthSignLangBrazil/SaoPauloBrazilChapter_BrazilianSignLanguage/code/models/landmark/dataset/configs/dataset.yaml",
        "val",
    )
    test_dataset = LandmarkDataset(
        "/home/ana/Projects/Omdena/HealthSignLangBrazil/SaoPauloBrazilChapter_BrazilianSignLanguage/code/models/landmark/dataset/configs/dataset.yaml",
        "test",
    )
    print(evaluate_knn_train_test(train_dataset, test_dataset))
    print(evaluate_knn_train_test(val_dataset, test_dataset, [1]))
