from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score
from models.landmark.utils import set_seed

set_seed()


def prepare_data(dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    features = []
    labels = []

    for batch in loader:
        feature, label = batch
        feature = feature.squeeze(0)
        print("feature", feature.shape)
        features.append(feature)
        labels.append(label)

    return torch.stack(features), torch.tensor(labels)


class KNNClassifier:
    def __init__(self):
        self.train_features = None
        self.train_labels = None

    def fit(self, features: torch.FloatTensor, labels: torch.FloatTensor):
        self.train_features = features
        self.train_labels = labels

    def predict(self, input: torch.FloatTensor, k: int = 1) -> int:
        # Compute Euclidean distances to all training samples
        distances = torch.norm(self.train_features - input, dim=1)
        knn_indices = torch.topk(distances, k, largest=False).indices
        knn_labels = self.train_labels[knn_indices]
        # Return the most frequent label
        predicted_label = torch.mode(knn_labels).values.item()
        return predicted_label


def evaluate_knn_train_test(train_dataset, test_dataset, k_values=[1, 3, 5, 7]):
    X_train, y_train = prepare_data(train_dataset)
    X_test, y_test = prepare_data(test_dataset)

    results = {}
    clf = KNNClassifier()
    clf.fit(X_train, y_train)
    for k in k_values:
        y_pred = torch.tensor([clf.predict(x, k) for x in X_test])
        acc = accuracy_score(y_test, y_pred)
        results[k] = acc

    return results
