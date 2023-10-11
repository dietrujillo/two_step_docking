import argparse
import os

import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataloader.pdbbind_dataset import PDBBindDataset
from dataloader.protein_ligand_complex import ProteinLigandComplex
from models.pocket_scoring.AffinityScoring import AffinityScoring


def evaluate_ranking(model: torch.nn.Module, pl_complexes: list[ProteinLigandComplex]):
    model.eval()
    correct_rank = []
    pl_complex_names = set([c.name for c in pl_complexes])
    for protein in pl_complex_names:
        print(protein)
        protein_complexes = list(filter(lambda x: x.name == protein, pl_complexes))
        dataset = PDBBindDataset(protein_complexes, include_label=True, pocket_predictions_dir=namespace.p2rank_cache,
                                 centroid_threshold=20)
        loader = DataLoader(dataset, batch_size=1)
        predictions, labels = [], []
        for item, label in loader:
            prediction = model(item.to(device)).detach().cpu().numpy()
            predictions.append(prediction)
            labels.append(label)
        protein_results = sorted(tuple(zip(predictions, labels)), reverse=True)
        print(protein_results)
        correct_rank.append(protein_results[0][1] == 1)
    print(sum(correct_rank) / len(correct_rank))


def get_oversampler(dataset):
    labels = []
    for _, label in dataset:
        labels.append(label)
    labels = torch.tensor(labels, dtype=torch.int32)
    label_counts = torch.bincount(labels)
    weights = [1 / label_counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights=weights, num_samples=label_counts[0].item() * 2)
    return sampler


def train_epoch(model, loader, dataset, epoch, optimizer, loss_fn, device, pl_complexes):
    model.train()
    epoch_loss = 0.
    correct_predictions = 0
    predictions, labels_list = [], []
    for i, data in enumerate(tqdm(loader)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels.float().unsqueeze(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct_predictions += (torch.round(outputs) == labels.unsqueeze(-1)).float().sum()

        predictions.extend(torch.round(outputs).cpu().detach().numpy())
        labels_list.extend(labels.cpu().detach().numpy())

    epoch_loss /= len(loader)
    epoch_accuracy = 100 * correct_predictions / len(dataset)
    print(f"Epoch {epoch + 1} | Loss: {epoch_loss} | Accuracy: {epoch_accuracy}")
    print(confusion_matrix(labels_list, predictions))

    if epoch % 10 == 0:
        evaluate_ranking(model, pl_complexes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--pockets_path", type=str, default="/home/diego/Universidad/Harvard/Lab/docking/.generated_pockets")
    parser.add_argument("--ligands_path", type=str, default="/home/diego/Universidad/Harvard/Lab/docking/data/posebusters_paper_data/posebusters_benchmark_set")
    parser.add_argument("--p2rank_cache", type=str, default="/home/diego/Universidad/Harvard/Lab/docking/.p2rank_cache/p2rank_output")

    parser.add_argument("--epochs", type=int, default=50)

    namespace = parser.parse_args()

    pl_complexes = []
    for pl_name in os.listdir(namespace.pockets_path):
        for pocket in os.listdir(os.path.join(namespace.pockets_path, pl_name)):
            pl_complexes.append(
                ProteinLigandComplex(
                    name=pl_name,
                    protein_path=os.path.join(namespace.pockets_path, pl_name, pocket),
                    ligand_path=os.path.join(namespace.ligands_path, pl_name, f"{pl_name}_ligand.sdf")
                )
            )

    dataset = PDBBindDataset(pl_complexes, include_label=True, include_hydrogen=True, pocket_predictions_dir=namespace.p2rank_cache, centroid_threshold=20)
    dataloader = DataLoader(dataset, batch_size=32, sampler=get_oversampler(dataset))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AffinityScoring().train().to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss().to(device)

    for epoch in range(namespace.epochs):
        train_epoch(model, dataloader, dataset, epoch, optimizer, loss_fn, device, pl_complexes)
    evaluate_ranking(model, pl_complexes)
