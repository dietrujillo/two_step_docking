import argparse
import os

import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb

import sys
sys.path.append("/home/dit905/dit/two_step_docking")
from dataloader.pdbbind_dataset import PDBBindDataset
from dataloader.protein_ligand_complex import ProteinLigandComplex
from models.pocket_scoring.AffinityScoring import AffinityScoring


def evaluate_ranking(model: torch.nn.Module, pl_complexes: list[ProteinLigandComplex]):
    model.eval()
    correct_rank = []
    pl_complex_names = set([c.name for c in pl_complexes])
    for protein in pl_complex_names:
        protein_complexes = list(filter(lambda x: x.name == protein, pl_complexes))
        dataset = PDBBindDataset(protein_complexes, include_label=True,
                                 pocket_predictions_dir=namespace.p2rank_cache,
                                 centroid_threshold=20)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        predictions, labels = [], []
        for item, label in loader:
            prediction = model(item.to(device)).detach().cpu().numpy()
            predictions.append(prediction)
            labels.append(label)
        protein_results = sorted(tuple(zip(predictions, labels)), reverse=True)
        correct_rank.append((protein_results[0][1] == 1).item())
    return sum(correct_rank) / len(correct_rank)


def get_split_names(pockets_path: str, train_split_path: str, val_split_path: str, test_split_path: str):
    all_pl_names = os.listdir(pockets_path)
    if train_split_path is not None and val_split_path is not None and test_split_path is not None:
        with open(train_split_path, "r") as train_file:
            train_pl_names = list(filter(lambda x: x in all_pl_names, [name.strip() for name in train_file.readlines()]))
        with open(val_split_path, "r") as val_file:
            val_pl_names = list(filter(lambda x: x in all_pl_names, [name.strip() for name in val_file.readlines()]))
        with open(test_split_path, "r") as test_file:
            test_pl_names = list(filter(lambda x: x in all_pl_names, [name.strip() for name in test_file.readlines()]))
    else:
        train_pl_names, val_pl_names, test_pl_names = torch.utils.data.random_split(all_pl_names, lengths=[0.7, 0.2, 0.1])
    print(f"{len(train_pl_names)=}, {len(val_pl_names)=}, {len(test_pl_names)=}")
    return train_pl_names, val_pl_names, test_pl_names


def get_oversampler(dataset):
    labels = []
    for _, label in dataset:
        labels.append(label)
    labels = torch.tensor(labels, dtype=torch.int32)
    label_counts = torch.bincount(labels)
    weights = [1 / label_counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights=weights, num_samples=label_counts[0].item() * 2)
    return sampler


def get_loader(pl_names: list[str], batch_size: int, pockets_path: str, ligands_path: str, p2rank_cache: str,
               include_hydrogen: bool = True, centroid_threshold: int = 20, shuffle=True, oversample=False):
    pl_complexes = []
    for pl_name in pl_names:
        for pocket in os.listdir(os.path.join(pockets_path, pl_name)):
            pl_complexes.append(
                ProteinLigandComplex(
                    name=pl_name,
                    protein_path=os.path.join(pockets_path, pl_name, pocket),
                    ligand_path=os.path.join(ligands_path, pl_name, f"{pl_name}_ligand.mol2")
                )
            )

    dataset = PDBBindDataset(pl_complexes, include_label=True, include_hydrogen=include_hydrogen,
                             pocket_predictions_dir=p2rank_cache, centroid_threshold=centroid_threshold)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        sampler=(get_oversampler(dataset) if shuffle and oversample else None))
    return pl_complexes, loader


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.
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
    return epoch_loss


def valid_epoch(model, loader, loss_fn):
    model.eval()
    validation_loss = 0.
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            validation_loss += loss_fn(outputs, labels.float().unsqueeze(-1)).item()
    return validation_loss


def train(namespace: argparse.Namespace, device: torch.device):
    
    wandb.init(
        project="pocket_ranking",
        config={
            "learning_rate": namespace.lr,
            "batch_size": namespace.batch_size,
            "pockets_path": namespace.pockets_path,
            "train_split_path": namespace.train_split_path,
            "val_split_path": namespace.val_split_path,
            "test_split_pathn": namespace.test_split_path,
            "epochs": namespace.epochs,
        }
    )
    
    model = AffinityScoring().train().to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=namespace.lr)
    loss_fn = torch.nn.BCELoss().to(device)

    train_pl_names, val_pl_names, test_pl_names = get_split_names(namespace.pockets_path,
                                                                  namespace.train_split_path,
                                                                  namespace.val_split_path,
                                                                  namespace.test_split_path)

    train_pl_complexes, train_loader = get_loader(train_pl_names, batch_size=namespace.batch_size,
                                                  pockets_path=namespace.pockets_path,
                                                  ligands_path=namespace.ligands_path,
                                                  p2rank_cache=namespace.p2rank_cache,
                                                  shuffle=False, oversample=True)

    val_pl_complexes, val_loader = get_loader(val_pl_names, namespace.batch_size,
                                              pockets_path=namespace.pockets_path,
                                              ligands_path=namespace.ligands_path,
                                              p2rank_cache=namespace.p2rank_cache,
                                              shuffle=False, oversample=False)

    for epoch in range(namespace.epochs):
        epoch_loss = train_epoch(model=model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device)

        if epoch % 10 == 0:
            valid_loss = valid_epoch(model=model, loader=val_loader, loss_fn=loss_fn)
            ranking_accuracy = evaluate_ranking(model, val_pl_complexes)
            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_loss, 
                "val_loss": valid_loss, 
                "ranking_accuracy": ranking_accuracy
            })


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pockets_path", type=str,
                        default="/home/diego/Universidad/Harvard/Lab/docking/.generated_pockets")
    parser.add_argument("--ligands_path", type=str,
                        default="/home/diego/Universidad/Harvard/Lab/docking/data/posebusters_paper_data/posebusters_benchmark_set")
    parser.add_argument("--p2rank_cache", type=str,
                        default="/home/diego/Universidad/Harvard/Lab/docking/.p2rank_cache/p2rank_output")
    parser.add_argument("--train_split_path", type=str, default=None)
    parser.add_argument("--val_split_path", type=str, default=None)
    parser.add_argument("--test_split_path", type=str, default=None)
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    
    namespace = parser.parse_args()

    train(namespace, device)