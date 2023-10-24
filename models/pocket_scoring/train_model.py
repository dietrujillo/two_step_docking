import argparse
import logging
import os
import sys

import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm

import wandb

sys.path.append("/home/dit905/dit/two_step_docking")
from dataloader.pdbbind_dataset import PDBBindDataset
from dataloader.protein_ligand_complex import ProteinLigandComplex
from models.pocket_scoring.AffinityScoring import AffinityScoring
from io_utils import read_ligand


def evaluate_ranking(model: torch.nn.Module, pl_complexes: list[ProteinLigandComplex]):
    model.eval()
    correct_rank = []
    pl_complex_names = set([c.name for c in pl_complexes])
    for protein in pl_complex_names:
        protein_complexes = []
        for pl_complex in filter(lambda x: x.name == protein, pl_complexes):
            try:
                ligand = read_ligand(pl_complex.ligand_path, include_hydrogen=True, sanitize=False)
                if ligand is not None:
                    protein_complexes.append(pl_complex)
            except OSError:
                logging.warning(f"Could not read ligand for {pl_complex.name}. Skipping.")
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
               include_hydrogen: bool = True, centroid_threshold: int = 20, shuffle=True, oversample=False,
               sanitize: bool = True):
    pl_complexes = []
    for pl_name in pl_names:
        for pocket in os.listdir(os.path.join(pockets_path, pl_name)):
            try:
                ligand = read_ligand(os.path.join(ligands_path, pl_name, f"{pl_name}_ligand.mol2"),
                                     include_hydrogen=include_hydrogen, sanitize=sanitize)
                if ligand is not None:
                    pl_complexes.append(
                        ProteinLigandComplex(
                            name=pl_name,
                            protein_path=os.path.join(pockets_path, pl_name, pocket),
                            ligand_path=os.path.join(ligands_path, pl_name, f"{pl_name}_ligand.mol2")
                        )
                    )
            except OSError:
                logging.warning(f"Could not read ligand for {pl_name}. Skipping.")

    dataset = PDBBindDataset(pl_complexes, include_label=True, include_hydrogen=include_hydrogen,
                             pocket_predictions_dir=p2rank_cache, centroid_threshold=centroid_threshold,
                             sanitize=sanitize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        sampler=(get_oversampler(dataset) if shuffle and oversample else None))
    return pl_complexes, loader


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.
    predictions = []
    labels_list = []
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
        predictions.extend(torch.round(outputs))
        labels_list.extend(labels.float().unsqueeze(-1))

    return epoch_loss, torch.tensor(predictions).to(device), torch.tensor(labels_list).to(device)


def val_epoch(model, loader, loss_fn):
    model.eval()
    validation_loss = 0.
    predictions = []
    labels_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            validation_loss += loss_fn(outputs, labels.float().unsqueeze(-1)).item()
            predictions.extend(torch.round(outputs))
            labels_list.extend(labels.float().unsqueeze(-1))

    return validation_loss, torch.tensor(predictions).to(device), torch.tensor(labels_list).to(device)


def model_checkpoint(model, filename, checkpoints_dir: str = ".checkpoints"):
    checkpoint_path = os.path.join(checkpoints_dir, filename)
    os.makedirs(checkpoints_dir, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def train(namespace: argparse.Namespace, device: torch.device):
    
    wandb.init(
        project="pocket_ranking",
        config={
            "learning_rate": namespace.lr,
            "batch_size": namespace.batch_size,
            "pockets_path": namespace.pockets_path,
            "train_split_path": namespace.train_split_path,
            "val_split_path": namespace.val_split_path,
            "test_split_path": namespace.test_split_path,
            "epochs": namespace.epochs,
        }
    )
    
    model = AffinityScoring().train().to(device)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=namespace.lr)
    loss_fn = torch.nn.BCELoss().to(device)
    metric = BinaryAccuracy().to(device)

    train_pl_names, val_pl_names, test_pl_names = get_split_names(namespace.pockets_path,
                                                                  namespace.train_split_path,
                                                                  namespace.val_split_path,
                                                                  namespace.test_split_path)

    train_pl_complexes, train_loader = get_loader(train_pl_names, batch_size=namespace.batch_size,
                                                  pockets_path=namespace.pockets_path,
                                                  ligands_path=namespace.ligands_path,
                                                  p2rank_cache=namespace.p2rank_cache,
                                                  shuffle=False, oversample=True, sanitize=False)

    val_pl_complexes, val_loader = get_loader(val_pl_names, namespace.batch_size,
                                              pockets_path=namespace.pockets_path,
                                              ligands_path=namespace.ligands_path,
                                              p2rank_cache=namespace.p2rank_cache,
                                              shuffle=False, oversample=False, sanitize=False)

    best_ranking_accuracy = 0.
    best_epoch = 0
    early_stopping_patience = 5

    for epoch in range(namespace.epochs):
        epoch_loss, train_predictions, train_labels = train_epoch(model=model, loader=train_loader,
                                                                  optimizer=optimizer, loss_fn=loss_fn, device=device)
        val_loss, val_predictions, val_labels = val_epoch(model=model, loader=val_loader, loss_fn=loss_fn)

        ranking_accuracy = evaluate_ranking(model, val_pl_complexes)

        train_accuracy = metric(train_predictions, train_labels)
        val_accuracy = metric(val_predictions, val_labels)

        if ranking_accuracy > best_ranking_accuracy:
            best_ranking_accuracy = val_accuracy
            best_epoch = epoch
            model_checkpoint(model, "best_model.pth")

        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss, 
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "ranking_accuracy": ranking_accuracy
        })

        if epoch - best_epoch > early_stopping_patience:
            logging.info(f"Early stopping at epoch {epoch}.")
            break


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pockets_path", type=str,
                        default="/home/diego/Universidad/Harvard/Lab/docking/.generated_pockets")
    parser.add_argument("--ligands_path", type=str,
                        default="/home/diego/Universidad/Harvard/Lab/docking/data/PDBBind_processed")
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
