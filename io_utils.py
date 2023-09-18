import logging
import os
from copy import deepcopy

import torch
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, MolFromMol2File, MolFromPDBFile, AddHs
from rdkit.Geometry import Point3D
from torch_geometric.data import HeteroData


def read_ligand(ligand_path: str, include_hydrogen: bool = True):
    if ligand_path.endswith(".sdf"):
        supplier = SDMolSupplier(ligand_path)
        ligand = supplier.__getitem__(0)
    elif ligand_path.endswith(".mol2"):
        ligand = MolFromMol2File(ligand_path)
    elif ligand_path.endswith(".pdb"):
        ligand = MolFromPDBFile(ligand_path)
    else:
        raise ValueError(f"Input ligand file must be one of {{.sdf, .mol2, .pdb}} file. Got {ligand_path}")
    if include_hydrogen:
        ligand = AddHs(ligand)
    return ligand


def write_ligand(ligand_path: str, ligand: Chem.Mol):
    writer = Chem.SDWriter(ligand_path)
    writer.write(ligand, confId=0)


def load_saved_model(model_path: str, model_class: type, model_parameters: dict) -> torch.nn.Module:
    """
    Loads a pretrained model from weights saved to disk.

    :param model_path: Path to the model file containing the weights in a state_dict.
    :param model_class: Python class of the model.
    :param model_parameters: Additional parameters to be passed to the model constructor.
    :return: torch module ready to be used.
    """
    logging.info(f"Loading saved model of class {model_class} from {model_path}...")
    model = model_class(**model_parameters)
    logging.debug(f"Loading model of class {model_class} with parameters {model_parameters}.")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logging.info("Loaded saved model.")
    return model


def graph_to_sdf(graph: HeteroData, output_path: str):
    """
    Saves a PyTorch Geometric molecular graph to a .sdf file using RDKit.

    :param graph: PyG HeteroData object containing an RDKit Molecule object with molecular information
    and ligand coordinates.
    :param output_path: where to save the .sdf file.
    """
    rdkit_ligand = deepcopy(graph["rdkit_ligand"])
    ligand_coordinates = graph['ligand'].pos.cpu().numpy()
    conformer = rdkit_ligand.GetConformer()
    for i in range(rdkit_ligand.GetNumAtoms()):
        conformer.SetAtomPosition(i, Point3D(*ligand_coordinates[i]))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Chem.SDWriter(output_path).write(rdkit_ligand, confId=0)
