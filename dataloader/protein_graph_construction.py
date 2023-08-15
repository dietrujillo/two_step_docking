from collections import Counter

import torch
from Bio.PDB import Structure
from torch.nn.functional import one_hot
from torch_geometric.data import HeteroData

VALID_AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "HIP",
    "HIE",
    "TPO",
    "HID",
    "LEV",
    "MEU",
    "PTR",
    "GLV",
    "CYT",
    "SEP",
    "HIZ",
    "CYM",
    "GLM",
    "ASQ",
    "TYS",
    "CYX",
    "GLZ",
]


def get_clean_protein(protein: Structure):
    """
    Clean up non-amino-acid residues and remove them from the protein.
    :param protein: biopython protein structure.
    :return: a cleaned up biopython structure where all residues are amino acids.
    """
    for chain in protein.get_chains():
        invalid_residues = []
        for residue in chain.get_residues():
            if residue.get_resname() == "HOH":
                invalid_residues.append(residue.id)
            atom_names = Counter([atom.name for atom in residue.get_atoms()])
            if atom_names["N"] == 0 or atom_names["C"] == 0 or atom_names["CA"] != 1:
                invalid_residues.append(residue.id)
        for residue in invalid_residues:
            chain.detach_child(residue.id)
    return protein


def get_protein_coordinates(protein: Structure) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Obtain 3D coordinates of the protein residues.
    :param protein: biopython protein structure.
    :return: relative coordinates centered around the origin, as well as the absolute coordinates in the PDB file.
    """
    coordinates = []
    for chain in protein.get_chains():
        for residue in chain.get_residues():
            alpha_carbon_coordinates = [list(atom.get_vector()) for atom in residue if atom.name == "CA"]
            coordinates.append(alpha_carbon_coordinates[0])
    absolute_coordinates = torch.tensor(coordinates)
    centroid = torch.mean(absolute_coordinates, dim=0)
    relative_coordinates = absolute_coordinates - centroid
    return relative_coordinates, absolute_coordinates, centroid


def get_protein_features(protein: Structure) -> torch.Tensor:
    """
    Obtain protein residue features.
    :param protein: biopython protein structure.
    :return: node feature matrix as a PyTorch tensor.
    """
    features = []
    for residue in protein.get_residues():
        if residue.get_resname() in VALID_AMINO_ACIDS:
            features.append(VALID_AMINO_ACIDS.index(residue.get_resname()))
        else:
            features.append(len(VALID_AMINO_ACIDS))
    features = torch.tensor(features)
    return one_hot(features, num_classes=len(VALID_AMINO_ACIDS) + 1)


def get_protein_edges(coordinates: torch.Tensor, cutoff: float) -> tuple[torch.Tensor, None]:
    """
    Get edge index and edge features from a protein. The edges are obtained through distance thresholds.
    :param coordinates: residue coordinates tensor.
    :param cutoff: distance cutoff in angstroms.
    :return: the edge index and edge feature tensors.
    """

    # TODO: for now, no edge attributes. When attributes are added, the interface will likely
    #  change to include the protein Structure object.

    distances = torch.cdist(coordinates, coordinates)
    adjacency_matrix = torch.where(distances < cutoff, 1, 0)
    edge_index = adjacency_matrix.nonzero().t().contiguous()

    return edge_index, None


def build_protein_graph(graph: HeteroData, protein: Structure, cutoff: float = 15) -> HeteroData:
    """
    Obtain a graph from a BioPython protein Structure and add to a PyTorch HeteroData object as a graph.
    :param graph: HeteroData object to which the protein will be added.
    :param protein: the protein as a BioPython Structure object.
    :param cutoff: the cutoff threshold for the distance graph, in angstroms.
    :return: the HeteroData graph, modified in place, containing protein information.
    """
    protein = get_clean_protein(protein)
    relative_coordinates, absolute_coordinates, centroid = get_protein_coordinates(protein)
    node_features = get_protein_features(protein)
    edge_index, edge_features = get_protein_edges(relative_coordinates, cutoff=cutoff)

    graph["protein"].pos = relative_coordinates
    graph["protein"].absolute_coordinates = absolute_coordinates
    graph["centroid"] = centroid

    graph["protein"].x = node_features
    graph["protein", "bond", "protein"].edge_index = edge_index
    graph["protein", "bond", "protein"].edge_attr = edge_features

    return graph
