"""
Adapted from DiffDock (Corso et al. (2022). DiffDock: Diffusion steps, twists, and turns for
 molecular docking. arXiv preprint arXiv:2210.01776.)
"""

import torch
from rdkit.Chem import Mol as Molecule
from rdkit.Chem.rdchem import BondType
from torch_geometric.data import HeteroData

BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]

VALID_FEATURES = {
    "possible_atomic_num_list": [str(n) for n in range(1, 119)] + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_numring_list": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring3_list": [False, True],
    "possible_is_in_ring4_list": [False, True],
    "possible_is_in_ring5_list": [False, True],
    "possible_is_in_ring6_list": [False, True],
    "possible_is_in_ring7_list": [False, True],
    "possible_is_in_ring8_list": [False, True],
    "possible_atom_type_2": [
        "C*",
        "CA",
        "CB",
        "CD",
        "CE",
        "CG",
        "CH",
        "CZ",
        "N*",
        "ND",
        "NE",
        "NH",
        "NZ",
        "O*",
        "OD",
        "OE",
        "OG",
        "OH",
        "OX",
        "S*",
        "SD",
        "SG",
        "misc",
    ],
    "possible_atom_type_3": [
        "C",
        "CA",
        "CB",
        "CD",
        "CD1",
        "CD2",
        "CE",
        "CE1",
        "CE2",
        "CE3",
        "CG",
        "CG1",
        "CG2",
        "CH2",
        "CZ",
        "CZ2",
        "CZ3",
        "N",
        "ND1",
        "ND2",
        "NE",
        "NE1",
        "NE2",
        "NH1",
        "NH2",
        "NZ",
        "O",
        "OD1",
        "OD2",
        "OE1",
        "OE2",
        "OG",
        "OG1",
        "OH",
        "OXT",
        "SD",
        "SG",
        "misc",
    ],
}


def _safe_index(lst, item) -> int:
    """
    If the item is in the list, return its index. Otherwise, return the last index in the list.
    :param lst: The list.
    :param item: The item.
    :return: index of the item in the list or last index in the list.
    """
    return lst.index(item) if item in lst else len(lst) - 1


def get_ligand_features(ligand: Molecule) -> torch.Tensor:
    """
    Obtain atom features for the nodes in the ligand graph.
    :param ligand: RDKit Mol ligand structure.
    :return: Node feature matrix.
    """
    ring_info = ligand.GetRingInfo()
    atom_features_list = []
    for atom_index, atom in enumerate(ligand.GetAtoms()):
        atom_features_list.append(
            [
                _safe_index(
                    VALID_FEATURES["possible_atomic_num_list"], atom.GetAtomicNum()
                ),
                VALID_FEATURES["possible_chirality_list"].index(
                    str(atom.GetChiralTag())
                ),
                _safe_index(
                    VALID_FEATURES["possible_degree_list"], atom.GetTotalDegree()
                ),
                _safe_index(
                    VALID_FEATURES["possible_formal_charge_list"],
                    atom.GetFormalCharge(),
                ),
                _safe_index(
                    VALID_FEATURES["possible_implicit_valence_list"],
                    atom.GetImplicitValence(),
                ),
                _safe_index(VALID_FEATURES["possible_numH_list"], atom.GetTotalNumHs()),
                _safe_index(
                    VALID_FEATURES["possible_number_radical_e_list"],
                    atom.GetNumRadicalElectrons(),
                ),
                _safe_index(
                    VALID_FEATURES["possible_hybridization_list"],
                    str(atom.GetHybridization()),
                ),
                VALID_FEATURES["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
                _safe_index(
                    VALID_FEATURES["possible_numring_list"],
                    ring_info.NumAtomRings(atom_index),
                ),
                VALID_FEATURES["possible_is_in_ring3_list"].index(
                    ring_info.IsAtomInRingOfSize(atom_index, 3)
                ),
                VALID_FEATURES["possible_is_in_ring4_list"].index(
                    ring_info.IsAtomInRingOfSize(atom_index, 4)
                ),
                VALID_FEATURES["possible_is_in_ring5_list"].index(
                    ring_info.IsAtomInRingOfSize(atom_index, 5)
                ),
                VALID_FEATURES["possible_is_in_ring6_list"].index(
                    ring_info.IsAtomInRingOfSize(atom_index, 6)
                ),
                VALID_FEATURES["possible_is_in_ring7_list"].index(
                    ring_info.IsAtomInRingOfSize(atom_index, 7)
                ),
                VALID_FEATURES["possible_is_in_ring8_list"].index(
                    ring_info.IsAtomInRingOfSize(atom_index, 8)
                ),
            ]
        )

    return torch.tensor(atom_features_list)


def _encode_bond_type(bond_type: BondType) -> list[int]:
    """
    One-hot encode the bond type for a given bond.
    :param bond_type: RDKit BondType.
    :return: One-hot encoded bond type as a list.
    """
    encoding = [0] * len(BOND_TYPES)
    if bond_type in BOND_TYPES:
        encoding[BOND_TYPES.index(bond_type)] = 1
    return encoding


def get_ligand_edges(ligand: Molecule) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Obtain ligand edges and edge features from the RDKit structure.
    :param ligand: the RDKit Mol ligand structure.
    :return: edge index and edge attributes matrices.
    """
    edge_features = []
    row, col = [], []
    for bond in ligand.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_features.extend(2 * [_encode_bond_type(bond.GetBondType())])

    edge_index = torch.tensor([row, col])
    edge_features = torch.tensor(edge_features)

    return edge_index, edge_features


def get_ligand_coordinates(ligand: Molecule, centroid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Obtain ligand coordinates in 3D space from an RDKit Mol object as well as relative to the ligand centroid.
    :param ligand: the RDKit ligand structure.
    :param centroid: the center of mass of the protein structure. If None, the center of the ligand will be used.
    :return: relative and absolute coordinates matrices.
    """
    absolute_coordinates = torch.tensor(ligand.GetConformer().GetPositions())

    if centroid is None:
        centroid = torch.mean(absolute_coordinates, dim=0)
    relative_coordinates = absolute_coordinates - centroid

    return relative_coordinates, absolute_coordinates, centroid


def build_ligand_graph(
    graph: HeteroData, ligand: Molecule, include_absolute_coordinates: bool = True, use_ligand_centroid: bool = False) -> HeteroData:
    """
    Obtain a graph from an RDKit ligand structure and add to a Pytorch Geometric HeteroData graph.
    :param graph: graph object to hold the graph data.
    :param ligand: RDKit structure.
    :param include_absolute_coordinates: whether to include the absolute coordinates in the data.
    :return: the modified HeteroData graph.
    """
    node_features = get_ligand_features(ligand)
    edge_index, edge_features = get_ligand_edges(ligand)

    relative_coordinates, absolute_coordinates, centroid = get_ligand_coordinates(
        ligand, graph["centroid"] if "centroid" in graph and not use_ligand_centroid else None
    )
    if use_ligand_centroid:
        graph["ligand_centroid"] = centroid

    graph["ligand"].pos = relative_coordinates
    if include_absolute_coordinates:
        graph["ligand"].absolute_coordinates = absolute_coordinates

    graph["ligand"].x = node_features
    graph["ligand", "bond", "ligand"].edge_index = edge_index
    graph["ligand", "bond", "ligand"].edge_attr = edge_features

    return graph
