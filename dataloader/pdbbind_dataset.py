import logging
import os
from typing import Union, Optional

import pandas as pd
import torch
from Bio.PDB import PDBParser
from rdkit.Chem import MolFromSmiles, MolFromMol2File, SDMolSupplier, AddHs, MolToSmiles
from rdkit.Chem.rdDistGeom import EmbedMolecule
from torch_geometric.data import Dataset, HeteroData

from dataloader.ligand_graph_construction import build_ligand_graph
from dataloader.protein_graph_construction import build_protein_graph
from dataloader.protein_ligand_complex import ProteinLigandComplex


class PDBBindDataset(Dataset):
    def __init__(
        self,
        data: list[ProteinLigandComplex],
        include_label: bool = False,
        include_absolute_coordinates: bool = True,
        include_hydrogen: bool = True,
        use_ligand_centroid: bool = False,
        pocket_predictions_dir: str = ".p2rank_cache/p2rank_output",
        **kwargs
    ):
        """
        The PDBBind dataset.
        :param data: a list of ProteinLigandComplex objects containing the paths to proteins and ligands.
        :param include_label: whether to include the true label in the data batches.
        :param include_absolute_coordinates: whether to include the absolute (true) coordinates in the ligand.
        :param include_hydrogen: whether to add hydrogen atoms to the ligand graph
        """
        super().__init__(**kwargs)
        self.data = data
        self.include_label = include_label
        self.include_absolute_coordinates = include_absolute_coordinates
        self.include_hydrogen = include_hydrogen
        self.use_ligand_centroid = use_ligand_centroid
        self.pocket_predictions_dir = pocket_predictions_dir

    def len(self) -> int:
        """
        Return the size of the dataset.
        :return: length of the dataset.
        """
        return len(self.data)

    def get_pocket_prediction(self, pl_complex: ProteinLigandComplex) -> tuple[int, Optional[pd.Series]]:
        pocket_num = int(os.path.basename(pl_complex.protein_path).split("_")[1].split(".")[0].split("_")[0])
        p2rank_predictions = pd.read_csv(
            os.path.join(self.pocket_predictions_dir,
                         f"{os.path.basename(pl_complex.name.split('_')[0])}_protein.pdb_predictions.csv"))
        try:
            pocket_prediction = p2rank_predictions.iloc[pocket_num]
        except IndexError:
            if len(p2rank_predictions) != 0:
                logging.error(f"pocket_centroid exception when accessing p2rank prediction table. "
                              f"{pl_complex.name=}, {pl_complex.protein_path=}, {len(p2rank_predictions)=}, {pocket_num=}")
            return 0, None
        return pocket_num, pocket_prediction

    def _read_ligand(self, ligand_path: str, include_hydrogen: bool = True):
        if ligand_path.endswith(".sdf"):
            supplier = SDMolSupplier(ligand_path)
            ligand = supplier.__getitem__(0)
        elif ligand_path.endswith(".mol2"):
            ligand = MolFromMol2File(ligand_path)
        else:
            raise ValueError(f"Input ligand file must be either .sdf or .mol2 file. Got {ligand_path}")
        if include_hydrogen:
            ligand = AddHs(ligand)
        return ligand

    def _add_protein_graph(self, graph: HeteroData) -> HeteroData:
        """
        Add protein (receptor) information to the protein-ligand complex graph.
        :param graph: HeteroData object to which protein information will be added. Will be modified in place.
        :return: the modified graph.
        """
        pdbparser = PDBParser()
        protein = pdbparser.get_structure(graph["name"], graph["protein_path"])
        graph = build_protein_graph(graph, protein)
        return graph

    def _add_ligand_graph(self, graph: HeteroData) -> HeteroData:
        """
        Add ligand information to the protein-ligand complex graph.
        :param graph: HeteroData object to which ligand information will be added. Will be modified in place.
        :return: the modified graph.
        """
        include_absolute_coordinates = self.include_absolute_coordinates
        if graph["ligand_smiles"] != {}:
            ligand = MolFromSmiles(graph["ligand_smiles"])
            if self.include_hydrogen:
                ligand = AddHs(ligand)
            EmbedMolecule(ligand)
            if self.include_absolute_coordinates:
                logging.warning(
                    f"The protein-ligand complex {graph['name']} was loaded from a SMILES string but "
                    "PDBBindDataset.include_absolute_coordinates is True. "
                    "If you need to include true coordinates for the ligand, provide a path to a "
                    ".sdf file instead of SMILES."
                )
                include_absolute_coordinates = False
        else:
            ligand = self._read_ligand(graph["ligand_path"], include_hydrogen=self.include_hydrogen)
        graph["rdkit_ligand"] = ligand

        if graph["ligand_reference_path"] != {}:
            reference_ligand = self._read_ligand(graph["ligand_reference_path"], include_hydrogen=self.include_hydrogen)
            graph["rdkit_reference_ligand"] = reference_ligand

        if graph["ligand_smiles"] == {}:
            del graph["ligand_smiles"]
            graph["ligand_smiles"] = MolToSmiles(ligand)

        graph = build_ligand_graph(graph, ligand, include_absolute_coordinates=include_absolute_coordinates,
                                   use_ligand_centroid=self.use_ligand_centroid)
        return graph
        
    def _compute_label(self, pl_complex: ProteinLigandComplex) -> float:
        """
        Compute true labels from the ProteinLigandComplex information.
        :param pl_complex: ProteinLigandComplex object containing paths to the protein and ligand files. In this
         function, a ligand SMILES does not suffice, as we need the true ligand coordinates to compute the label.
        :return: the item label as a float number measuring whether the pocket is the closest to the ligand.
        """
        raise NotImplementedError

    def get(self, index: int) -> Union[HeteroData, tuple[HeteroData, float]]:
        """
        Retrieve graph at a given index.
        :param index: index of the data to get.
        :return: a HeteroData graph with protein and ligand graphs, as well as a label when self.include_label is True.
        """
        pl_complex = self.data[index]
        out = HeteroData()
        out["name"] = pl_complex.name
        out["protein_path"] = pl_complex.protein_path
        out["ligand_path"] = pl_complex.ligand_path
        out["ligand_smiles"] = pl_complex.ligand_smiles
        if pl_complex.ligand_reference_path is not None:
            out["ligand_reference_path"] = pl_complex.ligand_reference_path

        out = self._add_protein_graph(out)
        out = self._add_ligand_graph(out)

        pocket_num, pocket_prediction = self.get_pocket_prediction(pl_complex)
        out["pocket_num"] = pocket_num
        out["pocket_prediction"] = pocket_prediction
        if pocket_prediction is not None:
            centroid = torch.Tensor(
                [pocket_prediction["   center_x"], pocket_prediction["   center_y"], pocket_prediction["   center_z"]])
            out["pocket_centroid"] = centroid
        else:
            out["pocket_centroid"] = out["centroid"]

        if self.include_label:
            label = self._compute_label(pl_complex)
            return out, label
        return out
