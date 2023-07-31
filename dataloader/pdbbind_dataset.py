import logging
from typing import Any, Callable, Optional

from Bio.PDB import PDBParser
from rdkit.Chem import MolFromSmiles, SDMolSupplier, AddHs
from rdkit.Chem.rdDistGeom import EmbedMolecule
from torch_geometric.data import Dataset, HeteroData

from dataloader.protein_ligand_complex import ProteinLigandComplex
from dataloader.protein_graph_construction import build_protein_graph
from dataloader.ligand_graph_construction import build_ligand_graph


class PDBBindDataset(Dataset):
    def __init__(
        self,
        data: list[ProteinLigandComplex],
        include_coordinates: bool = True,
        include_hydrogen: bool = False,
        root: Optional[str] = None,
        transform: Optional[Callable[..., Any]] = None,
        pre_transform: Optional[Callable[..., Any]] = None,
        pre_filter: Optional[Callable[..., Any]] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = data
        self.include_coordinates = include_coordinates
        self.include_hydrogen = include_hydrogen

    def len(self) -> int:
        return len(self.data)

    def _add_protein_graph(self, graph: HeteroData):
        pdbparser = PDBParser()
        protein = pdbparser.get_structure(graph["name"], graph["protein_path"])
        graph = build_protein_graph(
            graph, protein, include_coordinates=self.include_coordinates
        )
        return graph

    def _add_ligand_graph(self, graph: HeteroData):
        include_coordinates = self.include_coordinates
        if graph["ligand_path"] is not None:
            supplier = SDMolSupplier(graph["ligand_path"])
            ligand = supplier.__getitem__(0)
        else:
            ligand = MolFromSmiles(graph["ligand_smiles"])
            EmbedMolecule(ligand)
            if self.include_coordinates:
                logging.warning(
                    f"The protein-ligand complex {graph['name']} was loaded from a SMILES string but "
                    "PDBBindDataset.include_coordinates is True. "
                    "If you need to include true coordinates for the ligand, provide a path to a "
                    ".sdf file instead of SMILES."
                )
                include_coordinates = False

        if self.include_hydrogen:
            AddHs(ligand)
        graph["rdkit_ligand"] = ligand
        graph = build_ligand_graph(
            graph, ligand, include_coordinates=include_coordinates
        )
        return graph

    def get(self, index: int) -> HeteroData:
        pl_complex = self.data[index]
        out = HeteroData()
        out["name"] = pl_complex.name
        out["protein_path"] = pl_complex.protein_path
        out["ligand_path"] = pl_complex.ligand_path
        out["ligand_smiles"] = pl_complex.ligand_smiles

        out = self._add_protein_graph(out)
        out = self._add_ligand_graph(out)

        return out
