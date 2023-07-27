from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class ProteinLigandComplex:
    name: str
    protein_path: str
    ligand_path: Optional[str] = None
    ligand_smiles: Optional[str] = None

    def __post_init__(self):
        assert self.name is not None
        assert self.protein_path is not None
        assert self.ligand_path is not None or self.ligand_smiles is not None
