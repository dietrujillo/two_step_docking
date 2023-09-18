import os
import shutil
import subprocess
from typing import Optional

from rdkit import Chem
from torch_geometric.data import Batch, HeteroData

from dataloader.ligand_graph_construction import build_ligand_graph
from io_utils import graph_to_sdf, read_ligand, write_ligand


class SMINADocking:

    SMINA_PATH = "./smina"
    ADFR_PATH = "./ADFRsuite-1.0"

    def __init__(self, smina_path: str = SMINA_PATH,
                 adfr_path: str = ADFR_PATH,
                 data_path: str = "../../data/PDBBind_processed",
                 tempdir: str = ".tmp_smina",
                 output_dir: str = None,
                 use_whole_protein: bool = True,
                 box_size: float = 20, exhaustiveness: float = 16):
        self.smina_path = smina_path
        self.adfr_path = adfr_path
        self.data_path = data_path
        self.tempdir = tempdir
        self.output_dir = output_dir if output_dir is not None else tempdir
        self.use_whole_protein = use_whole_protein
        self.box_size = box_size
        self.exhaustiveness = exhaustiveness

    def _prepare_protein(self, protein_path: str) -> str:
        reduce_command = f"{os.path.join(self.adfr_path, 'reduce')} {protein_path} -QUIET".split()

        reduced_pdb_path = os.path.join(self.tempdir, f"reduced_{os.path.basename(protein_path)}")
        with open(reduced_pdb_path, "w") as reduced_pdb_file:
            subprocess.run(reduce_command, stdout=reduced_pdb_file, stderr=subprocess.DEVNULL)

        output_pdbqt = os.path.join(self.tempdir, f"{os.path.basename(protein_path)[:-4]}.pdbqt")
        prepare_receptor_command = (f"{os.path.join(self.adfr_path, 'prepare_receptor')} "
                                    f"-r {reduced_pdb_path} -o {output_pdbqt}").split()
        subprocess.run(prepare_receptor_command, stdout=subprocess.DEVNULL)

        return output_pdbqt

    def run_smina(self, item: HeteroData) -> Optional[HeteroData]:
        x, y, z = item["pocket_centroid"]
        pdb = item["name"]
        protein_path = os.path.join(self.data_path, pdb, f'{pdb}_protein.pdb') if self.use_whole_protein else item["protein_path"]

        clean_protein_path = self._prepare_protein(protein_path)
        if os.path.exists(clean_protein_path):
            protein_path = clean_protein_path

        input_ligand_path = os.path.join(self.tempdir, "ligand.sdf")
        graph_to_sdf(item, input_ligand_path)

        temp_output_path = os.path.join(self.tempdir,
                                        f"{os.path.basename(item['protein_path'])[:-4]}_ligand_prediction.sdf")

        command = (f"{self.smina_path} -r {protein_path} -l {input_ligand_path} "
                   f"--center_x={x} --center_y={y} --center_z={z} "
                   f"--size_x={self.box_size} --size_y={self.box_size} --size_z={self.box_size} "
                   f"--exhaustiveness {self.exhaustiveness} "
                   f"-o {temp_output_path} -q --num_modes=40").split()
        subprocess.run(command, stdout=subprocess.DEVNULL)

        os.makedirs(os.path.join(self.output_dir, item["name"]), exist_ok=True)
        output_ligand_path = os.path.join(self.output_dir, item["name"],
                                          f"{os.path.basename(item['protein_path'])[:-4]}_ligand_prediction.sdf")
        if os.path.exists(temp_output_path):
            ligand = read_ligand(temp_output_path)
            ligand = Chem.RemoveHs(ligand)
            write_ligand(output_ligand_path, ligand)

            if ligand is not None:
                item["rdkit_ligand"] = ligand
                item = build_ligand_graph(item, ligand)
                return item
        return None

    def __call__(self, batch: Batch):
        batch = batch.to_data_list()
        os.makedirs(self.tempdir, exist_ok=True)
        for i, item in enumerate(batch):
            batch[i] = self.run_smina(item)
        shutil.rmtree(self.tempdir)
        return batch

    def __repr__(self):
        return f"SMINADocking(**{str(self.__dict__)})"
