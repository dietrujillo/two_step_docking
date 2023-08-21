import os
import shutil
import subprocess

from rdkit.Chem import SDMolSupplier
from torch_geometric.data import Batch

from dataloader.ligand_graph_construction import build_ligand_graph
from pipeline import _graph_to_sdf

SMINA_PATH = "./smina"


class SMINADocking:
    def __init__(self, smina_path: str = SMINA_PATH, tempdir: str = ".tmp_smina",
                 box_size: float = 20, exhaustiveness: float = 16):
        self.smina_path = smina_path
        self.tempdir = tempdir
        self.box_size = box_size
        self.exhaustiveness = exhaustiveness

    def run_smina(self, item):
        if "ligand_path" not in item:
            ligand_path = os.path.join(self.tempdir, "ligand.sdf")
            item["ligand"].pos -= item["ligand_centroid"]
            _graph_to_sdf(item, ligand_path)
        else:
            ligand_path = item["ligand_path"]

        output_path = os.path.join(self.tempdir, "output.sdf")
        x, y, z = item["pocket_centroid"]
        command = (f"{self.smina_path} -r {item['protein_path']} -l {ligand_path} "
                   f"--center_x={x} --center_y={y} --center_z={z} "
                   f"--size_x={self.box_size} --size_y={self.box_size} --size_z={self.box_size} "
                   f"--exhaustiveness {self.exhaustiveness} "
                   f"-o {output_path} -q").split()
        subprocess.run(command, stdout=subprocess.DEVNULL)

        supplier = SDMolSupplier(output_path)
        ligand = supplier.__getitem__(0)
        if ligand is not None:
            item["rdkit_ligand"] = ligand
            item = build_ligand_graph(item, ligand)
            item["ligand"].pos += item["centroid"]
            return item
        return None

    def __call__(self, batch: Batch):
        batch = batch.to_data_list()
        os.makedirs(self.tempdir, exist_ok=True)
        for item in batch:
            self.run_smina(item)
        shutil.rmtree(self.tempdir)
        return batch
