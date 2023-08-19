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
                 autobox_size: float = 8, exhaustiveness: float = 16):
        self.smina_path = smina_path
        self.tempdir = tempdir
        self.autobox_size = autobox_size
        self.exhaustiveness = exhaustiveness

    def run_smina(self, item):
        if "ligand_path" not in item:
            ligand_path = os.path.join(self.tempdir, "ligand.sdf")
            item["ligand"].pos -= item["ligand_centroid"]
            _graph_to_sdf(item, ligand_path)
        else:
            ligand_path = item["ligand_path"]

        output_path = os.path.join(self.tempdir, "output.sdf")
        command = (f"{self.smina_path} -r {item['protein_path']} -l {ligand_path} --autobox_ligand {ligand_path} "
                   f"--autobox_add {self.autobox_size} --exhaustiveness {self.exhaustiveness} "
                   f"-o {output_path} -q").split()
        subprocess.run(command, stdout=subprocess.DEVNULL)

        supplier = SDMolSupplier(output_path)
        item["rdkit_ligand"] = supplier.__getitem__(0)
        item = build_ligand_graph(item, item["rdkit_ligand"])
        item["ligand"].pos += item["centroid"]
        return item

    def __call__(self, batch: Batch):
        batch = batch.to_data_list()
        os.makedirs(self.tempdir, exist_ok=True)
        for item in batch:
            self.run_smina(item)
        shutil.rmtree(self.tempdir)
        return batch
