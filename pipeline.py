import logging
import os
import shutil
import subprocess
import time
from collections import defaultdict
from typing import Callable, Optional

import dask
import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, PDBParser
from dask.diagnostics import ProgressBar
from posebusters.posebusters import PoseBusters
from rdkit import Chem
from spyrmsd import molecule
from spyrmsd.rmsd import symmrmsd
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataloader.pdbbind_dataset import PDBBindDataset
from dataloader.protein_ligand_complex import ProteinLigandComplex
from io_utils import load_saved_model, read_ligand

P2RANK_EXECUTABLE = "/home/dit905/dit/p2rank_24/prank"


class TwoStepBlindDocking:

    def __init__(self, top_k: int = 1, segmentation_distance_threshold: float = 20.0,
                 p2rank_executable_path: str = P2RANK_EXECUTABLE, p2rank_cache_path: str = ".p2rank_cache",
                 pockets_saved_path: str = ".generated_pockets",
                 ranked_pockets_path: str = ".ranked_pockets",
                 ignore_cache: bool = False,
                 pocket_scoring_module: Callable = None, pocket_scoring_path: str = None,
                 pocket_scoring_model_class: type = None, pocket_scoring_model_params: dict = None,
                 pocket_docking_module: Callable = None, pocket_docking_path: str = None,
                 pocket_docking_model_class: type = None, pocket_docking_model_params: dict = None,
                 docking_predictions_path: str = "docking_predictions",
                 scoring_batch_size: int = 32, docking_batch_size: int = 32):
        self.top_k = top_k
        self.segmentation_distance_threshold = segmentation_distance_threshold

        self.p2rank_executable = p2rank_executable_path
        self.p2rank_cache_path = p2rank_cache_path
        os.makedirs(p2rank_cache_path, exist_ok=True)

        self.pockets_saved_path = pockets_saved_path
        os.makedirs(pockets_saved_path, exist_ok=True)

        self.ranked_pockets_path = ranked_pockets_path
        os.makedirs(ranked_pockets_path, exist_ok=True)

        self.ignore_cache = ignore_cache

        self.pocket_scoring_module = pocket_scoring_module
        if pocket_scoring_module is None:
            if pocket_scoring_path is not None and pocket_scoring_model_class is not None:
                self.pocket_scoring_module = load_saved_model(pocket_scoring_path, pocket_scoring_model_class,
                                                              pocket_scoring_model_params)
            else:
                raise ValueError("No pocket scoring module specified. Please provide a model "
                                 "to pocket_scoring_module or a model class and weights path "
                                 "to pocket_scoring_model_class and pocket_scoring_model_params.")
        self.scoring_batch_size = scoring_batch_size

        self.pocket_docking_module = pocket_docking_module
        if pocket_docking_module is None:
            if pocket_docking_path is not None and pocket_docking_model_class is not None:
                self.pocket_docking_module = load_saved_model(pocket_docking_path, pocket_docking_model_class,
                                                              pocket_docking_model_params)
            else:
                raise ValueError("No pocket docking module specified. Please provide a model "
                                 "to pocket_docking_module or a model class and weights path "
                                 "to pocket_docking_model_class and pocket_docking_model_params.")
        self.docking_batch_size = docking_batch_size

        self.docking_predictions_path = docking_predictions_path
        os.makedirs(docking_predictions_path, exist_ok=True)

        self.evaluator = PoseBusters()

    def __repr__(self):
        return f"TwoStepBlindDocking(**{str(self.__dict__)})"

    def _run_p2rank(self, proteins: list[str], p2rank_input_filename: str, p2rank_output_folder: str):
        """
        Generates a file in the format expected by p2rank and then runs the p2rank algorithm from the shell.

        :param proteins: list of paths to protein .pdb files.
        :param p2rank_input_filename: the file from which p2rank will read inputs.
        :param p2rank_output_folder: the folder to which p2rank will output its predictions.
        """
        logging.debug("Generating p2rank input file")
        with open(p2rank_input_filename, "w") as file:
            for protein_path in proteins:
                assert os.path.exists(protein_path), f"{protein_path} cannot be found."
                file.write(f"{os.path.relpath(protein_path, start=os.path.dirname(p2rank_input_filename))}\n")

        logging.debug("Generated p2rank input file.")
        logging.info("Running p2rank pocket finding algorithm...")

        command = f"{self.p2rank_executable} predict {p2rank_input_filename} -o {p2rank_output_folder} -threads 16 " \
                  f"-log_level WARN -visualizations 0"
        subprocess.run(command, shell=True)

    @dask.delayed
    def _segment_protein_pocket(self, protein_id: str, protein_path: str, pocket_prediction: pd.Series,
                                output_path: str):
        """
        Obtain a segmentation of the protein pocket, including all residues in the predicted pocket from p2rank
        as well as all residues within a threshold distance of the pocket centroid.

        This function has the "delayed" decorator from dask (for parallelization) and has to be called within
        dask.compute to obtain output.

        :param protein_id: id for the protein to be segmented.
        :param protein_path: the path to the protein .pdb file.
        :param pocket_prediction: a Series object from the p2rank output predicted pocket.
        :param output_path: location to save the segmented pocket as a new PDB file.
        """
        pdbparser = PDBParser()
        pdbio = PDBIO()

        protein = pdbparser.get_structure(protein_id, protein_path)
        pocket_centroid = np.array(
            [pocket_prediction["   center_x"], pocket_prediction["   center_y"], pocket_prediction["   center_z"]])
        pocket_residues = pocket_prediction[" residue_ids"].split()

        for chain in protein.get_chains():
            removed_residues = []
            for residue in chain.get_residues():
                # p2rank predictions come in <chain>_<residue_number> format (A_112 for the 112th amino acid of chain A)
                if f"{chain.id}_{residue.get_id()[1]}" not in pocket_residues:
                    coords = []
                    for atom in residue.get_atoms():
                        coords.append(atom.get_coord())
                    residue_centroid = np.array(coords).mean(axis=0)
                    if np.linalg.norm(residue_centroid - pocket_centroid) >= self.segmentation_distance_threshold:
                        removed_residues.append(residue)
            for residue in removed_residues:
                chain.detach_child(residue.id)

        pdbio.set_structure(protein)
        pdbio.save(output_path)

    def _generate_segmented_pockets(self, protein_ids: list[str], protein_paths: list[str], p2rank_output_folder: str,
                                    pockets_saved_path: str = None):
        """
        Generate segmented pockets based on p2rank pocket predictions in parallel, for a large number of complexes.
        From a protein PDB and a list of pockets, create multiple PDB segmentations of each pocket and save to disk.

        :param protein_ids: list of protein IDs as strings.
        :param protein_paths: a list of paths to protein .pdb files.
        :param p2rank_output_folder: output folder of the p2rank prediction,
            containing predicted pockets for all proteins.
        :param pockets_saved_path: where to save the segmented pockets. If None, defaults to self.pockets_saved_path.
        """
        logging.debug("Preparing pocket segmentations from p2rank predicted pockets...")

        pockets_saved_path = pockets_saved_path if pockets_saved_path is not None else self.pockets_saved_path

        segmentations = []
        for protein_id, protein_path in zip(protein_ids, protein_paths):
            p2rank_predictions = pd.read_csv(
                os.path.join(p2rank_output_folder, f"{os.path.basename(protein_path)}_predictions.csv"))
            pdb_pockets_dir = os.path.join(pockets_saved_path, protein_id)
            os.makedirs(pdb_pockets_dir, exist_ok=True)

            if len(p2rank_predictions) == 0:
                logging.warning(f"p2rank failed to find pockets for {protein_id}. Using center of protein.")
                shutil.copyfile(protein_path, os.path.join(pdb_pockets_dir, f"{protein_id}_0.pdb"))
            else:
                for pocket_id, pocket in p2rank_predictions.iterrows():
                    segmentations.append(
                        self._segment_protein_pocket(
                            protein_id, protein_path, pocket,
                            os.path.join(pdb_pockets_dir, f"{protein_id}_{pocket_id}.pdb")
                        )
                    )

        with ProgressBar():
            dask.compute(segmentations)

    def _rank_pockets(self, pl_complexes: list[ProteinLigandComplex]):
        """
        Run a ligand dependent pocket scoring method on the p2rank segmented pockets and rename them according to
        their rank.

        :param pl_complexes: a list of protein-ligand complexes containing PDB ID, path to the protein .pdb
            and either path to the ligand .sdf or .mol2 or ligand SMILES.
        """
        data = []
        shutil.rmtree(self.ranked_pockets_path)
        for pl_complex in pl_complexes:
            os.makedirs(os.path.join(self.ranked_pockets_path, pl_complex.name))
            pockets = os.listdir(os.path.join(self.pockets_saved_path, pl_complex.name))
            if len(pockets) > 1:
                for i, pocket in enumerate(pockets):
                    data.append(ProteinLigandComplex(name=f"{pl_complex.name}_{int(pocket.split('_')[1][:-4])}",
                                                     protein_path=os.path.join(self.pockets_saved_path, pl_complex.name,
                                                                               pocket),
                                                     ligand_path=pl_complex.ligand_path,
                                                     ligand_smiles=pl_complex.ligand_smiles,
                                                     ligand_reference_path=pl_complex.ligand_reference_path))
            else:
                shutil.copyfile(os.path.join(self.pockets_saved_path, pl_complex.name, pockets[0]),
                                os.path.join(self.ranked_pockets_path, pl_complex.name,
                                             f"{pockets[0][:-4]}_rank1.pdb"))

        logging.info("Running ligand-dependent pocket scoring model...")
        dataset = PDBBindDataset(data, include_absolute_coordinates=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=self.scoring_batch_size)
        predictions = []
        for batch in tqdm(loader):
            predictions.extend(self.pocket_scoring_module(batch))

        logging.debug("Ranking individual pockets...")
        pocket_predictions = []
        for pl_complex, prediction in zip(data, predictions):
            pocket_predictions.append({"id": pl_complex.name.split("_")[0], "pocket_num": pl_complex.name.split("_")[1],
                                       "protein_path": pl_complex.protein_path, "prediction": prediction})
        pocket_predictions = pd.DataFrame(pocket_predictions)
        pocket_predictions["ranking"] = pocket_predictions.groupby("id")["prediction"].rank(ascending=True)
        pocket_predictions.sort_values(by=["id", "ranking"], ascending=True, inplace=True)
        pocket_predictions["ranked_protein_path"] = pocket_predictions.apply(
            lambda row: os.path.join(self.ranked_pockets_path,
                                     row["id"],
                                     f"{row['id']}_{row['pocket_num']}_rank{int(row['ranking'])}.pdb"), axis=1
        )

        for _, _, _, protein_path, _, _, ranked_protein_path in pocket_predictions.itertuples():
            shutil.copyfile(protein_path, ranked_protein_path)
        logging.debug("Finished ranking predicted pockets.")

    def check_ranked_pockets_in_cache(self, pl_complexes: list[ProteinLigandComplex]) -> bool:
        """
        Check whether the ranked pockets are already present in disk.
        :param pl_complexes: complexes containing complex ID.
        :return: boolean indicating whether the data is in the cache or not.
        """
        if not os.path.exists(self.pockets_saved_path):
            return False
        for pl_complex in pl_complexes:
            if pl_complex.name not in os.listdir(self.pockets_saved_path):
                return False
        return True

    def get_pockets(self, pl_complexes: list[ProteinLigandComplex],
                    p2rank_input_filename: str = "p2rank_input_file.txt",
                    p2rank_output_folder_name: str = "p2rank_output") -> Optional[list[ProteinLigandComplex]]:
        """
        Obtain a ranked list of the top pockets in a ligand-dependent fashion from a list of protein-ligand complexes.

        :param pl_complexes: a list of protein-ligand complexes containing PDB ID, path to the
            protein .pdb and either path to the ligand .sdf or .mol2 or ligand SMILES.
        :param p2rank_input_filename: name of the file generated from the PDB list in the format expected by p2rank.
        :param p2rank_output_folder_name: name of the directory used by p2rank to output its pocket predictions.
        :return: A list of PDB objects with the PDB ID,
            path to the top ranking protein pockets for the ligand and either path to the ligand or ligand SMILES.
        """
        p2rank_output_folder = os.path.join(self.p2rank_cache_path, p2rank_output_folder_name)

        if self.ignore_cache or not self.check_ranked_pockets_in_cache(pl_complexes):

            shutil.rmtree(self.p2rank_cache_path)
            shutil.rmtree(self.pockets_saved_path)

            logging.info("Running p2rank to predict protein pockets...")
            self._run_p2rank(proteins=[pl_complex.protein_path for pl_complex in pl_complexes],
                             p2rank_input_filename=p2rank_input_filename,
                             p2rank_output_folder=p2rank_output_folder)

            logging.info("Generating segmented pockets...")
            self._generate_segmented_pockets(protein_ids=[pl_complex.name for pl_complex in pl_complexes],
                                             protein_paths=[pl_complex.protein_path for pl_complex in pl_complexes],
                                             p2rank_output_folder=p2rank_output_folder)
        else:
            logging.info(f"Found pockets in {self.pockets_saved_path}. Skipping pocket segmentation."
                         f" To perform segmentation anyway, set the --ignore_cache flag.")

        if self.pocket_scoring_module is not None:
            logging.info("Running ligand-dependent pocket scoring and ranking...")
            self._rank_pockets(pl_complexes)
        else:
            logging.info("No pocket scoring module was supplied. Pocket ranking was not performed.")
            return None

        logging.info(f"Choosing top {self.top_k} pockets for each PDB...")
        top_pockets = []
        for pl_complex in pl_complexes:
            pdb_pockets_dir = os.path.join(self.ranked_pockets_path, pl_complex.name)
            for pocket in sorted(filter(lambda x: x.startswith(pl_complex.name) and "_rank" in x,
                                        os.listdir(pdb_pockets_dir)),
                                 key=lambda x: int(x.split("rank")[1][:-4]))[:self.top_k]:
                top_pockets.append(ProteinLigandComplex(name=pl_complex.name,
                                                        protein_path=os.path.join(pdb_pockets_dir, pocket),
                                                        ligand_path=pl_complex.ligand_path,
                                                        ligand_smiles=pl_complex.ligand_smiles,
                                                        ligand_reference_path=pl_complex.ligand_reference_path))
        return top_pockets

    def dock_to_pocket(self, pl_complexes: list[ProteinLigandComplex]):
        """
        Perform molecular docking using a pretrained model.

        :param pl_complexes: The list of protein-ligand complexes containing ID, path to the protein .pdb and path to
            the ligand .sdf or .mol2.
        :param predicted_ligands_path: path to saved predicted ligand positions. If None,
            defaults to self.docking_predictions_path.
        """

        logging.info("Running prediction module...")
        loader = DataLoader(PDBBindDataset(pl_complexes, include_absolute_coordinates=False, use_ligand_centroid=True,
                                           pocket_predictions_dir=f"{self.p2rank_cache_path}/p2rank_output"),
                            shuffle=False, batch_size=self.docking_batch_size)
        predictions = []
        for batch in tqdm(loader):
            predictions.extend(self.pocket_docking_module(batch))
        return predictions

    def predict(self, pl_complexes: list[ProteinLigandComplex], return_pockets: bool = False):
        """
        Run the full prediction pipeline on a list of protein-ligand complexes.

        :param pl_complexes: a list of ProteinLigandComplexes, containing ID, path to the protein,
        and either path to the ligand .sdf or ligand SMILES.
        :param return_pockets: whether to return the segmented pockets along with the predicted ligand positions.
        """
        logging.info("[START] Running prediction on the protein-ligand complexes...")

        assert self.pocket_scoring_module is not None, "Pocket scoring module not found. Prediction is not possible."
        assert self.pocket_docking_module is not None, "Pocket docking module not found. Prediction is not possible."

        start_time = time.process_time()
        segmented_ranked_pockets = self.get_pockets(pl_complexes)
        predictions = self.dock_to_pocket(segmented_ranked_pockets)
        elapsed_time = time.process_time() - start_time

        pockets_per_complex = len(segmented_ranked_pockets) / len(pl_complexes)
        logging.info(f"Inference took {elapsed_time * pockets_per_complex / len(segmented_ranked_pockets)} seconds per protein-ligand complex.")

        if return_pockets:
            return predictions, segmented_ranked_pockets
        return predictions

    def _evaluate_rmsd(self, pl_complex: ProteinLigandComplex, predicted_molecule_path: str):
        predicted_molecule = read_ligand(predicted_molecule_path, include_hydrogen=False)

        if predicted_molecule is not None:

            predicted_molecule = molecule.Molecule.from_rdkit(predicted_molecule)

            true_ligand_path = pl_complex.ligand_reference_path if pl_complex.ligand_reference_path is not None else pl_complex.ligand_path
            true_molecule = molecule.Molecule.from_rdkit(read_ligand(true_ligand_path, include_hydrogen=False))

            rmsd = symmrmsd(
                predicted_molecule.coordinates,
                true_molecule.coordinates,
                predicted_molecule.atomicnums,
                true_molecule.atomicnums,
                predicted_molecule.adjacency_matrix,
                true_molecule.adjacency_matrix,
            )

            return rmsd
        else:
            logging.error(f"Ligand for PDB {pl_complex.name} was not readable. Skipping it in the evaluation.")

    def _evaluate_validity(self, pl_complex: ProteinLigandComplex, predicted_molecule_path: str):
        true_molecule_path = pl_complex.ligand_reference_path if pl_complex.ligand_reference_path is not None else pl_complex.ligand_path
        out = self.evaluator.bust([predicted_molecule_path], true_molecule_path, pl_complex.protein_path)
        return out.drop(columns=["rmsd_≤_2å"]).sum(axis=1)[0]

    def evaluate(self, pl_complexes: list[ProteinLigandComplex]) -> dict:
        logging.debug("Deleting previous prediction folder.")
        shutil.rmtree(self.docking_predictions_path)
        predictions, pockets = self.predict(pl_complexes, return_pockets=True)

        logging.info("Running evaluations...")

        rmsd_dict = {}
        validity_dict = {}
        for pl_complex in tqdm(pl_complexes):
            pl_complex_predictions_path = os.path.join(self.docking_predictions_path, pl_complex.name)
            if os.listdir(pl_complex_predictions_path):
                final_prediction_path = os.path.join(pl_complex_predictions_path, f"{pl_complex.name}_ligand_prediction.sdf")
                sorted_ligands = sorted(filter(lambda x: x.endswith(".sdf") and x != os.path.basename(final_prediction_path),
                                               os.listdir(pl_complex_predictions_path)),
                                        key=lambda x: float(x.split("score")[1].split(".sdf")[0]))
                shutil.copyfile(os.path.join(pl_complex_predictions_path, sorted_ligands[0]), final_prediction_path)

                rmsd = self._evaluate_rmsd(pl_complex, final_prediction_path)
                
                if rmsd is not None:
                    rmsd_dict[pl_complex.name] = rmsd
                    validity_dict[pl_complex.name] = self._evaluate_validity(pl_complex, final_prediction_path)

        rmsd_values = np.array(list(rmsd_dict.values()))
        results = {
            "rmsd_under_1A": len(list(filter(lambda x: x <= 1, rmsd_values))) / len(rmsd_dict),
            "rmsd_under_2A": len(list(filter(lambda x: x <= 2, rmsd_values))) / len(rmsd_dict),
            "rmsd_under_5A": len(list(filter(lambda x: x <= 5, rmsd_values))) / len(rmsd_dict),
            "valid_and_under_1A": len(list(filter(lambda x: x[0] <= 1 and x[1] == 24, zip(rmsd_values, validity_dict.values())))) / len(validity_dict),
            "valid_and_under_2A": len(list(filter(lambda x: x[0] <= 2 and x[1] == 24, zip(rmsd_values, validity_dict.values())))) / len(validity_dict),
            "valid_and_under_5A": len(list(filter(lambda x: x[0] <= 5 and x[1] == 24, zip(rmsd_values, validity_dict.values())))) / len(validity_dict),
            "rmsd_quantiles": np.quantile(rmsd_values, np.arange(0, 1, 0.05))
        }

        return results
