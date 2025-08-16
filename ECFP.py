import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np


class ECFPModel(torch.nn.Module):
    def __init__(self, smiles_list, input_dim=2048, output_dim=128, radius=2):
        super(ECFPModel, self).__init__()
        self.smiles_list = smiles_list  # 数据集里的 SMILES
        self.radius = radius
        self.nBits = input_dim
        self.proj = torch.nn.Linear(input_dim, output_dim)
        self.fingerprint_dict = self.precompute_fingerprints(smiles_list)

    def precompute_fingerprints(self, smiles_list):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        import numpy as np

        fingerprint_dict = {}
        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.nBits)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprint_dict[idx] = arr
        return fingerprint_dict  # 以 index 索引

    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, indices=None):

        if indices is None:
            # 如果没有传索引，默认取所有
            fingerprints = np.stack(list(self.fingerprint_dict.values()))
        else:
            fingerprints = np.stack([self.fingerprint_dict[idx] for idx in indices])

        fingerprints = torch.tensor(fingerprints, dtype=torch.float32).to(self.proj.weight.device)
        embedding = self.proj(fingerprints)

        return None, None, embedding

