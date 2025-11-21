from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
DLKCAT_CODE_DIR = (ROOT_DIR / "models" / "DLKcat" / "DeeplearningApproach" / "Code" / "example")
DLKCAT_DATA_DIR = (ROOT_DIR / "data" / "DLKcat") 

if str(DLKCAT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(DLKCAT_CODE_DIR))


from .base import BaseModel
# from prediction_for_input import split_sequence
import model
import pandas as pd
import zipfile
import torch


class DLKcatWrapper(BaseModel):
    name = "DLKcat"

    def __init__(self):
        super().__init__()
        self._prepare_resources()

    def _prepare_resources(self):
        input_zip_file = ROOT_DIR / "models" / "DLKcat" / "DeeplearningApproach" / "Data" / "input.zip"
        
        if not input_zip_file.exists():
            raise FileNotFoundError(f"DLKcat resource zip not found at: {input_zip_file}")

        with zipfile.ZipFile(input_zip_file, 'r') as zip_ref:
            zip_ref.extractall(DLKCAT_DATA_DIR)

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        # Based on the predicition_for_input script from DLKcat

        fingerprint_dict = model.load_pickle(DLKCAT_DATA_DIR / "input" / "fingerprint_dict.pickle")
        atom_dict = model.load_pickle(DLKCAT_DATA_DIR / "input" / "atom_dict.pickle")
        bond_dict = model.load_pickle(DLKCAT_DATA_DIR / "input" / "bond_dict.pickle")
        word_dict = model.load_pickle(DLKCAT_DATA_DIR / "input" / "sequence_dict.pickle")
        
        n_fingerprint = len(fingerprint_dict)
        n_word = len(word_dict)

        radius=2
        ngram=3

        dim=10
        layer_gnn=3
        side=5
        window=11
        layer_cnn=3
        layer_output=3
        lr=1e-3
        lr_decay=0.5
        decay_interval=10
        weight_decay=1e-6
        iteration=100

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # torch.manual_seed(1234)
        Kcat_model = model.KcatPrediction(device, n_fingerprint, n_word, 2*dim, layer_gnn, window, layer_cnn, layer_output).to(device)
        Kcat_model.load_state_dict(torch.load('../../Results/output/all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50', map_location=device))
        # print(state_dict.keys())
        # model.eval()
        predictor = Predictor(Kcat_model)

        return pd.DataFrame()

class Predictor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        predicted_value = self.model.forward(data)

        return predicted_value

