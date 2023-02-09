# script to generate ProtTrans language model embeddings

import h5py
from pathlib import Path
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import gc
from typing import Sequence
from tqdm import tqdm
import os
from Bio import SeqIO
import esm

class ESMEmbedder():
    def __init__(self, model_name="esm1b_t33_650M_UR50S"):
        # self.model = esm.pretrained.esm1b_t33_650M_UR50S()
        self.model = getattr(esm.pretrained, model_name)

    def embed(self):
        pass

    def embed_with_cache(self):
        pass

class ProtTransEmbedder():
    def __init__(self, model_name = "Rostlab/prot_t5_xl_uniref50", device=None, half=True):
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        self.embedding_dim = self.model.config.to_dict()['d_model']
        gc.collect()
        if not device:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        if half:
            self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()

    def embed_cache(self, seqs: Sequence[str], names: Sequence[str], save_dir, batch_size=16, max_length=2000):
        sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in seqs]
        sequences = [seq[:max_length] for seq in sequences]

        for i in tqdm(range(len(sequences) // batch_size + 1)):
            features = []
            sequences_Example = sequences[i * batch_size: (i + 1) * batch_size]
            names_Example = names[i * batch_size: (i + 1) * batch_size]
            ids = self.tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)
            input_ids = torch.tensor(ids['input_ids'], device=self.device)
            attention_mask = torch.tensor(ids['attention_mask'], device=self.device)
            with torch.no_grad():
                # print(input_ids.device, attention_mask.device)
                embedding = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = embedding.last_hidden_state.cpu().numpy()
            for seq_num in range(len(embedding)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = embedding[seq_num][:seq_len - 1]
                features.append(seq_emd)
            # print(features[0].shape)
            assert len(names_Example) == len(features)
            for k in range(len(features)):
                np.save(f"{os.path.join(save_dir, names_Example[k])}.npy", features[k])
        
    def embed(self, seqs: Sequence[str], batch_size=160, max_length=1000):
        sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in seqs]
        sequences = [seq[:max_length] for seq in sequences]
        features = []
        for i in (range((len(sequences) + batch_size - 1) // batch_size)):
            sequences_Example = sequences[i * batch_size: (i + 1) * batch_size]
            ids = self.tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)
            input_ids = torch.tensor(ids['input_ids']).to(self.device)
            attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
            with torch.no_grad():
                embedding = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = embedding.last_hidden_state
            for seq_num in range(len(embedding)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = embedding[seq_num][:seq_len - 1]
                features.append(seq_emd)
        assert len(sequences) == len(features), print(len(sequences), len(features))
        torch.cuda.empty_cache()

        return features


def insert_str(origin: str, add=' '):
    inserted = add.join([c for c in origin])
    return inserted

def generate_embed_file(embedder, fasta_file, save_file):
    seqs = []
    names = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        names.append(record.id)
        seqs.append(insert_str(record.seq))
    tmp_save_dir = "./data/embeddings/prot_t5_xl_uniref50/"
    if not os.path.exists(tmp_save_dir):
        os.mkdir(tmp_save_dir)
    embedder.embed_cache(seqs, names, save_dir=tmp_save_dir)

    print("Begin generating h5 file:", save_file)
    dataset = h5py.File(save_file, 'a')
    for name in tqdm(names):
        tmp_data = np.load(f"{os.path.join(tmp_save_dir, name)}.npy")
        dataset.create_dataset(name=name, data=tmp_data, dtype=np.float16)
    keys = list(dataset.keys())
    print(f'Entries in {save_file}: {len(keys)}')
    dataset.close()


if __name__ == '__main__':
    root_dir = Path.cwd()
    save_dir = root_dir / "data/embeddings"
    data_dir = root_dir / "data"
    embedder = ProtTransEmbedder()
    generate_embed_file(embedder, data_dir / "test.fasta", save_dir / "test.h5")