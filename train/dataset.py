from pytorch_lightning import LightningDataModule
from tokenizers import Tokenizer # https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from transliterate import Transliterator
from typing import Optional
import os
import random
import numpy as np

class IterableMultilingualDataset(Dataset):
    def __init__(self, csv_file, temperature=None, shuffle=False):
        '''
        In the csv, if no entry is provided for the ipa column, 
        it is assumed that the IPA is not used and empty string is returned.
        '''
        self.paths = [] # Holds the raw paths to the source and target files
        self.data = [] # Holds every example in the dataset in order of the csv file
        self.pair_ranges = [] # Holds tuples of the start and end indices for each language pair in the self.data list. The start index is inclusive, the end index is exclusive.
        self.temperature = temperature # The temperature for sampling from the language pairs
        self.shuffle = shuffle # Whether to shuffle the examples in each language pair

        # Confirm the csv file exists
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"The csv file {csv_file} does not exist")

        # Read the csv file
        with open(csv_file, 'r') as f:
            header = f.readline()

            # Confirm the header is correct: src_lang,tgt_lang,src_orth_path,src_transliteration_path,tgt_ortho_path
            if header != "src_lang,tgt_lang,src_orthography_path,src_transliteration_path,tgt_orthography_path,tgt_transliteration_path\n":
                raise ValueError("The header of the input csv file must be 'src_lang,tgt_lang,src_orthography_path,src_transliteration_path,tgt_orthography_path,tgt_transliteration_path'")

            # Read the paths
            for line in f:
                if line == "\n" or line == "" or line == " ":
                    continue
                src_lang, tgt_lang, src_orth_path, src_transliteration_path, tgt_ortho_path, tgt_transliteration_path = line.strip().split(',')
                self.paths.append((src_lang, tgt_lang, src_orth_path, src_transliteration_path, tgt_ortho_path, tgt_transliteration_path))
            
        # Load the data
        for src_lang, tgt_lang, src_orth_path, src_transliteration_path, tgt_ortho_path, tgt_transliteration_path in tqdm(self.paths, desc="Reading data", total=len(self.paths), leave=True):
            with open(src_orth_path, 'r') as f: src_lines = f.readlines()
            with open(tgt_ortho_path, 'r') as f: tgt_lines = f.readlines()
            with open(src_transliteration_path, 'r') as f: src_transliteration_lines = f.readlines()
            with open(tgt_transliteration_path, 'r') as f: tgt_transliteration_lines = f.readlines()
            
            # Confirm the number of lines in the source and target files are the same
            assert len(src_lines) == len(tgt_lines) and len(src_lines) == len(src_transliteration_lines) and len(src_transliteration_lines) == len(tgt_transliteration_lines)

            if self.shuffle:
                zipped = list(zip(src_lines, src_transliteration_lines, tgt_lines, tgt_transliteration_lines))
                random.shuffle(zipped)
                src_lines, src_transliteration_lines, tgt_lines, tgt_transliteration_lines = zip(*zipped)

            for src, src_trans, tgt, tgt_trans in zip(src_lines, src_transliteration_lines, tgt_lines, tgt_transliteration_lines):
                self.data.append((src.strip(), src_trans.strip(), tgt.strip(), tgt_trans.strip(), src_lang, tgt_lang))

            self.pair_ranges.append((len(self.data) - len(src_lines), len(self.data), len(self.data) - len(src_lines)))

        self.temperature_probs = self._get_temperature_probs()

    def _get_temperature_probs(self):
        '''
        Returns the probabilities for each language pair based on the temperature
        If The temperature is greater than 1, the probabilities will be more smooth. 
        If the temperature is less than 1, the probabilities will be more confident.
        If the temperature is 1, the probabilities will be unchanged.
        '''
        if self.temperature is None:
            return None
        else:
            probs = []

            # Get the initial probabilities based on the number of examples in each language pair
            for i in range(len(self.pair_ranges)):
                start, end, current = self.pair_ranges[i]
                probs.append((end - start) / len(self.data))

            # Apply the temperature. 
            probs = np.array(probs) ** (1 / self.temperature) # Apply the temperature to the probabilities by taking the power of the reciprocal of the temperature
            probs = probs / np.sum(probs) # Normalize the probabilities

            return probs
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        '''
        Returns the source segment, target segment, source language, and target language for the i-th example of a sampled language pair
        '''
        if self.temperature is None:
            # No sampling, just iterate over the entire dataset.
            return self.data[i]
        else:
            # Sample from the dataset with temperature
            return self._sample()

    def _sample(self):
        '''
        Temperature-based sampling for which language pair to sample from though it goes in sequence for the selected language pair
        '''
        # Choose a dataset
        dataset_idx = np.random.choice(len(self.temperature_probs), p=self.temperature_probs)
        
        # Get the start and end indices for the language pair
        start, end, current = self.pair_ranges[dataset_idx]

        if current >= end:
            # Reset the current index for the language pair
            current = start

            # Shuffle the items in the language pair range
            if self.shuffle:
                shuffled_slice = self.data[start:end]
                random.shuffle(shuffled_slice)
                self.data[start:end] = shuffled_slice
        
        # Sample the current and increment
        src, src_trans, tgt, tgt_trans, src_lang, tgt_lang = self.data[current]
        self.pair_ranges[dataset_idx] = (start, end, current + 1)

        return src, src_trans, tgt, tgt_trans, src_lang, tgt_lang

class DataModule(LightningDataModule):
    """
    Handles loading and batching of the IterableMultilingualDataset in addition to...
    Returning the data in the proper format for each of the following experiment types via collate function: 
        - no_transliteration
        - concatenated_input
        - interlaced_input
        - shared_encoder
        - dual_encoder
    """
    def __init__(self, ortho_tokenizer, trans_tokenizer, combined_tokenizer, train_csv, val_csv, test_csv=None, data_type="no_transliteration", train_batch_size=32, val_batch_size=32, test_batch_size=32, temperature=None, transliteration_scheme="ipa", transliterator_path=None):
        super().__init__()
        self.ortho_tokenizer = ortho_tokenizer
        self.trans_tokenizer = trans_tokenizer
        self.combined_tokenizer = combined_tokenizer
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.data_type = data_type
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        if temperature == "" or temperature == 0:
            self.temperature = None
        elif temperature is None:
            self.temperature = None
        else:
            if not isinstance(temperature, (int, float)):
                raise ValueError("Temperature must be a number, got: {temperature}")
            self.temperature = temperature

        self.pad_id = self.ortho_tokenizer.token_to_id("[PAD]")
        if self.trans_tokenizer is not None:
            assert self.pad_id == self.trans_tokenizer.token_to_id("[PAD]")

        self.transliteration_scheme = transliteration_scheme
        if transliterator_path is not None:
            self.transliterator = Transliterator(transliterator_path)
    
    def setup(self, stage: Optional[str] = None):
        if self.test_csv is not None:
            self.test_dataset = IterableMultilingualDataset(self.test_csv, temperature=None, shuffle=False)
            print(f"Test dataset size: {len(self.test_dataset)}")
        else:
            self.train_dataset = IterableMultilingualDataset(self.train_csv, temperature=self.temperature, shuffle=True)
            self.val_dataset = IterableMultilingualDataset(self.val_csv, temperature=None, shuffle=False)
            print(f"Training dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")
            
        
    
    def collate_fn(self, batch):
        """
        For each item in the batch:
            - collect the necessary fields
                - "no_transliteration": src_ortho, tgt_ortho, src_lang, tgt_lang
                - "concatenated_input": src_ortho, src_trans, tgt_ortho, src_lang, tgt_lang
                - "interlaced_input":  src_ortho, src_trans, tgt_ortho, src_lang, tgt_lang (using space to split for simplicity, on the fly transliteration)
                - "shared_encoder": src_ortho, src_trans, tgt_ortho, src_lang, tgt_lang
                - "dual_encoder": src_ortho, src_trans, tgt_ortho, src_lang, tgt_lang
            - pad to the longest sequence in the batch
        """
        src, src_trans, tgt, tgt_trans, src_lang, tgt_lang = zip(*batch)

        # Add BOS and EOS to tgt segments
        tgt = ["[BOS]" + t + "[EOS]" for t in tgt]

        if self.data_type == "concatenated_input":
            # Prepend tgt language tokens to source segments
            src = [lang + s for s, lang in zip(src, tgt_lang)]
            src_trans = [lang + s for s, lang in zip(src_trans, tgt_lang)]

            concat_src = ["[BOS]" + s + "[SEP]" + t + "[EOS]" for s, t in zip(src, src_trans)]
            src_toks = self.combined_tokenizer.encode_batch(concat_src)
            tgt_toks = self.combined_tokenizer.encode_batch(tgt)
            
            return src_toks, tgt_toks, src_lang, tgt_lang

        elif self.data_type == "interlaced_input":
            # Using only the src, transliterate on the fly and transliterate the input split on spaces before tokenizing.
            interlaced_src = []
            for s, lang in zip(src, src_lang):
                if lang != "[ENGLISH]":
                    split_s = s.split(" ")
                    interlaced_s = ""
                    for word in split_s:
                        transliterated_word = self.transliterator.transliterate(word, transliteration_method=self.transliteration_scheme)
                        interlaced_s += word + " " + transliterated_word + " "
                    interlaced_src.append(interlaced_s.strip())
                else:
                    interlaced_src.append(s)
            
            # Prepend tgt language tokens to source segments
            interlaced_src = ["[BOS]" + lang + s + "[EOS]" for s, lang in zip(interlaced_src, tgt_lang)]

            src_toks = self.combined_tokenizer.encode_batch(interlaced_src)
            tgt_toks = self.combined_tokenizer.encode_batch(tgt)

            return src_toks, tgt_toks, src_lang, tgt_lang
        elif self.data_type == "shared_encoder":
            # Prepend tgt language tokens to source segments
            src = ["[BOS]" + lang + s + "[EOS]" for s, lang in zip(src, tgt_lang)]
            src_trans = ["[BOS]" + lang + s + "[EOS]" for s, lang in zip(src_trans, tgt_lang)]

            # When compared to the dual encoder, the difference is we are using a single tokenizer for both the source and transliterated source segments.
            src_toks = self.combined_tokenizer.encode_batch(src)
            src_trans_toks = self.combined_tokenizer.encode_batch(src_trans)
            tgt_toks = self.combined_tokenizer.encode_batch(tgt)

            # TODO: MAKE ALL OF BELOW ACTUALLY EFFICIENT...
            # Get the longest of all tokenized segments to ensure padding is consistent
            pad_len = max(len(src_toks[0]), len(src_trans_toks[0]))

            # Pad the tokenized segments to the longest length
            self.combined_tokenizer.enable_padding(pad_id=self.pad_id, pad_token="[PAD]", length=pad_len)

            # Repad
            src_toks = self.combined_tokenizer.encode_batch(src)
            src_trans_toks = self.combined_tokenizer.encode_batch(src_trans)

            # Set the padding back
            self.combined_tokenizer.enable_padding(pad_id=self.pad_id, pad_token="[PAD]")

            return src_toks, src_trans_toks, tgt_toks, src_lang, tgt_lang
        elif self.data_type == "dual_encoder":
            # Prepend tgt language tokens to source segments
            src = ["[BOS]" + lang + s + "[EOS]" for s, lang in zip(src, tgt_lang)]
            src_trans = ["[BOS]" + lang + s + "[EOS]" for s, lang in zip(src_trans, tgt_lang)]

            # When compared to the shared encoder, the difference is we are using two separate tokenizers for the source and transliterated source segments.
            src_toks = self.ortho_tokenizer.encode_batch(src)
            src_trans_toks = self.trans_tokenizer.encode_batch(src_trans)

            tgt_toks = self.ortho_tokenizer.encode_batch(tgt)

            return src_toks, src_trans_toks, tgt_toks, src_lang, tgt_lang
        else:
            # Default to no_transliteration
            # Prepend tgt language tokens to source segments
            src = ["[BOS]" + lang + s + "[EOS]" for s, lang in zip(src, tgt_lang)]

            src_toks = self.ortho_tokenizer.encode_batch(src)
            tgt_toks = self.ortho_tokenizer.encode_batch(tgt)

            return src_toks, tgt_toks, src_lang, tgt_lang
            

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )


def test_datamodule():
    ortho_tokenizer = Tokenizer.from_file("testing_data/tokenizer_ortho.json")
    ortho_tokenizer.enable_padding(pad_id=ortho_tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    trans_tokenizer = Tokenizer.from_file("testing_data/tokenizer_trans.json")
    trans_tokenizer.enable_padding(pad_id=trans_tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    combined_tokenizer = Tokenizer.from_file("testing_data/tokenizer_combined.json")
    combined_tokenizer.enable_padding(pad_id=combined_tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    train_csv = "train/testing_data/csvs/small.csv"
    val_csv = "train/testing_data/csvs/small.csv"

    # Test train dataloader - default no_transliteration
    dm = DataModule(
        ortho_tokenizer=ortho_tokenizer,
        trans_tokenizer=trans_tokenizer,
        combined_tokenizer=combined_tokenizer,
        train_csv=train_csv,
        val_csv=val_csv,
        train_batch_size=2,
        val_batch_size=4,
        temperature=1000
    )
    dm.setup()
    
    print("Testing datamodule with no_transliteration...")
    for i in range(2):
        src_items, tgt_items = next(iter(dm.train_dataloader()))

        print(src_items.ids)

        exit()

        print(f"\nBatch {i}:")
        for src_item, tgt_item in zip(src_items, tgt_items):
            print(f"Source: {src_item.ids}\nTarget: {tgt_item.ids}\n")
    



def test_dataset():
    dataset = IterableMultilingualDataset(
        "train/testing_data/csvs/small.csv",
        temperature=1,
        shuffle=True
    )

    # Test looping through 5 times
    print("Testing dataset...")

    # Test temperature sampling
    num_tha = 0
    num_khm = 0

    for i in range(1500):
        src, src_trans, tgt, tgt_trans, src_lang, tgt_lang = dataset[i]

        if src_lang == "tha":
            num_tha += 1
        elif tgt_lang == "khm":
            num_khm += 1

    print(f"Number of Thai examples: {num_tha}")
    print(f"Number of Khmer examples: {num_khm}")

if __name__ == "__main__":
    # Set seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # test_dataset()
    test_datamodule()