from dataset import IterableMultilingualDataset, DataModule
from typing import Optional
from tqdm import tqdm
from transformers import BartConfig
from tokenizers import Tokenizer
from model import Model
import numpy as np
import random
import signal
import sys
import time
import torch
import os
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import yaml
import warnings

torch.set_float32_matmul_precision('high')

def inference(tokenizers: dict, config: dict, checkpoint: Optional[str] = None, inference_csv: Optional[str] = None):
    # Set up model
    model = Model.load_from_checkpoint(checkpoint, tokenizers=tokenizers, config=config)

    # To device
    device = 'cuda' if torch.cuda.is_available() and config['accelerator'] == 'gpu' else 'cpu'
    model = model.to(device)

    # Set up data module
    data_module = DataModule(
        ortho_tokenizer=tokenizers['o_tokenizer'],
        trans_tokenizer=tokenizers['t_tokenizer'],
        combined_tokenizer=tokenizers['c_tokenizer'],
        train_csv=None,
        val_csv=None,
        test_csv=inference_csv,
        data_type=config['architecture_type'],
        train_batch_size=config['train_batch_size'],
        val_batch_size=config['val_batch_size'],
        test_batch_size=config['inference_batch_size'],
        temperature=None,
        transliteration_scheme=config.get('transliteration_scheme', 'ipa'),
        transliterator_path=config.get('transliterator_json', None),
    )

    data_module.setup()

    print(f"Running inference on {config['inference_csv']}")

    # Loop through the test dataloader and perform inference
    test_dataloader = data_module.test_dataloader()
    results = []
    language_directions = []
    for batch in tqdm(test_dataloader, desc="Running inference"):
        if config['architecture_type'] == 'shared_encoder' or config['architecture_type'] == 'dual_encoder':
            src_items, phonetic_items, tgt_items, src_langs, tgt_langs = batch
            phonetic = [p.ids for p in phonetic_items]
            phonetic_mask = [p.attention_mask for p in phonetic_items]
            phonetic = torch.tensor(phonetic).to(model.device)
            phonetic_mask = torch.tensor(phonetic_mask).to(model.device)
        else:
            src_items, tgt_items, src_langs, tgt_langs = batch
            phonetic = None
            phonetic_mask = None
        
        src = [s.ids for s in src_items]
        src_mask = [s.attention_mask for s in src_items]
        tgt = [t.ids for t in tgt_items]
        tgt_mask = [t.attention_mask for t in tgt_items]

        # Move to device
        src = torch.tensor(src).to(model.device)
        src_mask = torch.tensor(src_mask).to(model.device)
        tgt = torch.tensor(tgt).to(model.device)
        tgt_mask = torch.tensor(tgt_mask).to(model.device)

        # Generate
        generated_tokens = model.generate(
            input_ids=src,
            attention_mask=src_mask,
            phonetic_input_ids=phonetic,
            phonetic_attention_mask=phonetic_mask,
            num_beams=config['num_beams'],
            max_length=config['max_length'],
        )

        tokenizer = tokenizers['c_tokenizer'] if config['architecture_type'] in ['concatenated_input', 'interlaced_input', 'shared_encoder'] else tokenizers['o_tokenizer']
        generated_texts = [tokenizer.decode(g.tolist(), skip_special_tokens=True) for g in generated_tokens]

        results.extend(generated_texts)
        language_directions.extend([f"{src_lang}-{tgt_lang}" for src_lang, tgt_lang in zip(src_langs, tgt_langs)])

    # Organize by language direction (maintaining order)
    language_results = {}
    for lang_dir, text in zip(language_directions, results):
        if lang_dir not in language_results:
            language_results[lang_dir] = []
        language_results[lang_dir].append(text)

    # Output results to config[output_dir] + '/preds/' + lang_pair + '.txt'
    output_dir = os.path.join(config['output_dir'], 'preds')
    os.makedirs(output_dir, exist_ok=True)
    for lang_pair, texts in language_results.items():
        lang_pair_file = os.path.join(output_dir, f"{lang_pair}.txt")
        with open(lang_pair_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text.strip() + '\n')
        print(f"Results for language pair {lang_pair} saved to {lang_pair_file}")

def train_many_to_many(tokenizers: dict, config: dict):
    # Set up model
    model = Model(tokenizers, config)

    # Set up data module
    data_module = DataModule(
        ortho_tokenizer=tokenizers['o_tokenizer'],
        trans_tokenizer=tokenizers['t_tokenizer'],
        combined_tokenizer=tokenizers['c_tokenizer'],
        train_csv=config['train_csv'],
        val_csv=config['val_csv'],
        data_type=config['architecture_type'],
        train_batch_size=config['train_batch_size'],
        val_batch_size=config['val_batch_size'],
        temperature=config['temperature'],
        transliteration_scheme=config.get('transliteration_scheme', 'ipa'),
        transliterator_path=config.get('transliterator_json', None),
    )

    # Set up model checkpoint callback
    model_checkpoint = ModelCheckpoint(
        dirpath=config['output_dir'] + "/checkpoints",
        filename="model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=config['save_top_k'],
        monitor='val_loss',
        mode='min',
    )

    # Set up early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['patience'],
        mode='min',
        verbose=True,
    )

    # Set up trainer
    trainer = Trainer(
        default_root_dir=config['output_dir'],
        max_epochs=config['max_epochs'],
        accelerator=config['accelerator'],
        devices=config['devices'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        strategy="ddp",
        callbacks=[early_stopping, model_checkpoint],
        logger=CSVLogger(config['output_dir'], 'logs'),
        log_every_n_steps=config['log_every_n_steps'],
        val_check_interval=config['val_check_interval'],
        plugins=SLURMEnvironment(
            requeue_signal=signal.SIGHUP,
        ),
        # check_val_every_n_epoch=20
    )

    # Fit the model
    trainer.fit(model, data_module)

def main():
    # Start time
    time_start = time.time()

    # Load config
    yaml_path = sys.argv[1]
    inference_flag = len(sys.argv) > 2
    checkpoint_path = sys.argv[2] if inference_flag else None
    inference_csv = sys.argv[3] if inference_flag and len(sys.argv) > 3 else None

    # Verify the checkpoint path if in inference mode
    if inference_flag and (not checkpoint_path or not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found for inference.")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file '{yaml_path}' not found.")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f) # Could add a verification of the config later...

    # Set seed
    seed_everything(config.get('seed', 8212021), workers=True)
    
    # Set up global rank stuff for printing only one time no matter how many GPUs are used
    global_rank = int(os.environ.get('SLURM_PROCID', 0))
    
    # Print training/experiment info
    if global_rank == 0:
        print("Configuration loaded successfully from:", yaml_path)
        print("Experiment name: ", config['experiment_name'])
        print("Output directory: ", config['output_dir'])
        print("Train CSV: ", config['train_csv'])
        print("Val CSV: ", config['val_csv'])
        print("Train (effective) batch size: ", config['train_batch_size']*config['devices']*config['accumulate_grad_batches'])
        print("Temperature: ", config['temperature'])
    
    # Set up tokenizers (all three) with padding enabled
    tokenizers = {
        "o_tokenizer": Tokenizer.from_file(config['orthographic_tokenizer_json']),
        "t_tokenizer": Tokenizer.from_file(config['transliteration_tokenizer_json']),
        "c_tokenizer": Tokenizer.from_file(config['combined_tokenizer_json'])
    }
    tokenizers["o_tokenizer"].enable_padding(pad_id=tokenizers["o_tokenizer"].token_to_id("[PAD]"), pad_token="[PAD]")
    tokenizers["t_tokenizer"].enable_padding(pad_id=tokenizers["t_tokenizer"].token_to_id("[PAD]"), pad_token="[PAD]")
    tokenizers["c_tokenizer"].enable_padding(pad_id=tokenizers["c_tokenizer"].token_to_id("[PAD]"), pad_token="[PAD]")

    tokenizers["o_tokenizer"].enable_truncation(max_length=config['max_position_embeddings'])
    tokenizers["t_tokenizer"].enable_truncation(max_length=config['max_position_embeddings'])
    tokenizers["c_tokenizer"].enable_truncation(max_length=config['max_position_embeddings'])

    if global_rank == 0: print("Tokenizers loaded successfully.\n") 

    if inference_flag:
        if global_rank == 0: print("Running in inference mode...\n")
        inference(tokenizers, config,
            checkpoint_path if checkpoint_path else config.get('checkpoint', None),
            inference_csv if inference_csv else config.get('inference_csv', None)
            )
    else:
        if global_rank == 0: print("Running in training mode...\n")
        # TODO: Pick the training function based on config
        train_many_to_many(tokenizers, config)

    # Print training time
    time_end = time.time()
    total_time_hours_minutes = time.strftime('%H:%M:%S', time.gmtime(time_end - time_start))
    print("Total time taken: ", total_time_hours_minutes)

if __name__ == "__main__":
    main()