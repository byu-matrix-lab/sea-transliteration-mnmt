
from pytorch_lightning import LightningModule
from torch.optim import Adam, AdamW
import torch
from transformers import BartConfig, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from tokenizers import Tokenizer
from shared_encoder_bart import BartForConditionalGeneration as SharedEncoderBartForConditionalGeneration
from dual_encoder_bart import BartForConditionalGeneration as DualEncoderBartForConditionalGeneration
from dual_encoder_bart import EncoderDecoderCache

class Model(LightningModule):
    def __init__(self, tokenizers: dict, config: dict):
        """
        tokenizers: A dictionary containing the tokenizers for the model, 
            specifically "o_tokenizer", "t_tokenizer", and "c_tokenizer" keys.
        config: The yaml configuration dictionary containing all necessary 
            hyperparameters and settings for the model.
        """
        super().__init__()
        self.tokenizers = tokenizers
        self.config = config
        self.pad_token_id = tokenizers['o_tokenizer'].token_to_id('[PAD]')

        # Get vocab_sizes based on architecture_type
        if config['architecture_type'] == 'no_transliteration':
            vocab_size = tokenizers['o_tokenizer'].get_vocab_size()
            phonetic_vocab_size = None  # No phonetic encoder in this case
        elif config['architecture_type'] == 'concatenated_input' or config['architecture_type'] == 'interlaced_input':
            vocab_size = tokenizers['c_tokenizer'].get_vocab_size()
            phonetic_vocab_size = None
        elif config['architecture_type'] == 'shared_encoder':
            vocab_size = tokenizers['c_tokenizer'].get_vocab_size()
            use_shared_encoder = True
        elif config['architecture_type'] == 'dual_encoder':
            vocab_size = tokenizers['o_tokenizer'].get_vocab_size()
            phonetic_vocab_size = tokenizers['t_tokenizer'].get_vocab_size()

        # Set up BartConfig object
        bart_config = BartConfig(
            vocab_size = vocab_size,
            max_position_embeddings = config.get('max_position_embeddings', 512),

            encoder_input_dim = None, # Check why this is different from before. It seems things have been updated. 
            encoder_layers = config.get('encoder_layers', 6),
            encoder_ffn_dim = config.get('encoder_ffn_dim', 2048),
            encoder_attention_heads = config.get('encoder_attention_heads', 8),
            encoder_layerdrop = config.get('encoder_layerdrop', 0.0),

            decoder_layers = config.get('decoder_layers', 6),
            decoder_ffn_dim = config.get('decoder_ffn_dim', 2048),
            decoder_attention_heads = config.get('decoder_attention_heads', 8),
            decoder_layerdrop = config.get('decoder_layerdrop', 0.0),

            activation_function = config.get('activation_function', 'relu'),
            d_model = config.get('d_model', 512),
            dropout = config.get('dropout', 0.1),
            attention_dropout = config.get('attention_dropout', 0.1),
            activation_dropout = config.get('activation_dropout', 0.0),
            classifier_dropout = config.get('classifier_dropout', 0.0),
            init_std = config.get('init_std', 0.02),
            scale_embedding = config.get('scale_embedding', True),
            is_encoder_decoder = config.get('is_encoder_decoder', True),
            tie_word_embeddings = config.get('tie_word_embeddings', True),
            pad_token_id = tokenizers['o_tokenizer'].token_to_id('[PAD]'),
            bos_token_id = tokenizers['o_tokenizer'].token_to_id('[BOS]'),
            eos_token_id = tokenizers['o_tokenizer'].token_to_id('[EOS]'),
        )

        if config['architecture_type'] == 'shared_encoder':
            # Use the custom BartForConditionalGeneration with shared encoder
            self.model = SharedEncoderBartForConditionalGeneration(bart_config)
        elif config['architecture_type'] == 'dual_encoder':
            # Add necessary attirbutes to bart_config
            bart_config.phonetic_encoder = True
            bart_config.phonetic_vocab_size = phonetic_vocab_size

            # Can add more attributes specific to the 2nd encoder if needed

            # Use the custom BartForConditionalGeneration with dual encoder
            self.model = DualEncoderBartForConditionalGeneration(bart_config)
        else:
            self.model = BartForConditionalGeneration(bart_config)
        
    
    def forward(self, **inputs):
        return self.model(**inputs)

    def prep_model_inputs(self, **input_kwargs):
        return input_kwargs

    def training_step(self, batch, batch_idx):
        if self.config['architecture_type'] == 'shared_encoder' or self.config['architecture_type'] == 'dual_encoder':
            src_items, phonetic_items, tgt_items, src_langs, tgt_langs = batch
            phonetic = [p.ids for p in phonetic_items]
            phonetic_mask = [p.attention_mask for p in phonetic_items]
        else:
            src_items, tgt_items, src_langs, tgt_langs = batch

        src = [s.ids for s in src_items]
        src_mask = [s.attention_mask for s in src_items]
        tgt = [t.ids for t in tgt_items]
        tgt_mask = [t.attention_mask for t in tgt_items]

        labels = torch.tensor(tgt).to(self.device)
        labels[labels == self.pad_token_id] = -100  # Ignore padding tokens in the loss

        if self.config['architecture_type'] == 'shared_encoder' or self.config['architecture_type'] == 'dual_encoder':
            model_inputs = self.prep_model_inputs(
                input_ids=torch.tensor(src).to(self.device),
                attention_mask=torch.tensor(src_mask).to(self.device),
                phonetic_input_ids=torch.tensor(phonetic).to(self.device),
                phonetic_attention_mask=torch.tensor(phonetic_mask).to(self.device),
                labels=labels,
                decoder_attention_mask=torch.tensor(tgt_mask).to(self.device)
            )
        else:
            model_inputs = self.prep_model_inputs(
                input_ids=torch.tensor(src).to(self.device),
                attention_mask=torch.tensor(src_mask).to(self.device),
                labels=labels,
                decoder_attention_mask=torch.tensor(tgt_mask).to(self.device)
            )

        outputs = self.model(**model_inputs)

        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.config['train_batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        '''
        Validate using loss
        '''
        if self.config['architecture_type'] == 'shared_encoder' or self.config['architecture_type'] == 'dual_encoder':
            src_items, phonetic_items, tgt_items, src_langs, tgt_langs = batch
            phonetic = [p.ids for p in phonetic_items]
            phonetic_mask = [p.attention_mask for p in phonetic_items]
        else:
            src_items, tgt_items, src_langs, tgt_langs = batch

        src = [s.ids for s in src_items]
        src_mask = [s.attention_mask for s in src_items]
        tgt = [t.ids for t in tgt_items]
        tgt_mask = [t.attention_mask for t in tgt_items]

        labels = torch.tensor(tgt).to(self.device)
        labels[labels == self.pad_token_id] = -100  # Ignore padding tokens in the loss

        if self.config['architecture_type'] == 'shared_encoder' or self.config['architecture_type'] == 'dual_encoder':
            model_inputs = self.prep_model_inputs(
                input_ids=torch.tensor(src).to(self.device),
                attention_mask=torch.tensor(src_mask).to(self.device),
                phonetic_input_ids=torch.tensor(phonetic).to(self.device),
                phonetic_attention_mask=torch.tensor(phonetic_mask).to(self.device),
                labels=labels,
                decoder_attention_mask=torch.tensor(tgt_mask).to(self.device)
            )
        else:
            model_inputs = self.prep_model_inputs(
                input_ids=torch.tensor(src).to(self.device),
                attention_mask=torch.tensor(src_mask).to(self.device),
                labels=labels,
                decoder_attention_mask=torch.tensor(tgt_mask).to(self.device)
            )
        
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.config['val_batch_size'])

        # Print some example output using generate function
        if batch_idx == 0 and self.global_rank == 0:
            generated_tokens = self.generate(
                input_ids=torch.tensor(src).to(self.device),
                attention_mask=torch.tensor(src_mask).to(self.device),
                phonetic_input_ids=torch.tensor(phonetic).to(self.device) if self.config['architecture_type'] in ['shared_encoder', 'dual_encoder'] else None,
                phonetic_attention_mask=torch.tensor(phonetic_mask).to(self.device) if self.config['architecture_type'] in ['shared_encoder', 'dual_encoder'] else None,
                num_beams=5,
                max_length=128,
            )
            tokenizer = self.tokenizers['c_tokenizer'] if self.config['architecture_type'] in ['concatenated_input', 'interlaced_input', 'shared_encoder'] else self.tokenizers['o_tokenizer']
            # print("Generated tokens: ", generated_tokens.shape)
            # print(generated_tokens)
            for i in range(self.config['num_val_prints']):
                print(f"\nGenerated text {i}:\t", tokenizer.decode(generated_tokens[i].tolist()))
                print(f"Target text {i}:\t\t", tokenizer.decode(tgt[i]))
                print()
            
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=float(self.config.get("learning_rate", 1e-4)))
        secheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.get("lr_step_size", 4000),
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': secheduler,
                'monitor': 'val_loss'
            }
        }

    def generate(self, input_ids, attention_mask=None, phonetic_input_ids=None, phonetic_attention_mask=None, num_beams=5, max_length=256, **kwargs):
        """
        Generate text from the model using the provided input_ids and attention_mask.
        Expects Tensors for input_ids and attention_mask, and optionally phonetic_input_ids and phonetic_attention_mask.
        """
        if self.config['architecture_type'] == 'shared_encoder':
            # Use the encoder before to ensure proper cache handling
            encoder_outputs = self.model.model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
            phonetic_encoder_outputs = self.model.model.get_encoder()(input_ids=phonetic_input_ids, attention_mask=phonetic_attention_mask)

            combined_hidden = encoder_outputs.last_hidden_state + phonetic_encoder_outputs.last_hidden_state
            combined_encoder_outputs = BaseModelOutput(last_hidden_state=combined_hidden)

            model_inputs = self.prep_model_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                phonetic_input_ids=phonetic_input_ids,
                phonetic_attention_mask=phonetic_attention_mask,
                encoder_outputs=combined_encoder_outputs,
                num_beams=num_beams,
                min_length=0,   
                max_length=max_length,
                **kwargs
            )
        elif self.config['architecture_type'] == 'dual_encoder':
            # Init EncoderDecoderCache
            cache = EncoderDecoderCache()
            model_inputs = self.prep_model_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                phonetic_input_ids=phonetic_input_ids,
                phonetic_attention_mask=phonetic_attention_mask,
                past_key_values=cache,
                num_beams=num_beams,
                min_length=0,   
                max_length=max_length,
                **kwargs
            )
        else:
            model_inputs = self.prep_model_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                min_length=0,   
                max_length=max_length,
                **kwargs
            )
        
        return self.model.generate(**model_inputs)
