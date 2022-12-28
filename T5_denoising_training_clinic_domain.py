import torch
import argparse
from glob import glob
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import numpy as np
import io
from torch.utils.data import Dataset,DataLoader
from transformers import (Adafactor, T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer)

PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese-v1.1"

USE_GPU = torch.cuda.is_available()
torch.cuda.empty_cache()
pl.seed_everything(23)


args_dict = dict(
    data_dir = "/mnt/neliochen/mMSMARCO_QGdata/*/",
    model_name_or_path = PRETRAINED_MODEL_NAME,
    tokenizer_name_or_path = PRETRAINED_MODEL_NAME,
    learning_rate = 1e-3,
    weight_decay = 0.0,
    eps = (1e-30, 1e-3),
    clip_threshold = 1.0,
    decay_rate = -0.8,
    beta1 = None,
    relative_step = False,
    scale_parameter = False,
    warmup_init = False,
    gradient_accumulation_steps = 4,
    accelerator = "gpu" if USE_GPU else "auto",
    fp_16 = False,
    num_train_epochs = 2,
    train_batch_size = 8,
   
)

tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

class denoisingDataset(Dataset):
    def __init__(self, tokenizer, data_dir, input_max_len=256, target_max_length=64):
        self.input_max_len = input_max_len
        self.target_max_len = target_max_length
        self.tokenizer = tokenizer
        self.file_path = data_dir
        self.inputs, self.targets = self.read_data_files() 

    def __len__(self):
        return len(self.inputs)

    def read_data_files(self):
        inputs_list, targets_list = zip(*[l.split("\t") for l in
                               io.open(self.file_pathpath ,
                                       encoding='utf8').read().splitlines()])
        inputs = np.array(inputs_list, dtype = object)
        del inputs_list
        targets = np.array(targets_list, dtype = object)
        del targets_list    
        return inputs, targets        


    def __getitem__(self, index):
        input_text = self.inputs[index]
        target_text = self.targets[index]
    
        encoding_input = self.input_encoding_build(input_text)
        encoding_target = self.target_encoding_build(target_text)
       
        source_ids = encoding_input["input_ids"].squeeze()
        target_ids = encoding_target["input_ids"].squeeze()

        source_mask = encoding_input["attention_mask"].squeeze()
        target_mask = encoding_target["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask,
                "target_ids": target_ids, "target_mask": target_mask}        

    def input_encoding_build(self, input_text):
        return self.tokenizer(
            input_text, max_length = self.input_max_len, truncation = True, padding = "max_length", return_tensors = "pt"
                    )
    
    def target_encoding_build(self,target_text):
        return self.tokenizer(
            target_text, max_length = self.target_max_len, truncation = True, padding = "max_length", return_tensors = "pt"
                    )



class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        self.save_hyperparameters(hparams)

    def forward(self, input_ids, attention_mask = None, decoder_input_ids = None, decoder_attention_mask = None, labels = None):
        return self.model(
        input_ids,
        attention_mask = attention_mask,
        decoder_input_ids = decoder_input_ids,
        decoder_attention_mask = decoder_attention_mask,
        labels = labels,
        ) 

    def _step(self, batch):
        labels = batch["target_ids"]
        #labels set to -100 are ignored, will not be computed in trainging
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids = batch["source_ids"],
            attention_mask = batch["source_mask"],
            labels = labels,
            decoder_attention_mask = batch["target_mask"]
        
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, on_step = True, on_epoch = True, prog_bar = True, logger = True, sync_dist = True, rank_zero_only = True)
        return loss


    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.hparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        ]
        optimizer = Adafactor(optimizer_grouped_parameters, lr = self.hparams.learning_rate, eps = self.hparams.eps, 
                              clip_threshold = self.hparams.clip_threshold, decay_rate = self.hparams.decay_rate,
                              beta1 = self.hparams.beta1, weight_decay = self.hparams.weight_decay, relative_step = self.hparams.relative_step,
                              scale_parameter = self.hparams.scale_parameter, warmup_init = self.hparams.warmup_init)
        self.optimizer = optimizer
        return [optimizer]


    def get_dataset(self, tokenizer, args):
        return denoisingDataset(
            tokenizer = tokenizer,
            data_dir = args.data_dir,
        
        )    
    

    def setup(self, stage = None):
        if stage == 'fit' or None:
            train_dataset = self.get_dataset(tokenizer = self.tokenizer, type_path = "train_dataset-0.parquet", args = self.hparams)
            self.train_dataset = train_dataset

            
    def train_dataloader(self):
        sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=False)
        dataloader = DataLoader(self.train_dataset, batch_size=self.hparams.train_batch_size, sampler=sampler, shuffle=False, drop_last=True)
        return dataloader
                 

logger = TensorBoardLogger(save_dir = "/mnt/neliochen/tb_logs/", name = "Adafactor-1e-3-mMSMARCOmodel") 

checkpoint_callback = ModelCheckpoint(
    monitor = "val_loss", dirpath = "/mnt/neliochen/models/", filename = "0.05mMSMARCO-QG-{epoch:02d}-{val_loss:.2f}", every_n_epochs = 1)

args = argparse.Namespace(**args_dict)    

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    max_epochs=args.num_train_epochs,
    precision= 16 if args.fp_16 else 32,
    callbacks=[checkpoint_callback],
    logger = logger,
    devices = -1,
    strategy = "ddp",
    replace_sampler_ddp=False
)

def run():

    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)

    trainer.fit(model)                