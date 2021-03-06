import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from transformers import (AdamW, AutoConfig,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

warnings.filterwarnings('ignore')


class TextClassifier(pl.LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            label2id: List[str],
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            weight_decay: float = 0.0,
            warmup_steps: int = 0,
            predictions_file: str = 'predictions.pt',
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(
            self.hparams.model_name_or_path,
            num_labels=len(self.hparams.label2id),
            id2label={v: k for k, v in self.hparams.label2id.items()},
            label2id=self.hparams.label2id,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.model_name_or_path, config=self.config
        )
        self.precision_metric = pl.metrics.Precision(num_classes=len(self.hparams.label2id))
        self.recall_metric = pl.metrics.Recall(num_classes=len(self.hparams.label2id))
        self.accuracy_metric = pl.metrics.Accuracy()

    def metric(self, preds, labels, mode='val'):
        p = self.precision_metric(preds, labels)
        r = self.recall_metric(preds, labels)
        a = self.accuracy_metric(preds, labels)
        return {f'{mode}_precision': p, f'{mode}_recall': r, f'{mode}_acc': a}

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        del batch['idx']
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        del batch['idx']
        outputs = self(**batch)
        val_loss, logits = outputs.loss, outputs.logits
        preds = torch.argmax(logits, axis=1)
        metric_dict = self.metric(preds, batch['labels'])
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        idxs = batch.pop('idx')
        outputs = self(**batch)
        preds = torch.argmax(outputs.logits, axis=1)
        metric_dict = self.metric(preds, batch['labels'], mode='test')
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        # write test predictions to disk to view
        self.write_prediction('idxs', idxs, self.hparams.predictions_file)
        self.write_prediction('preds', preds, self.hparams.predictions_file)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

    @rank_zero_only
    def save_pretrained(self, save_dir):
        self.hparams.save_dir = save_dir
        self.model.save_pretrained(self.hparams.save_dir)
        self.tokenizer.save_pretrained(self.hparams.save_dir)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser.add_argument('--learning_rate', type=float, default=2e-5)
        parent_parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        parent_parser.add_argument('--weight_decay', type=float, default=0.0)
        parent_parser.add_argument('--warmup_steps', type=int, default=0)
        parent_parser.add_argument('--predictions_file', type=str, default='predictions.pt')
        return parent_parser
