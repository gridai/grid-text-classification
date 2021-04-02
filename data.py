from argparse import ArgumentParser
from typing import Optional

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def convert_to_features(example_batch, indices, tokenizer, text_fields, padding, truncation, max_length):
    # Either encode single sentence or sentence pairs
    if len(text_fields) > 1:
        texts_or_text_pairs = list(zip(example_batch[text_fields[0]], example_batch[text_fields[1]]))
    else:
        texts_or_text_pairs = example_batch[text_fields[0]]

    # Tokenize the text/text pairs
    features = tokenizer.batch_encode_plus(
        texts_or_text_pairs, padding=padding, truncation=truncation, max_length=max_length
    )

    # idx is unique ID we can use to link predictions to original data
    features['idx'] = indices

    return features


def preprocess(ds, tokenizer, text_fields, padding='max_length', truncation='only_first', max_length=128):
    ds = ds.map(
        convert_to_features,
        batched=True,
        with_indices=True,
        fn_kwargs={
            'tokenizer': tokenizer,
            'text_fields': text_fields,
            'padding': padding,
            'truncation': truncation,
            'max_length': max_length,
        },
    )
    return ds


def transform_labels(example, idx, label2id: dict):
    str_label = example["labels"]
    example['labels'] = label2id[str_label]
    example['idx'] = idx
    return example


class TextClassificationDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name_or_path: str = 'bert-base-uncased',
            padding: str = 'max_length',
            truncation: str = 'only_first',
            max_length: int = 128,
            batch_size: int = 16,
            num_workers: int = 8,
            use_fast: bool = True,
            seed: int = 42,
            train_file: Optional[str] = None,
            valid_file: Optional[str] = None,
            test_file: Optional[str] = None
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_fast = use_fast
        self.seed = seed
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.tokenizer = None

    def setup(self, stage: Optional[str] = None):

        data_files = {}

        if self.train_file is not None:
            data_files["train"] = self.train_file
        if self.valid_file is not None:
            data_files["validation"] = self.valid_file
        if self.test_file is not None:
            data_files["test"] = self.test_file

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=self.use_fast)
        self.ds = load_dataset(self.dataset_name, self.subset_name, data_files=data_files)

        if self.train_val_split is not None:
            split = self.ds['train'].train_test_split(self.train_val_split)
            self.ds['train'] = split['train']
            self.ds['validation'] = split['test']

        if self.target != "label":
            self.ds.rename_column_(self.target, "labels")

        self.ds = preprocess(self.ds, self.tokenizer, self.text_fields, self.padding, self.truncation, self.max_length)

        if self.do_transform_labels:
            self.ds = self.ds.map(
                transform_labels,
                with_indices=True,
                fn_kwargs={'label2id': self.label2id}
            )

        cols_to_keep = [
            x
            for x in ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'idx']
            if x in self.ds['train'].features
        ]
        self.ds.set_format("torch", columns=cols_to_keep)

    def train_dataloader(self):
        return DataLoader(self.ds['train'], batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds['validation'], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds['test'], batch_size=self.batch_size, num_workers=self.num_workers)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
        parent_parser.add_argument('--padding', type=str, default='max_length')
        parent_parser.add_argument('--truncation', type=str, default='only_first')
        parent_parser.add_argument('--max_length', type=int, default=128)
        parent_parser.add_argument('--batch_size', type=int, default=16)
        parent_parser.add_argument('--num_workers', type=int, default=8)
        parent_parser.add_argument('--use_fast', type=bool, default=True)
        parent_parser.add_argument('--seed', type=int, default=42)
        parent_parser.add_argument('--train_file', type=str, default=None)
        parent_parser.add_argument('--valid_file', type=str, default=None)
        parent_parser.add_argument('--test_file', type=str, default=None)
        return parent_parser


class IMDBDataModule(TextClassificationDataModule):
    dataset_name = 'csv'
    subset_name = None
    text_fields = ['review']
    target = "sentiment"
    label2id = {"negative": 0, "positive": 1}
    do_transform_labels = True
    train_val_split = None
