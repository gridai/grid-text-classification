import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pytorch_lightning as pl

from data import IMDBDataModule
from model import TextClassifier

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', default=os.getcwd(), type=str, help="IMDB data directory")
    parser.add_argument('--imdb', default="https://pl-flash-data.s3.amazonaws.com/imdb.zip", type=str, help="IMDB data source")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = IMDBDataModule.add_argparse_args(parser)
    parser = TextClassifier.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    if not(os.path.isdir(args.datadir+"/imdb")):
        os.makedirs(args.datadir, exist_ok=True)
        with urlopen(args.imdb) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(args.datadir)
    
    dm = IMDBDataModule.from_argparse_args(args)
    dm.setup('fit',datadir=args.datadir)
    model = TextClassifier(
        model_name_or_path=dm.model_name_or_path,
        label2id=dm.label2id,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        predictions_file=args.predictions_file
    )
    model.tokenizer = dm.tokenizer
    model.total_steps = (
            (len(dm.ds['train']) // (args.batch_size * max(1, (args.gpus or 0))))
            // args.accumulate_grad_batches
            * float(args.max_epochs)
    )
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    model.save_pretrained("/lightning_logs/outputs")
