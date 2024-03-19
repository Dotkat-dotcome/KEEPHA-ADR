"""..."""

from datasets import concatenate_datasets
from datasets import load_dataset
from datasets import load_from_disk
from datasets import load_metric

# from seqeval.metrics import classification_report
# from seqeval.scheme import BILOU
from datetime import datetime
from tqdm.auto import tqdm

from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import get_scheduler

from torch.optim import AdamW
from torch.utils.data import DataLoader

# import ipdb
from utils import utils
from utils.early_stopping import EarlyStopping

import argparse, evaluate, json
import numpy as np
import os
import pandas as pd
import torch
import wandb

data_description = {
    "description": """This is a description.""",
    # "citation": """""",
    #             """,
    # "homepage": "https://biosemantics.erasmusmc.nl/index.php/resources/mantra-gsc",
    # "url": "http://biosemantics.org/MantraGSC/Mantra-GSC.zip",
    # "url": "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/lifeline/",
    "file_name_blacklist": [],
}

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(DEVICE)


class NERTest(object):
    """docstring for NERTest"""

    def __init__(self, config, mode="eval"):
        super(NERTest, self).__init__()

        # load the "static" config params
        with open(config, "r") as read_handle:
            cf = json.load(read_handle)

        self.debug = cf["debug"]

        # load the dictionaries containing the languages and their IDs
        with open(cf["language2id_file"], "r") as rh_1, open(
            cf["language_dict_file"]
        ) as rh_2:
            self.lang2id = json.load(rh_1)
            self.id2lang = {idx: lang for lang, idx in self.lang2id.items()}
            self.lang_dict = json.load(rh_2)

        self.time = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

        if not self.debug:
            self.out_dir = cf["out_dir"]
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

        if cf["use_wandb"]:
            wandb.init(config=cf)
            self.cf = wandb.config
            self.debug = False
        else:
            wandb.init(mode="disabled")
            self.cf = cf


        self.data_urls = self.cf["data_url"]

        self.dataset = None
        self.model = None
        self.label_list = None
        self.label2id = None
        self.id2label = None

        self.metric = evaluate.load("seqeval")

        self.text_id = None
        self.subtokens = None

    def prepare_tokenizer(self):
        """Load a pre-trained tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cf["model_name"],
            use_fast=True,
            add_prefix_space=True,
            strip_accent=False,
        )

    def prepare_model(self):
        print(f"\navailable labels: {self.label_list}\n\n")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.cf["model_name"],
            num_labels=len(self.label_list),
            ignore_mismatched_sizes=True,
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def inference(self):
        """Run the test loop."""

        self.dataset.set_format("torch")
        data_collator = DataCollatorForTokenClassification(self.tokenizer, padding=True)

        test_dataloader = DataLoader(
            self.dataset["test"],
            batch_size=self.cf["batch_size"],
            collate_fn=data_collator,
        )

        self.model.to(DEVICE)
        self.model.eval()

        # progress_bar = tqdm(range(num_training_steps))
        eval_predictions = []
        eval_true_labels = []

        for batch in test_dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
            }

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            true_labels = batch["labels"].detach().cpu().numpy().tolist()
            eval_predictions.extend(predictions)
            eval_true_labels.extend(true_labels)

        results = utils.covert_predictions(
            self.tokenizer,
            self.id2label,
            self.text_id,
            self.text,
            eval_predictions,
            eval_true_labels,
            self.out_dir,
        )
        return 0

    def prepare_data(self):
        """..."""
        # path_to_preprocessed_data = self.cf["path_to_preprocessed_data"]

        # get tokenizer
        self.prepare_tokenizer()

        # If the data was already pre-processed once, we don't need to do it again,
        # just re-load it
        # if os.path.exists(path_to_preprocessed_data):
        #     print(
        #         f"Reloading already pre-processed data from '{path_to_preprocessed_data}'."
        #     )
        #     dataset = load_from_disk(path_to_preprocessed_data)

        #     if "preprocessed" in dataset:
        #         del dataset["preprocessed"]

        #     self.label2id, self.id2label, self.label_list = utils.get_label_id_dicts(
        #         config=self.cf
        #     )

        # # Otherwise, create one dataset out of each data URL
        # else:
        # for i, (data_url, lang) in enumerate(self.data_urls.items()):
        # replace the url to account for each path
        data_description["data_dir"] = self.data_urls

        dataset = load_dataset(
            "dfki-nlp/brat",
            num_proc=1,
            download_mode="force_redownload",
            **data_description,
            ignore_verifications=True,
        )
        dataset = dataset.map(batched=True)

        # concatenate the current dataset with the ones from the
        # other URLs
        # if i == 0:
        concatenated_datasets = dataset
        # else:
        assert (
            dataset["train"].features.type
            == concatenated_datasets["train"].features.type
        )

        # concatenated_datasets["train"] = concatenate_datasets(
        #     [concatenated_datasets["train"], dataset["train"]]
        # )

        # concatenated_datasets["dev"] = concatenate_datasets(
        #     [concatenated_datasets["dev"], dataset["dev"]]
        # )

        # concatenated_datasets["test"] = concatenate_datasets(
        #     [concatenated_datasets["test"], dataset["test"]]
        # )

        # convert offsets into IOB tags
        concatenated_datasets = concatenated_datasets.map(
            utils.convert_offsets_to_iob,
            batched=True,
            fn_kwargs={"tokenizer": self.tokenizer, "lang2id": self.lang2id},
        )

        # the label list is the same for each sample
        # self.label_list = concatenated_datasets["train"][0]["label_names"]

        self.label2id, self.id2label, self.label_list = utils.get_label_id_dicts(
            config=self.cf
        )

        # tokenize the sequences and align the labels to the sub-tokens
        tokenized_dataset = concatenated_datasets.map(
            utils.covert_labels,
            batched=True,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "label2id": self.label2id,
                "label_all_tokens": True,
            },
            # # remove all unnecessary columns
            # remove_columns=dataset["train"].column_names,
        )
        # TODO: chunk data
        # save the data to disc to save pre-processing time
        # if not self.debug:
        #     tokenized_dataset.save_to_disk(path_to_preprocessed_data)

        dataset = tokenized_dataset

        print(dataset)

        self.text_id = dataset["test"]["text_id"]
        self.text = dataset["test"]["text"]

        dataset = dataset.remove_columns(
            [
                col
                for col in dataset["test"].features
                if col
                not in [
                    "language",
                    "input_ids",
                    "token_type_ids",
                    "attention_mask",
                    "labels",
                ]
            ]
        )
        self.dataset = dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config", default=None, help="Path to config file.")

    args = parser.parse_args()

    ner_trainer = NERTest(args.config)

    ner_trainer.prepare_data()

    ner_trainer.prepare_model()

    ner_trainer.inference()
