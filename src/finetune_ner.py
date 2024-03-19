"""..."""

from datasets import concatenate_datasets
from datasets import load_dataset
from datasets import load_from_disk
from datasets import load_metric

from datetime import datetime
from seqeval.metrics import classification_report
from seqeval.scheme import BILOU
from tqdm.auto import tqdm

from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import get_scheduler

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler


from utils import utils
from utils.early_stopping import EarlyStopping

import argparse
import evaluate
import json
import numpy as np
import os
import torch
import wandb

import logging

data_description = {
    "description": """This is a description.""",
    # "citation": """""",
    #             """,
    # "homepage": "https://biosemantics.erasmusmc.nl/index.php/resources/mantra-gsc",
    # "url": "http://biosemantics.org/MantraGSC/Mantra-GSC.zip",
    # "url": "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/lifeline/",
    "file_name_blacklist": [],
}


PHI_TOKENS = [
    "<user>",
    "<pi>",
    "<doc>",
    "<doctor>",
    "<url>",
    "<date>",
]


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(DEVICE)


class NERTraining(object):
    """docstring for NERTraining"""

    def __init__(self, config, mode="train"):
        super(NERTraining, self).__init__()

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

        if cf["use_wandb"]:
            wandb.init(config=cf)
            self.cf = wandb.config
            self.debug = False

        if not self.debug:
            self.out_dir = utils.create_output_dir(config=self.cf, time=self.time)

        else:
            wandb.init(mode="disabled")
            self.cf = cf

        print(self.cf)

        # since we have several directories to get the data from,
        # we have to create a dataset for each directory (TODO: really?)
        # self.data_urls = utils.prepare_data_urls(
        #     config=self.cf, lang_dict=self.lang_dict
        # )
        self.data_urls = self.cf["data_url"]

        self.dataset = None
        self.model = None
        self.label_list = None
        self.label2id = None
        self.id2label = None
        self.language_count = []

        self.metric = evaluate.load("seqeval")

    def prepare_tokenizer(self):
        """Load a pre-trained tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cf["model_name"],
            use_fast=True,
            add_prefix_space=True,
            strip_accent=False,
            # additional_special_tokens=,
        )

        num_added_toks = self.tokenizer.add_tokens(PHI_TOKENS)
        print("We have added", num_added_toks, "tokens")

    def prepare_model(self):
        print(self.label_list)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.cf["model_name"],
            num_labels=len(self.label_list),
            ignore_mismatched_sizes=True,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        # Notice: resize_token_embeddings expect to receive the full size of
        # the new vocabulary, i.e., the length of the tokenizer.
        self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_scheduler_and_optimizer(self, wu_steps, train_steps):
        """..."""
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.cf["learning_rate"],
            weight_decay=self.cf["weight_decay"],
        )

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=wu_steps,
            num_training_steps=train_steps,
        )

        return lr_scheduler, optimizer

    # def compute_metrics(self, predictions, labels, languages):
    def compute_metrics(self, predictions, labels):
        """Compute metrics for sequence labeling.

        Compute metrics using the HF seqeval implementation
        (https://huggingface.co/spaces/evaluate-metric/seqeval)

        We further collect the *best macro F1* score overall for model optimization.
        """
        print("Computing metrics ... ")

        # Remove ignored index (special tokens)
        cleaned_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        # print(cleaned_predictions)

        # print(true_labels)
        # calculate results overall
        cls_report_dict = classification_report(
            y_true=true_labels,
            y_pred=cleaned_predictions,
            mode="strict",
            scheme=BILOU,
            output_dict=True,
            digits=5,
        )

        cls_report = classification_report(
            y_true=true_labels,
            y_pred=cleaned_predictions,
            mode="strict",
            scheme=BILOU,
            output_dict=False,
            digits=5,
        )
        print(cls_report)
        # wandb.log({"overall": cls_report_dict})

        # return the classification report over all languages
        return cls_report_dict

    def train_model(self):
        """Run the training loop."""

        self.dataset.set_format("torch")
        data_collator = DataCollatorForTokenClassification(self.tokenizer, padding=True)

        print("Using weighted sampling")
        sampler = utils.get_balancing_sampler(
            self.dataset["train"],
            language_count=self.language_count,
            device=DEVICE,
        )
        # sampler = RandomSampler(self.dataset["train"])

        train_dataloader = DataLoader(
            self.dataset["train"],
            sampler=sampler,
            batch_size=self.cf["batch_size"],
            collate_fn=data_collator,
        )
        eval_dataloader = DataLoader(
            self.dataset["dev"],
            batch_size=self.cf["batch_size"],
            collate_fn=data_collator,
        )

        es = EarlyStopping(patience=self.cf["patience"], mode="max")

        if self.debug:
            num_epochs = 30
            # eval_steps = 5
            # save_steps = 10
            warmup_steps = 10

        else:
            num_epochs = self.cf["epochs"]
            eval_steps = 500
            save_steps = 500
            warmup_steps = self.cf.get("warmup_steps", 200)

        num_training_steps = num_epochs * len(train_dataloader)

        lr_scheduler, optimizer = self.prepare_scheduler_and_optimizer(
            wu_steps=warmup_steps, train_steps=num_training_steps
        )

        self.model.to(DEVICE)

        progress_bar = tqdm(range(num_training_steps))
        best_macro_f1 = 0.0
        best_model = None

        for epoch in range(num_epochs):
            wandb.log({"current_epoch": epoch})

            # set model to train mode
            self.model.train()

            for batch in train_dataloader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}

                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "labels": batch["labels"],
                }

                outputs = self.model(**inputs)
                loss = outputs.loss
                wandb.log({"train/loss": loss})
                # print("train/loss", str(loss))
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # set model to evaluation mode
            self.model.eval()

            eval_predictions = []
            # eval_languages = []
            eval_true_labels = []

            for batch in eval_dataloader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}

                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "labels": batch["labels"],
                }

                with torch.no_grad():
                    outputs = self.model(**inputs)

                eval_loss = outputs.loss
                wandb.log({"eval/loss": eval_loss})
                # print("eval/epoch", epoch)
                # print("eval/loss", str(eval_loss))

                logits = outputs.logits
                predictions = (
                    torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
                )
                true_labels = batch["labels"].detach().cpu().numpy().tolist()

                eval_predictions.extend(predictions)
                eval_true_labels.extend(true_labels)

            results = self.compute_metrics(
                predictions=eval_predictions,
                labels=eval_true_labels,
            )

            current_macro_f1 = results["macro avg"]["f1-score"]
            current_micro_f1 = results["micro avg"]["f1-score"]
            # print("eval/f1", current_macro_f1, current_micro_f1)
            # print(f"\n{results}\n")
            wandb.log({"macro_f1": current_macro_f1})
            wandb.log({"micro_f1": current_micro_f1})

            # save best F1 and model
            if current_macro_f1 > best_macro_f1:
                best_macro_f1 = current_macro_f1
                best_model = self.model
                wandb.log(
                    {
                        "best_macro_f1": best_macro_f1,
                        "epoch_of_best_f1": epoch,
                    }
                )

                if self.cf["save_best_model"]:
                    print("Saving best model")
                    if not self.debug:
                        wandb.log({"model_id": self.out_dir})

                        self.tokenizer.save_pretrained(self.out_dir)

                        best_model.save_pretrained(self.out_dir)

            if es.step(current_macro_f1):
                print(f"Stopping training with F1 of {current_macro_f1}.")
                break

        return best_model

    def prepare_data(self):
        """..."""
        # path_to_preprocessed_data = f"{self.cf['path_to_preprocessed_data']}/concatenated_dataset_{self.cf['language']}"
        path_to_preprocessed_data = f"{self.cf['path_to_preprocessed_data']}/"

        print(path_to_preprocessed_data)

        # get tokenizer
        self.prepare_tokenizer()

        # If the data was already pre-processed once, we don't need to do it again,
        # just re-load it
        if os.path.exists(path_to_preprocessed_data):
            print(
                f"Reloading already pre-processed data from '{path_to_preprocessed_data}'."
            )
            dataset = load_from_disk(path_to_preprocessed_data)
            # dataset = load_dataset(path_to_preprocessed_data)
            if "preprocessed" in dataset:
                del dataset["preprocessed"]
            # the label list is the same for each sample
            # self.label_list = dataset["train"][0]["label_names"]

            self.label2id, self.id2label, self.label_list = utils.get_label_id_dicts(
                config=self.cf
            )

        # Otherwise, create one dataset out of each data URL
        else:
            # for i, (data_url, lang) in enumerate(self.data_urls.items()):
            # replace the url to account for each path
            data_description["data_dir"] = self.data_urls
            dataset = load_dataset(
                "dfki-nlp/brat",
                num_proc=8,
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
            if not self.debug:
                tokenized_dataset.save_to_disk(path_to_preprocessed_data)

            dataset = tokenized_dataset

        print(f"dataset: {dataset}")

        dataset = dataset.remove_columns(
            [
                col
                for col in dataset["train"].features
                if col
                not in [
                    "language",
                    # "text_id",
                    "input_ids",
                    "token_type_ids",
                    "attention_mask",
                    "labels",
                ]
            ]
        )

        self.dataset = dataset.shuffle(seed=self.cf["seed"])
        languages = self.dataset["train"]["language"]

        self.language_count = np.bincount([x for x in languages])
        print(self.language_count)


# if __name__ == "__main__":
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("config", default=None, help="Path to config file.")

    args = parser.parse_args()

    ner_trainer = NERTraining(args.config)

    ner_trainer.prepare_data()

    ner_trainer.prepare_model()

    trained_model = ner_trainer.train_model()


if __name__ == "__main__":
    main()

# ----------------------
#  hp search
# ----------------------
# if __name__ == "__main__":
#     sweep_config = {
#         'method': 'random', #grid, random
#         'metric': {
#         'name': 'best_macro_f1',
#         'goal': 'maximize'
#         },
#         'parameters': {

#             'learning_rate': {
#                 'values': [ 5e-5, 3e-5, 2e-5]
#             },
#             'batch_size': {
#                 'values': [4, 8]
#             },
#             'epochs':{
#                 'values':[5, 10, 25]
#             }
#         }
#     }
#     sweep_defaults = {
#             'learning_rate': 2e-5,

#             'batch_size': 16,

#             'epochs':5
#     }

#     sweep_id = wandb.sweep(sweep_config)
#     wandb.agent(sweep_id,function=main)
