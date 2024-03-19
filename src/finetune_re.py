from datasets import concatenate_datasets, Dataset
from datasets import load_dataset, load_from_disk, load_metric

from datetime import datetime
from tqdm.auto import tqdm

from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
import torch

from utils import utils
from utils.early_stopping import EarlyStopping

import argparse
import evaluate
import os,json
import numpy as np
import wandb

data_description = {
    "description": """This is a description.""",
    "file_name_blacklist": [],
}

MARKER_TOKENS = {
    'additional_special_tokens': [
        "[E1]",
        "[/E1]",
        "[E2]",
        "[/E2]",
        ]}
    

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

class RETraining(object):
    """docstring for NERTraining"""

    def __init__(self, config, mode="train"):
        super(RETraining, self).__init__()

        # load the "static" config params
        with open(config, "r") as read_handle:
            cf = json.load(read_handle)

        self.debug = cf["debug"]

        # load the dictionaries containing the languages and their IDs
        # load the dictionaries containing the languages and their IDs
        with open(cf["language2id_file"], "r") as rh_1, open(
            cf["language_dict_file"]
        ) as rh_2:
            self.lang2id = json.load(rh_1)
            self.id2lang = {idx: lang for lang, idx in self.lang2id.items()}
            self.lang_dict = json.load(rh_2)

        self.time = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

        if not self.debug:
            self.out_dir = utils.create_output_dir(config=cf, time=self.time)

        if cf["use_wandb"]:
            wandb.init(config=cf, project="keepha-lrec-re")
            self.cf = wandb.config
            self.debug = False
        else:
            wandb.init(mode="disabled")
            self.cf = cf

        self.data_urls = self.cf["data_url"]

        self.dataset = None
        self.model = None
        self.rel_list = None
        self.rel2id = None
        self.id2rel = None
        self.language_count = []

        self.metric = evaluate.load("f1")

    def prepare_tokenizer(self):
        """Load a pre-trained tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cf["model_name"],
            use_fast=True,
            add_prefix_space=True,
            strip_accent=False,
        )

        num_added_phi_toks = self.tokenizer.add_tokens(PHI_TOKENS)
        num_added_marker_toks = self.tokenizer.add_special_tokens(MARKER_TOKENS)

        print("We have added", num_added_phi_toks, "phi tokens")
        print("We have added special tokens", num_added_marker_toks, "marker tokens")

    def prepare_model(self):
        print(self.rel_list)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cf["model_name"],
            num_labels=len(self.rel_list),
            ignore_mismatched_sizes=True,
        )
        # Notice: resize_token_embeddings expect to receive the full size of
        # the new vocabulary, i.e., the length of the tokenizer.
        self.model.resize_token_embeddings(len(self.tokenizer))
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

    def prepare_scheduler_and_optimizer(self, wu_ratio, train_steps):
        """..."""
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.cf["learning_rate"],
            weight_decay=self.cf["weight_decay"],
        )

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=train_steps*wu_ratio,
            num_training_steps=train_steps,
        )

        return lr_scheduler, optimizer


    def train_model(self):
        """Run the training loop."""

        self.dataset.set_format("torch")

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
        )
        eval_dataloader = DataLoader(
            self.dataset["dev"],
            batch_size=self.cf["batch_size"],
        )

        es = EarlyStopping(patience=self.cf["patience"], mode="max")

        if self.debug:
            num_epochs = 30
            # eval_steps = 5
            # save_steps = 10

        else:
            num_epochs = self.cf["epochs"]
            eval_steps = 500
            save_steps = 500

        num_training_steps = num_epochs * len(train_dataloader)

        lr_scheduler, optimizer = self.prepare_scheduler_and_optimizer(
            wu_ratio=0.1, train_steps=num_training_steps
        )

        criterion = nn.CrossEntropyLoss()
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
                loss = criterion(outputs.logits, batch["labels"])
                # loss = outputs.loss
                # wandb.log({"train/loss": loss})
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

                eval_loss = criterion(outputs.logits, batch["labels"])

                wandb.log({"eval/loss": eval_loss})
                print("eval/epoch", epoch)
                print("eval/loss", str(eval_loss))

                logits = outputs.logits
                predictions = (
                    torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
                )
                true_labels = batch["labels"].detach().cpu().numpy().tolist()

                eval_predictions.extend(predictions)
                eval_true_labels.extend(true_labels)

            print(eval_predictions)

            current_macro_f1 =  self.metric.compute(predictions=eval_predictions, references=eval_true_labels, average="macro")["f1"]
            current_micro_f1 =  self.metric.compute(predictions=eval_predictions, references=eval_true_labels, average="micro")["f1"]
 
            print("eval/f1", current_macro_f1, current_micro_f1)
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

                        best_model.module.save_pretrained(self.out_dir)

            if es.step(current_macro_f1):
                print(f"Stopping training with F1 of {current_macro_f1}.")
                break

        return best_model
    
    

    def prepare_data(self):
        """..."""
        # path_to_preprocessed_data = f"{self.cf['path_to_preprocessed_data']}/concatenated_dataset_{self.cf['language']}"
        path_to_preprocessed_data = f"{self.cf['path_to_preprocessed_data']}/"
        # get tokenizer
        self.prepare_tokenizer()

        # If the data was already pre-processed once, we don't need to do it again,
        # just re-load it
        if os.path.exists(path_to_preprocessed_data):
            print(
                f"Reloading already pre-processed data from '{path_to_preprocessed_data}'."
            )
            tokenized_dataset = load_from_disk(path_to_preprocessed_data)
            # the label list is the same for each sample
            # self.label_list = dataset["train"][0]["label_names"]

            self.rel2id, self.id2rel, self.rel_list = utils.get_rel_id_dicts(
                config=self.cf
            )

        # Otherwise, create one dataset out of each data URL
        else:
            
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

            print("ENTER CONVERT SENTS")
            # convert annotations into sentences with entity pairs
            concatenated_datasets = concatenated_datasets.map(
                utils.convert_sents,
                batched=True,
                fn_kwargs={
                "lang2id": self.lang2id},
            )

            concatenated_datasets = concatenated_datasets.remove_columns(
            [
                col
                for col in concatenated_datasets["train"].features
                if col
                not in [
                    "language",
                    "text_id",
                    "sents",
                    "relations",
                    "relation_ids",
                ]
            ])

            
            for split in ['train', 'dev', 'test']:
                feature_dict = {'text_id':None, 'sents':None, 'relations':None, 'relation_ids':None, 'language':None}
                for i, doc_data in enumerate(concatenated_datasets[split]):
                    if feature_dict['text_id'] != None:
                        feature_dict['text_id'] += [doc_data['text_id']] * len(doc_data['relation_ids'])
                    else:
                        feature_dict['text_id'] = [doc_data['text_id']] * len(doc_data['relation_ids'])
                    
                    if feature_dict['language'] != None:
                        feature_dict['language'] += [doc_data['language']] * len(doc_data['relation_ids'])
                    else:
                        feature_dict['language'] = [doc_data['language']] * len(doc_data['relation_ids'])

                    
                    if feature_dict['sents'] != None:
                        feature_dict['sents'] += doc_data['sents']
                    else:
                        feature_dict['sents'] = doc_data['sents']
                    if feature_dict['relations'] != None:
                        feature_dict['relations'] += doc_data['relations']
                    else:
                        feature_dict['relations'] = doc_data['relations']
                    if feature_dict['relation_ids'] !=None:
                        feature_dict['relation_ids'] += doc_data['relation_ids']
                    else:
                        feature_dict['relation_ids'] = doc_data['relation_ids']
                dataset[split] = Dataset.from_dict(feature_dict)
            
            # the label list is the same for each sample
            # self.label_list = concatenated_datasets["train"][0]["label_names"]

            self.rel2id, self.id2rel, self.rel_list = utils.get_rel_id_dicts(
                config=self.cf
            )

            # tokenize the sequences and align the labels to the sub-tokens
            tokenized_dataset = dataset.map(
                utils.convert_rel,
                batched=True,
                fn_kwargs={
                    "tokenizer": self.tokenizer,
                    "rel2id": self.rel2id,
                    "max_length": self.cf["max_length"],
                },
                # # remove all unnecessary columns
                # remove_columns=dataset["train"].column_names,
            )
            # TODO: chunk data
            # save the data to disc to save pre-processing time
            if not self.debug:
                tokenized_dataset.save_to_disk(path_to_preprocessed_data)


        print(tokenized_dataset["train"][5])

        tokenized_dataset = tokenized_dataset.remove_columns(
            [
                col
                for col in tokenized_dataset["train"].features
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
        self.dataset = tokenized_dataset
        languages = self.dataset["train"]["language"]

        self.language_count = np.bincount([x for x in languages])
        print(self.language_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=None, help="Path to config file.")
    args = parser.parse_args()

    re_trainer =  RETraining(args.config)

    re_trainer.prepare_data()
    re_trainer.prepare_model()
    trained_model = re_trainer.train_model()


# if __name__ == "__main__":
#     main()

# ----------------------
#  hp search
# ----------------------
if __name__ == "__main__":
    sweep_config = {
        "method": "grid", 
        "metric": {
        "name": "best_macro_f1",
        "goal": "maximize"
        },
        "parameters": {
            
            "seed": {
                "values": [26747, 424881]
            },
        }
    }
    sweep_defaults = {
            "seed": 26747,
    }

    sweep_id = wandb.sweep(sweep_config, project="keepha-lrec-re")
    wandb.agent(sweep_id,function=main)
