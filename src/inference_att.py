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
from sklearn.metrics import classification_report

import argparse
import evaluate
import os,json
import numpy as np
import pandas as pd
import wandb

data_description = {
    "description": """This is a description.""",
    "file_name_blacklist": [],
}

MARKER_TOKENS = {
    'additional_special_tokens': [
        "[E]",
        "[/E]",
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

class ATTTest(object):
    """docstring for ATTTest"""

    def __init__(self, config, mode="eval"):
        super(ATTTest, self).__init__()

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
            # self.out_dir = utils.create_output_dir(config=cf, time=self.time)
            self.out_dir = cf["out_dir"]
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

        if cf["use_wandb"]:
            wandb.init(config=cf, project="keepha-lrec")
            self.cf = wandb.config
            self.debug = False
        else:
            wandb.init(mode="disabled")
            self.cf = cf

        self.data_urls = self.cf["data_url"]

        self.dataset = None
        self.model = None
        self.att_list = None
        self.att2id = None
        self.id2att = None
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
        print(self.att_list)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cf["model_name"],
            num_labels=len(self.att_list),
            ignore_mismatched_sizes=True,
        )
        # Notice: resize_token_embeddings expect to receive the full size of
        # the new vocabulary, i.e., the length of the tokenizer.
        self.model.resize_token_embeddings(len(self.tokenizer))
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)


    def inference(self):
        """Run the test loop."""

        self.dataset.set_format("torch")

        test_dataloader = DataLoader(
            self.dataset,
            batch_size=self.cf["batch_size"],
        )
        self.model.to(DEVICE)
        self.model.eval()
        # progress_bar = tqdm(range(num_training_steps))

        criterion = nn.CrossEntropyLoss()   

        eval_predictions = []
        # eval_languages = []
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

            test_loss = criterion(outputs.logits, batch["labels"])

            wandb.log({"test/loss": test_loss})
            print("test/loss", str(test_loss))

            logits = outputs.logits
            predictions = (
                torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            )
            true_labels = batch["labels"].detach().cpu().numpy().tolist()

            eval_predictions.extend(predictions)
            eval_true_labels.extend(true_labels)

            print(eval_predictions)

        macro_f1 =  self.metric.compute(predictions=eval_predictions, references=eval_true_labels, average="macro")["f1"]
        micro_f1 =  self.metric.compute(predictions=eval_predictions, references=eval_true_labels, average="micro")["f1"]
    
        print("test/f1", macro_f1, micro_f1)
        wandb.log({"macro_f1": macro_f1})
        wandb.log({"micro_f1": micro_f1})

        results = classification_report(y_true=eval_true_labels, 
                                        y_pred=eval_predictions, 
                                        labels=range(len(self.att_list)), 
                                        target_names=self.att_list, 
                                        zero_division=1.0,
                                        output_dict=True
                                        )
        df = pd.DataFrame(results).transpose()
        df.to_csv(os.path.join(self.out_dir, "report.csv"),  sep=',')

        return results
    
    

    def prepare_data(self):
        """..."""
        # path_to_preprocessed_data = f"{self.cf['path_to_preprocessed_data']}/concatenated_dataset_{self.cf['language']}"
        path_to_preprocessed_data = f"{self.cf['path_to_preprocessed_data']}/"
        # get tokenizer
        self.prepare_tokenizer()

        # If the data was already pre-processed once, we don't need to do it again,
        # just re-load it
        # if os.path.exists(path_to_preprocessed_data):
        #     print(
        #         f"Reloading already pre-processed data from '{path_to_preprocessed_data}'."
        #     )
        #     tokenized_dataset = load_from_disk(path_to_preprocessed_data)
        #     # the label list is the same for each sample
        #     # self.label_list = dataset["train"][0]["label_names"]

        #     self.att2id, self.id2att, self.att_list = utils.get_att_id_dicts(
        #         config=self.cf
        #     )

        # # Otherwise, create one dataset out of each data URL
        # else:
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
        
        concatenated_datasets = dataset
    
        assert (
            dataset["train"].features.type
            == concatenated_datasets["train"].features.type
        )

        print("ENTER CONVERT SENTS")
            # convert annotations into sentences with entity pairs
        concatenated_datasets = concatenated_datasets.map(
            utils.convert_att_sents,
            batched=True,
            fn_kwargs={
            "lang2id": self.lang2id},
        )


        for split in ['test']:
            feature_dict = {'text_id':None, 'sents':None, 'attributes':None, 'attribute_ids':None, 'language':None}
            for i, doc_data in enumerate(concatenated_datasets[split]):
                if feature_dict['text_id'] != None:
                    feature_dict['text_id'] += [doc_data['text_id']] * len(doc_data['attribute_ids'])
                else:
                    feature_dict['text_id'] = [doc_data['text_id']] * len(doc_data['attribute_ids'])
                if feature_dict['language'] != None:
                    feature_dict['language'] += [doc_data['language']] * len(doc_data['attribute_ids'])
                else:
                    feature_dict['language'] = [doc_data['language']] * len(doc_data['attribute_ids'])
                if feature_dict['sents'] != None:
                    feature_dict['sents'] += doc_data['sents']
                else:
                    feature_dict['sents'] = doc_data['sents']
                if feature_dict['attributes'] != None:
                    feature_dict['attributes'] += doc_data['attributes']
                else:
                    feature_dict['attributes'] = doc_data['attributes']
                if feature_dict['attribute_ids'] !=None:
                    feature_dict['attribute_ids'] += doc_data['attribute_ids']
                else:
                    feature_dict['attribute_ids'] = doc_data['attribute_ids']
            dataset[split] = Dataset.from_dict(feature_dict)


        # the label list is the same for each sample
        # self.label_list = concatenated_datasets["train"][0]["label_names"]

        self.att2id, self.id2att, self.att_list = utils.get_att_id_dicts(
            config=self.cf
        )
        
        # tokenize the sequences and align the labels to the sub-tokens
        tokenized_dataset = dataset["test"].map(
            utils.convert_att,
            batched=True,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "att2id": self.att2id,
                "max_length": self.cf["max_length"],
            },
            # # remove all unnecessary columns
            # remove_columns=dataset["train"].column_names,
        )
        # TODO: chunk data
        # save the data to disc to save pre-processing time
        if not self.debug:
            tokenized_dataset.save_to_disk(path_to_preprocessed_data)


        print(tokenized_dataset[5])

        tokenized_dataset = tokenized_dataset.remove_columns(
            [
                col
                for col in tokenized_dataset.features
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
        languages = self.dataset["language"]

        self.language_count = np.bincount([x for x in languages])
        print(self.language_count)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=None, help="Path to config file.")

    args = parser.parse_args()

    att_trainer =  ATTTest(args.config)

    att_trainer.prepare_data()
    att_trainer.prepare_model()
    results = att_trainer.inference()
    #todo:save results


if __name__ == "__main__":
    main()

