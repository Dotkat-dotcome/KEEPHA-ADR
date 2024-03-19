This repository provides the baseline systems documented in the paper ``A Dataset for Pharmacovigilance in German, French, and Japanese: Annotating Adverse Drug Reactions across Languages"

# Getting started

```
pip install -r ./requirements.txt
```


# Example usage

FTo reproduce the results for the three Information Extraction tasks, you need a run file and a config file. 
You can find config files in the folder `./configs` for respective tasks.

## Name Entity Recognition (NER)

```
python src/finetune_ner.py configs/ner_configs/CONFIG_TRAIN.json
python src/inference_ner.py configs/eval_ner_configs/CONFIG_TEST.json
```

## Attribute Classification (AC)
```
python src/finetune_att.py configs/att_configs/CONFIG_TRAIN.json
python src/inference_att.py configs/eval_att_configs/CONFIG_TEST.json
```

## Relation Extraction (RE)
```
python src/finetune_re.py configs/re_configs/CONFIG_TRAIN.json
python src/inference_re.py configs/eval_re_configs/CONFIG_TEST.json
```


## Evaluation

```
java -cp brateval.jar au.com.nicta.csp.brateval.CompareEntities EVAL_FOLDER REF_FOLDER false
```


## Data

If you are interested in the data, please contact Lisa Raithel (raithel@tu-berlin.de)
The dataset is to be extracted in the directory `./data`


## Citation

Please cite our paper if you plan to use `KEEPHA-ADR`:

```
@inproceedings{raithel_dataset_2024,
  title = {A {{Dataset}} for {{Pharmacovigilance}} in {{German}}, {{French}}, and {{Japanese}}: {{Annotating Adverse Drug Reactions}} across {{Languages}}},
  booktitle = {Proceedings of the {{Language Resources}} and {{Evaluation Conference}}},
  author = {Raithel*, Lisa and Yeh*, Hui-Syuan and Yada, Shuntaro and Grouin, Cyril and Lavergne, Thomas and N{\'e}v{\'e}ol, Aur{\'e}lie and Paroubek, Patrick and Thomas, Philippe and M{\"o}ller, Sebastian and Nishiyama, Tomohiro and Aramaki, Eiji and Matsumoto, Yuji and Roller, Roland and Zweigenbaum, Pierre},
  year = {2024},
  month = may,
  publisher = {{ European Language Resources Association}},
  address = {{Torino, Italy}},
  copyright = {All rights reserved}
}
```
