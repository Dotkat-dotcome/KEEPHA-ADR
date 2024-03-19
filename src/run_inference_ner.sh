#!/usr/bin/env bash

python3 -m pip install --upgrade pip
pip install -r requirements_remote.txt


# all on all
python src/inference_ner.py configs/config_inference_remote_xlm_all_on_all_1.json
python src/inference_ner.py configs/config_inference_remote_xlm_all_on_all_2.json
python src/inference_ner.py configs/config_inference_remote_xlm_all_on_all_3.json
python src/inference_ner.py configs/config_inference_remote_xlm_all_on_all_4.json
python src/inference_ner.py configs/config_inference_remote_xlm_all_on_all_5.json


# de on all
python src/inference_ner.py configs/config_inference_remote_xlm_de_on_all_1.json
python src/inference_ner.py configs/config_inference_remote_xlm_de_on_all_2.json
python src/inference_ner.py configs/config_inference_remote_xlm_de_on_all_3.json
python src/inference_ner.py configs/config_inference_remote_xlm_de_on_all_4.json
python src/inference_ner.py configs/config_inference_remote_xlm_de_on_all_5.json


# de+ ja on all
python src/inference_ner.py configs/config_inference_remote_xlm_de+ja_on_all_1.json
python src/inference_ner.py configs/config_inference_remote_xlm_de+ja_on_all_2.json
python src/inference_ner.py configs/config_inference_remote_xlm_de+ja_on_all_3.json
python src/inference_ner.py configs/config_inference_remote_xlm_de+ja_on_all_4.json
python src/inference_ner.py configs/config_inference_remote_xlm_de+ja_on_all_5.json


# ja on all

python src/inference_ner.py configs/config_inference_remote_xlm_ja_on_all_1.json
python src/inference_ner.py configs/config_inference_remote_xlm_ja_on_all_2.json
python src/inference_ner.py configs/config_inference_remote_xlm_ja_on_all_3.json
python src/inference_ner.py configs/config_inference_remote_xlm_ja_on_all_4.json
python src/inference_ner.py configs/config_inference_remote_xlm_ja_on_all_5.json

# fr on all

# python src/inference_ner.py configs/config_inference_remote_xlm_fr_on_all_1.json
# python src/inference_ner.py configs/config_inference_remote_xlm_fr_on_all_2.json
# python src/inference_ner.py configs/config_inference_remote_xlm_fr_on_all_3.json
# python src/inference_ner.py configs/config_inference_remote_xlm_fr_on_all_4.json
# python src/inference_ner.py configs/config_inference_remote_xlm_fr_on_all_5.json
