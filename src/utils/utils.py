"""..."""

# import spacy

# from spacy.training import offsets_to_biluo_tags
from utils.format_data import *

# from utils.iob_utils import offsets_to_biluo_tags
import json
import numpy as np
import os
import torch

from tokenizers import Encoding
from torch.utils.data import WeightedRandomSampler
from transformers import BatchEncoding

# nlp = spacy.load("en_core_web_trf")

# class Term:
#     def __init__(self,id,name,spans,text):
#         self.id = id
#         self.name = name
#         self.spans = sorted(spans, key=lambda x:x[0])
#         self.text = text


def get_balancing_sampler(dataset, language_count, device):
    """Create a data sampler based on the class weights."""
    languages = dataset["language"].cpu()
    language_count = torch.from_numpy(language_count)
    language_count.to(device)
    label_weights = 1 / language_count
    print(f"language weights: {label_weights}, {type(label_weights)}")
    sample_weight = torch.hstack([label_weights[label] for label in languages])

    return WeightedRandomSampler(
        weights=sample_weight.double(),
        num_samples=len(sample_weight),
        replacement=True,
    )


def get_label_id_dicts(config):
    """Load the label2id dict."""

    # load the fixed label-to-id dictionary
    with open(config["label2id_file"], "r") as rh:
        label2id = json.load(rh)
        label_list = [key for key in label2id]
        # commented: unseen entity is allowed
        # # make sure the loaded dicts and the data labels match
        # assert set(label_list) == set(
        #     label2id.keys()
        # ), f"{set(label_list)} vs. {set(label2id.keys())}"

    id2label = {i: l for l, i in label2id.items()}

    return label2id, id2label, label_list


def sort_by_language(predictions, labels, languages):
    """Make one set of predictions-labels for each language."""
    by_language = {lang: {"predictions": [], "true_labels": []} for lang in languages}

    for pred, label, language in zip(predictions, labels, languages):
        by_language[language]["predictions"].append(pred)
        by_language[language]["true_labels"].append(label)

    return by_language


def add_language_feature(examples, language_list, lang2id):
    """For every file, add the language it is written in."""
    # adds new_info[index of x] as a field to x
    lang_id_list = [lang2id[lang] for lang in language_list]

    examples.update({"language": lang_id_list})

    return examples


def prepare_data_urls(config, lang_dict):
    """Collect all necessary paths and create data URLs."""
    # get base data URL and update it according to the specs in the config
    base_url = config["data_url"]
    language = config["language"]
    lang_per_url = []

    if language != "all":
        # data_dir = os.path.join(base_url, lang_dict[language])
        data_urls = [base_url]
        # data_urls = [
        #     os.path.join(data_dir, d)
        #     for d in os.listdir(data_dir)
        #     if os.path.isdir(os.path.join(data_dir, d))
        # ]

    else:
        # first get the "base" URL for each language
        base_urls = [
            os.path.join(base_url, d)
            for d in os.listdir(base_url)
            if os.path.isdir(os.path.join(base_url, d))
        ]
        # then go down each path to get the final dir containing the files
        data_urls = []
        for url in base_urls:
            data_urls.extend(
                [
                    os.path.join(url, d)
                    for d in os.listdir(url)
                    if os.path.isdir(os.path.join(url, d))
                ]
            )

    lang_per_url = [lang_dict[url.split("/")[-3]] for url in data_urls]
    return dict(zip(data_urls, lang_per_url))


def create_output_dir(config, time):
    """Create the output directory."""
    out_dir = os.path.join(
        config["out_dir"],
        f"checkpoint_{config['model_name'].replace('/', '-')}_{time}",
    )
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def overlap(x: tuple, y: tuple):
    return bool(len(range(max(x[0], y[0]), min(x[1], y[1]) + 1)))


def remove_span_overlaps(entities):
    """Remove spans that overlap and only take the longest span.

    https://stackoverflow.com/questions/74861529/remove-overlaping-tuple-
    ranges-from-list-leaving-only-the-longest-range
    """
    sorted_entities = sorted(entities, key=lambda x: x[0])
    # define a difference function
    diff = lambda x: abs(x[0] - x[1])

    filtered = [sorted_entities[0]]

    for x in sorted_entities[1:]:
        if overlap(filtered[-1], x):
            if diff(filtered[-1]) == diff(x):
                filtered.append(x)
            else:
                filtered[-1] = max(filtered[-1], x, key=diff)
        else:
            filtered.append(x)
    return filtered


# ----------------------------------------------
# helpers for NER


def offsets_to_biluo_tokenize(text, annotations, tokenizer):
    tokenized_batch: BatchEncoding = tokenizer(text)
    tokenized_text: Encoding = tokenized_batch[0]
    # tokenized_text = ""
    tokens = tokenized_text.tokens
    # import ipdb; ipdb.set_trace()
    aligned_labels = ["O"] * len(
        tokens
    )  # Make a list to store our labels the same length as our tokens
    for anno in annotations:
        annotation_token_ix_set = (
            set()
        )  # A set that stores the token indices of the annotation
        # anno["start"] = anno[0]
        # anno["end"] =  anno[1]
        # anno["labels"] = anno[2]
        for char_ix in range(anno[0], anno[1]):
            token_ix = tokenized_text.char_to_token(char_ix)
            if token_ix is not None:
                annotation_token_ix_set.add(token_ix)

        if len(annotation_token_ix_set) == 1:
            # If there is only one token
            token_ix = annotation_token_ix_set.pop()
            prefix = (
                "U"  # This annotation spans one token so is prefixed with U for unique
            )
            aligned_labels[token_ix] = f"{prefix}-{anno[2]}"

        else:
            last_token_in_anno_ix = len(annotation_token_ix_set) - 1
            for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                if num == 0:
                    prefix = "B"
                elif num == last_token_in_anno_ix:
                    prefix = "L"  # Its the last token
                else:
                    prefix = "I"  # We're inside of a multi token annotation
                aligned_labels[token_ix] = f"{prefix}-{anno[2]}"
    return tokens, aligned_labels


def convert_offsets_to_iob(examples, tokenizer, lang2id):
    """Convert the brat offsets to IOB tags."""

    """
    FR: lifeline_v1_FR
    DE: lifeline_v2
    JA: JP, qu_, twjp
    """

    did = examples["file_name"]
    languages = []
    for file_name in did:
        if "lifeline_v1_FR" in file_name:
            languages.append(lang2id["fr"])
        elif "lifeline_v2" in file_name:
            languages.append(lang2id["de"])
        elif file_name.startswith(("JP", "qa_", "twjp")):
            languages.append(lang2id["ja"])
        else:
            print("no language matches")
            assert False

    documents = examples["context"]
    annotations = examples["spans"]
    overlapping_spans_counter = 0

    all_tokens = []
    all_tags = []
    unique_tags = set()
    long_tokens = []

    for i, doc in enumerate(documents):
        # processed = nlp(doc)
        processed = doc
        anno_spans = annotations[i]["locations"]
        anno_types = annotations[i]["type"]

        entities = []
        tags = []
        for span, type in zip(anno_spans, anno_types):
            entities.append((span["start"][0], span["end"][0], type))
        # all_ents += entities

        # tokens = [tok.text for tok in processed]

        if entities:
            # tags = offsets_to_biluo_tags(processed, entities)
            tokens, tags = offsets_to_biluo_tokenize(processed, entities, tokenizer)

        # else:
        #     pass
        # tags = ["O"] * len(tokens)

        assert len(tags) == len(
            tokens
        ), f"#tags does not match #tokens\n\ntokens: {tokens}\n\ntags: {tags}"

        if len(tags) > 510:
            long_tokens.append(i)
        all_tokens.append(tokens)

        # in case some tokens could not be aligned, replace them with the
        # "O" tags (mostly happens with tags that span only a part of one token)
        # tags = [tag.replace("-", "O") if tag == "-" else tag for tag in tags] # (to be redone)
        # for tok, t in zip(tokens, tags):
        #     print(f"'{tok}' --> '{t}'")
        all_tags.append(tags)
        unique_tags.update(tags)

    # to fit the shape requirements, add the unique tags for every sample
    list_of_unique_tags = [list(unique_tags)] * len(documents)

    print(
        f"Removed {overlapping_spans_counter} overlapping spans.\n#unique tags: {len(unique_tags)}"
    )
    return {
        "tags": all_tags,
        "tokens": all_tokens,
        "label_names": list_of_unique_tags,
        "text_id": did,
        "text": examples["context"],
        "language": languages,
    }


def covert_labels(examples, label2id, tokenizer, label_all_tokens=True):
    """NER: Convert the IOB tags to ids."""
    MaxLength = 512

    tokenized_inputs = tokenizer(
        examples["context"],
        # is_split_into_words=True,
        truncation=True,
        max_length=MaxLength,
        padding="max_length",
        # return_token_type_ids=False
    )
    # add_prefix_space=True

    labels = []

    for i, label_list in enumerate(examples["tags"]):
        j = 0
        label_ids = []
        # label_ids = [0,0]

        while j < MaxLength:
            # for label in label_list:
            # label_ids.append(label2id[label])
            if j < len(label_list):
                label_ids.append(label2id[label_list[j]])
                j += 1

            else:
                label_ids.append(label2id["O"])
                j += 1

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def convert_biolu_to_brat(text, biolu_tags, tokenizer):
    brat_annotations = []
    current_entity = None
    doc = tokenizer(text)

    print(doc)

    for i, tag in enumerate(biolu_tags):
        try:
            if tag == "O":
                if current_entity:
                    brat_annotations.append(current_entity)
                    current_entity = None

            elif tag.startswith("B-"):
                if current_entity:
                    brat_annotations.append(current_entity)
                entity_type = tag[2:]
                print(f"doc.token_to_chars(i): {doc.token_to_chars(i)}")
                start_index = doc.token_to_chars(i)[0]
                end_index = doc.token_to_chars(i)[1]
                # current_entity = f"T{len(brat_annotations) + 1}\t{entity_type} {start_index} {end_index + 1}\t{text[start_index:end_index + 1]}"
                current_entity = Term(
                    "T" + str(len(brat_annotations) + 1),
                    entity_type,
                    [[start_index, end_index]],
                    text[start_index:end_index],
                )

            elif tag.startswith("I-"):
                if current_entity:
                    end_index = doc.token_to_chars(i)[1]
                    # current_entity = f"{current_entity[:current_entity.rfind(' ')]} {end_index}"
                    # 'T1\tDISORDER 203 207\tvom 219'
                    # current_entity.split('\t')[1].split(' ')[-1] = end_index
                    # current_entity.split('\t')[1].split(' ')[-1]
                    current_entity.spans[0][1] = end_index
                    current_entity.text = text[current_entity.spans[0][0] : end_index]

            elif tag.startswith("L-"):
                if current_entity:
                    end_index = doc.token_to_chars(i)[1]
                    # current_entity = f"{current_entity[:current_entity.rfind(' ')]} {end_index + 1}"
                    # current_entity.split('\t')[1].split(' ')[-1] = end_index
                    current_entity.spans[0][1] = end_index
                    current_entity.text = text[current_entity.spans[0][0] : end_index]
                    brat_annotations.append(current_entity)
                    current_entity = None

            elif tag.startswith("U-"):
                if current_entity:
                    brat_annotations.append(current_entity)
                entity_type = tag[2:]
                start_index = doc.token_to_chars(i)[0]
                end_index = doc.token_to_chars(i)[1]
                # current_entity = f"T{len(brat_annotations) + 1}\t{entity_type} {start_index} {end_index + 1}\t{text[start_index:end_index + 1]}"
                current_entity = Term(
                    "T" + str(len(brat_annotations) + 1),
                    entity_type,
                    [[start_index, end_index]],
                    text[start_index:end_index],
                )
        except:
            print("Prediction is outside of the text overall spans. Ignored.")
            break

    # Handle the last entity if any
    if current_entity:
        brat_annotations.append(current_entity)

    return brat_annotations


def covert_predictions(
    tokenizer,
    id2label,
    text_id,
    text_sample,
    eval_predictions,
    eval_true_labels,
    # gold_dir,
    output,
):
    for i, (prediction, gold, text, did) in enumerate(
        zip(eval_predictions, eval_true_labels, text_sample, text_id)
    ):
        print(f"text: {text}\n")
        tags = [id2label[p] for p in prediction]
        print(f"predicted: {tags}")
        gold_converted = [id2label[g] for g in gold]
        print(f"\ngold: {gold_converted}\n")

        # convert iob to brat
        brat_annotations = convert_biolu_to_brat(text, tags, tokenizer)
        print(brat_annotations)


        # # compare with the TRUE gold
        # gold_obj = AnnotationFile(os.path.join(gold_dir, did+".ann"))
        # badTerms = gold_obj.getBadTerms()

        # if len(badTerms)!=0:
        #     spans = []
        #     for t in badTerms:
        #         spans += [[t.spans[0][0], t.spans[-1][1]]]
        #     spans = getFullSentences(text,spans)
        #     spans = mergeSpans(spans)   
        #     #they are sorted after merging

        #     offset=0
        #     j=0
        #     for i in range(len(spans)):
        #         while j<len(brat_annotations) and brat_annotations[j].spans[0][0] < spans[i][0]:
        #             brat_annotations[j].spans[0][0] += offset 
        #             brat_annotations[j].spans[0][1] += offset 
        #             j+=1
        #         offset+=spans[i][1] - spans[i][0]

        #     # the last removed sent-spans
        #     while j<len(brat_annotations):
        #             brat_annotations[j].spans[0][0] += offset 
        #             brat_annotations[j].spans[0][1] += offset 
        #             j+=1


        a = AnnotationFile(None, brat_annotations, [], [])
        a.write(os.path.join(output, did + ".ann"))


# ----------------------------------------------
# helpers for RE


def convert_sents(examples, lang2id):
    """Convert the brat-document relations to sentence samples."""
    """
    FR: lifeline_v1_FR
    DE: lifeline_v2
    JA: JP, qu_, twjp
    """

    did = examples["file_name"]
    print(did)
    languages = []
    for file_name in did:
        print(file_name)
        if "lifeline_v1_FR" in file_name:
            languages.append(lang2id["fr"])
        elif "lifeline_v2" in file_name:
            languages.append(lang2id["de"])
        elif file_name.startswith(("JP", "qa_", "twjp")):
            languages.append(lang2id["ja"])
        else:
            print("no language matches")
            assert False
    print(languages)

    did = examples["file_name"]
    documents = examples["context"]
    annotations = examples["relations"]

    all_anno_types = []
    all_rids = []
    all_marked_examples = []
    all_examples = []

    for i, doc in enumerate(documents):
        anno_types = []
        rids = []
        marked_examples = []
        dict_examples = []

        ents_ids = examples["spans"][i]["id"]
        ents_spans = examples["spans"][i]["locations"]
        pairs = annotations[i]["arguments"]

        for j in range(0, len(pairs)):
            # print(j)
            anno_ents = pairs[j]["target"]
            anno_order = pairs[j]["type"]
            anno_type = annotations[i]["type"][j]
            #     rid = annotations[i]["id"][j]

            #     anno_types.append(anno_type)
            #     rids.append(rid)

            # find the spans for the two ents
            try:
                span1 = ents_spans[ents_ids.index(anno_ents[0])]  # list of sub-spans list
                span2 = ents_spans[ents_ids.index(anno_ents[1])]  # list of sub-spans list
                spans = span1["start"] + span1["end"] + span2["start"] + span2["end"]
            except:
                pass # some none-yet-close badTerms were removed after the none-link
        
            left_most = min(spans)
            right_most = max(spans)
            sents_spans = getFullSentences(doc, [[left_most, right_most]])
            text_example = doc[sents_spans[0][0] : sents_spans[0][1]]

            ex_dict = {}
            ex_dict["h"] = {}
            ex_dict["t"] = {}
            ex_dict["h"]["name"] = ""
            ex_dict["t"]["name"] = ""
            ex_dict["relation"] = anno_type
            
            # mark the entities
            markers = []
            if anno_order == ["Arg1", "Arg2"]:
                for i2, start in enumerate(span1["start"]):
                    markers.append([start, "[E1]"])
                    ex_dict["h"]["name"] += doc[span1["start"][i2]: span1["end"][i2]]
                for end in span1["end"]:
                    markers.append([end, "[/E1]"])

                for i2, start in enumerate(span2["start"]):
                    markers.append([start, "[E2]"])
                    ex_dict["t"]["name"] += doc[span2["start"][i2]: span2["end"][i2]]
                for end in span2["end"]:
                    markers.append([end, "[/E2]"])
                
                try:
                    ex_dict["h"]["type"] = examples["spans"][i]["type"][ents_ids.index(anno_ents[0])]
                    ex_dict["t"]["type"] = examples["spans"][i]["type"][ents_ids.index(anno_ents[1])]
                except:
                    pass

            else:
                for i2, start in enumerate(span2["start"]):
                    markers.append([start, "[E1]"])
                    ex_dict["h"]["name"] += doc[span2["start"][i2]: span2["end"][i2]]
                for end in span2["end"]:
                    markers.append([end, "[/E1]"])

                for i2, start in enumerate(span1["start"]):
                    markers.append([start, "[E2]"])
                    ex_dict["t"]["name"] += doc[span1["start"][i2]: span1["end"][i2]]
                for end in span1["end"]:
                    markers.append([end, "[/E2]"])

                try:
                    ex_dict["h"]["type"] = examples["spans"][i]["type"][ents_ids.index(anno_ents[0])]
                    ex_dict["t"]["type"] = examples["spans"][i]["type"][ents_ids.index(anno_ents[1])]
                except:
                    pass
            
            
            markers.sort(key=lambda k: (k[0]), reverse=True)
            marked_example = text_example
            for marker in markers:
                marked_example = (
                    marked_example[: marker[0] - sents_spans[0][0]]
                    + marker[1]
                    + marked_example[marker[0] - sents_spans[0][0] :]
                )

            marked_examples.append(marked_example)
            
            word_count = len(marked_example.split())
            # if word_count >=300:
            #     import ipdb; ipdb.set_trace()

            ex_dict["tokens"] = marked_example.split()
            ex_dict = {'tokens': ex_dict.pop('tokens'), **ex_dict}
            dict_examples.append(ex_dict)
            # import ipdb; ipdb.set_trace()
        # all_anno_types.append(anno_types)
        # all_rids.append(rids)
        # all_marked_examples.append(marked_examples)
        all_examples.append(dict_examples)

    return {
        "text_id": did,
        # "sents": all_marked_examples,
        # "relations": all_anno_types,
        # "relation_ids": all_rids,
        # "language": languages,
        "dicts": all_examples,
    }


def convert_rel(examples, rel2id, tokenizer, max_length):
    """Prepare the tokenized dataset."""

    print(f"""NUM. EXAMPLES: { len(examples["sents"]) }""")

    word_count = [len(sent.split()) for sent in examples["sents"]]
    print(f"AVG. WORD COUNT:{sum(word_count)/len(word_count)}")
    print(f"MAX. WORD COUNT:{max(word_count)}")

    token_count = [len(tokenizer.tokenize(sent)) for sent in examples["sents"]]
    print(f"AVG. TOKEN COUNT:{sum(token_count)/len(token_count)}")
    print(f"MAX. TOKEN COUNT:{max(token_count)}")

    num_remove_toolong = 0
    for i, sent in enumerate(examples["sents"]):
        num_token = len(tokenizer.tokenize(sent))
        if num_token >= 512:
            examples["sents"][i] = ""
            num_remove_toolong += 1
    print(f"We have emptied text: {num_remove_toolong}")
    print(f"""NUM. EXAMPLES: { len(examples["sents"]) }""")

    tokenized_inputs = tokenizer(
        examples["sents"],
        # is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    examples["relations"] = [
        "SIGNALS_CHANGE_OF" if rel == "SIGNALS_OF_CHANGE" else rel
        for rel in examples["relations"]
    ]
    tokenized_inputs["labels"] = [rel2id[i] for i in examples["relations"]]

    return tokenized_inputs


def get_rel_id_dicts(config):
    """Load the rel2id dict."""

    # load the fixed rel-to-id dictionary
    with open(config["rel2id_file"], "r") as rh:
        rel2id = json.load(rh)
        rel_list = [key for key in rel2id]
    id2rel = {i: l for l, i in rel2id.items()}

    return rel2id, id2rel, rel_list


# ----------------------------------------------
# helpers for ATT


def convert_att_sents(examples, lang2id):
    """Convert the brat-document attributions to sentence samples."""

    did = examples["file_name"]
    print(did)
    languages = []
    for file_name in did:
        print(file_name)
        if "lifeline_v1_FR" in file_name:
            languages.append(lang2id["fr"])
        elif "lifeline_v2" in file_name:
            languages.append(lang2id["de"])
        elif file_name.startswith(("JP", "qa_", "twjp")):
            languages.append(lang2id["ja"])
        else:
            print("no language matches")
            assert False
    print(languages)

    documents = examples["context"]
    annotations = examples["attributions"]

    all_anno_types = []
    all_aids = []
    all_marked_examples = []

    for i, doc in enumerate(documents):
        # anno_types = annotations[i]["value"]
        # aids = annotations[i]["id"]
        anno_types = []
        aids = []
        marked_examples = []

        ents_ids = examples["spans"][i]["id"]
        ents_spans = examples["spans"][i]["locations"]

        for j in range(len(annotations[i]["target"])):
            ent = annotations[i]["target"][j]

            # pass unknown attributes
            if (
                annotations[i]["value"][j] == "None"
                or annotations[i]["value"][j] == "time"
            ):
                continue
            anno_types.append(annotations[i]["value"][j])
            aids.append(annotations[i]["id"][j])

            # find the spans for the two ents
            spans = ents_spans[ents_ids.index(ent)]
            all_spans = spans["start"] + spans["end"]
            sents_spans = getFullSentences(doc, [[min(all_spans), max(all_spans)]])
            text_example = doc[sents_spans[0][0] : sents_spans[0][1]]

            # mark the entitie
            markers = []
            for start in spans["start"]:
                markers.append([start, "[E]"])
            for end in spans["end"]:
                markers.append([end, "[/E]"])

            markers.sort(key=lambda k: (k[0]), reverse=True)
            marked_example = text_example
            for marker in markers:
                marked_example = (
                    marked_example[: marker[0] - sents_spans[0][0]]
                    + marker[1]
                    + marked_example[marker[0] - sents_spans[0][0] :]
                )

            marked_examples.append(marked_example)

        all_anno_types.append(anno_types)
        all_aids.append(aids)
        all_marked_examples.append(marked_examples)

    return {
        "text_id": did,
        "sents": all_marked_examples,
        "attributes": all_anno_types,
        "attribute_ids": all_aids,
        "language": languages,
    }


def convert_att(examples, att2id, tokenizer, max_length):
    """Prepare the tokenized dataset."""

    print(f"""NUM. EXAMPLES: { len(examples["sents"]) }""")

    word_count = [len(sent.split()) for sent in examples["sents"]]
    print(f"AVG. WORD COUNT:{sum(word_count)/len(word_count)}")
    print(f"MAX. WORD COUNT:{max(word_count)}")

    token_count = [len(tokenizer.tokenize(sent)) for sent in examples["sents"]]
    print(f"AVG. TOKEN COUNT:{sum(token_count)/len(token_count)}")
    print(f"MAX. TOKEN COUNT:{max(token_count)}")

    num_remove_toolong = 0
    for i, sent in enumerate(examples["sents"]):
        num_token = len(tokenizer.tokenize(sent))
        if num_token >= 512:
            examples["sents"][i] = ""
            num_remove_toolong += 1
    print(f"We have emptied text: {num_remove_toolong}")
    print(f"""NUM. EXAMPLES: { len(examples["sents"]) }""")

    tokenized_inputs = tokenizer(
        examples["sents"],
        # is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    examples["attributes"] = [
        "point" if rel == "relative_point_in_time" else rel
        for rel in examples["attributes"]
    ]
    examples["attributes"] = [
        "increase" if rel == "increased" else rel for rel in examples["attributes"]
    ]
    examples["attributes"] = [
        "decrease" if rel == "decreased" else rel for rel in examples["attributes"]
    ]
    examples["attributes"] = [
        "negated" if rel == "true" else rel for rel in examples["attributes"]
    ]

    tokenized_inputs["labels"] = [att2id[i] for i in examples["attributes"]]

    return tokenized_inputs


def get_att_id_dicts(config):
    """Load the att2id dict."""

    # load the fixed att-to-id dictionary
    with open(config["att2id_file"], "r") as rh:
        att2id = json.load(rh)
        att_list = [key for key in att2id]
    id2att = {i: l for l, i in att2id.items()}

    return att2id, id2att, att_list


# def tokenize_and_align_labels(examples, tokenizer, label2id, label_all_tokens=True):
#     """Tokenize and align the data.

#     Sub-tokenize the sequences and align the labels with the
#     sub-tokens / sub-token IDs.
#     """

#     # sub-tokenize input sentences and convert to IDs
#     # tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
#     tokenized_inputs = tokenizer(
#         examples["tokens"],
#         is_split_into_words=True,
#         truncation=True,
#         max_length=512,
#         padding="max_length",
#         # add_prefix_space=True
#     )

#     # we do not need this; only for validating
#     sub_tokens_all = []
#     for idx_list in tokenized_inputs["input_ids"]:
#         sub_tokens = tokenizer.convert_ids_to_tokens(idx_list)
#         sub_tokens_all.append(sub_tokens)
#     labels = examples["tags"]

#     labels = []
#     word_idss = []
#     for i, label_list in enumerate(examples["tags"]):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)

#         previous_word_idx = None
#         label_ids = []
#         for word_idx in word_ids:
#             # Special tokens have a word id that is None. We set the label to -100 so they are automatically
#             # ignored in the loss function.
#             # import ipdb; ipdb.set_trace()
#             if word_idx is None:
#                 label_ids.append(-100)

#             # We set the label for the first token of each word.
#             elif word_idx != previous_word_idx:
#                 current_label = label_list[word_idx]
#                 label_ids.append(label2id[current_label])
#             # For the other tokens in a word, we set the label to either the current label or -100, depending on
#             # the label_all_tokens flag.
#             else:
#                 # import ipdb; ipdb.set_trace()
#                 label_ids.append(
#                     label2id[label_list[word_idx]] if label_all_tokens else -100
#                 )
#             previous_word_idx = word_idx

#         labels.append(label_ids)
#         word_idss.append(word_ids)

#     # tokenized_inputs["word_labels"] = examples["tags"]
#     # tokenized_inputs["word_ids"] = word_idss
#     tokenized_inputs["labels"] = labels
#     tokenized_inputs["sub_tokens"] = sub_tokens_all

#     return tokenized_inputs


# def convert_word_labels(sub_predictions, sub_words):
#     predictions = []

#     for i, word_ids in enumerate(sub_words):
#         prediction = []
#         for j, word_idx in enumerate(word_ids):
#             if word_idx is None:
#                 pass
#             # We set the label for the first token of each word.
#             elif word_idx != previous_word_idx:
#                 prediction.append(sub_predictions[i][j])
#             else:
#                 pass

#             previous_word_idx = word_idx

#         predictions.append(prediction)


#     return predictions
