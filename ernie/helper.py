#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow import data, TensorShape, int64, int32
from math import exp
from os import makedirs
from shutil import rmtree, move, copytree
from huggingface_hub import hf_hub_download
import os


def get_features(tokenizer, sentences, labels):
    features = []
    for i, sentence in enumerate(sentences):
        inputs = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=tokenizer.model_max_length
        )
        input_ids, token_type_ids = \
            inputs['input_ids'], inputs['token_type_ids']
        padding_length = tokenizer.model_max_length - len(input_ids)

        if tokenizer.padding_side == 'right':
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            token_type_ids = token_type_ids + \
                [tokenizer.pad_token_type_id] * padding_length
        else:
            attention_mask = [0] * padding_length + [1] * len(input_ids)
            input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
            token_type_ids = \
                [tokenizer.pad_token_type_id] * padding_length + token_type_ids

        assert tokenizer.model_max_length \
            == len(attention_mask) \
            == len(input_ids) \
            == len(token_type_ids)

        feature = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': int(labels[i])
        }

        features.append(feature)

    def gen():
        for feature in features:
            yield (
                {
                    'input_ids': feature['input_ids'],
                    'attention_mask': feature['attention_mask'],
                    'token_type_ids': feature['token_type_ids'],
                },
                feature['label'],
            )

    dataset = data.Dataset.from_generator(
        gen,
        ({
            'input_ids': int32,
            'attention_mask': int32,
            'token_type_ids': int32
        }, int64),
        (
            {
                'input_ids': TensorShape([None]),
                'attention_mask': TensorShape([None]),
                'token_type_ids': TensorShape([None]),
            },
            TensorShape([]),
        ),
    )

    return dataset


def softmax(values):
    exps = [exp(value) for value in values]
    exps_sum = sum(exp_value for exp_value in exps)
    return tuple(map(lambda x: x / exps_sum, exps))


def make_dir(path):
    try:
        makedirs(path)
    except FileExistsError:
        pass


def remove_dir(path):
    rmtree(path)


def copy_dir(source_path, target_path):
    copytree(source_path, target_path)


def move_dir(source_path, target_path):
    move(source_path, target_path)

def download_from_hub(repo_id, filename, revision=None, cache_dir=None):
    try:
        hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, cache_dir=cache_dir)
    except Exception as exp:
        raise exp

            
    if cache_dir is not None:

        files = os.listdir(cache_dir)

        for f in files:
            if '.lock' in f:
                name = f[0:-5]

                os.rename(cache_dir+name, cache_dir+filename)
                os.remove(cache_dir+name+'.lock')
                os.remove(cache_dir+name+'.json')
