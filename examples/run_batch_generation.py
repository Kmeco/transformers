#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import json
import logging
import os
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedTokenizer
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer)
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


class LoadDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, split_token='<EOD>', block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line + split_token for line in f.read().split(split_token) if (len(line) > 0 and not line.isspace())]
        # add special tokens which shouldn't be split
        special_tokens_dict = {'cls_token': '<TLDR>', 'eos_token': '<EOD>'} #, 'additional_special_tokens': ['<EOT>']}
        tokenizer.add_special_tokens(special_tokens_dict)

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
        self.examples = [ex for ex in self.examples if tokenizer.encode('<TLDR>')[0] in ex]

        self.labels = []
        max_block = torch.arange(block_size)
        for ex in self.examples:
            # note that this will throw an exeption if token is not in the training example.
            try:
                idx = ex.index(tokenizer.encode('<TLDR>')[0])
            except ValueError as e:
                print("Example does not contain <TLDR> token.")
                print(tokenizer.decode(ex))
                exit()
            mask = (max_block <= idx)[:len(ex)]
            masked_labels = torch.tensor(ex) * ~mask - mask.type(torch.int) * 100  # ignore context when computing loss
            self.labels.append(masked_labels)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long), self.labels[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
             "The training dataset will be truncated in block of this size for training."
             "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)

    # load all json files in the data directory
    data_path = os.path.join(args.eval_data_file, "*.json")
    eval_dataset = []
    for f_name in glob(data_path):
        with open(f_name) as f:
            line = json.load(f)
            eval_dataset.append(line)

    special_tokens_dict = {'cls_token': '<TLDR>', 'eos_token': '<EOD>'}  # , 'additional_special_tokens': ['<EOT>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    #
    # # Note that DistributedSampler samples randomly
    #
    # def collate(examples):
    #     xs, ys = list(zip(*examples))
    #     return pad_sequence(xs, batch_first=True), pad_sequence(ys, batch_first=True)
    #
    # eval_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(
    #     eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    # )
    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    for example in tqdm(eval_dataset, desc="Evaluating"):
        inputs = tokenizer.encode(" ".join(example['article']) + ' <TLDR>', add_special_tokens=True)

        if len(inputs) > args.block_size:
            continue

        inputs = torch.tensor(inputs).to(args.device)

        output_sequence = model.generate(
            input_ids=inputs.unsqueeze(0),
            max_length=args.length + len(inputs),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        output_sequence.squeeze_()

        # Decode text
        text = tokenizer.decode(output_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]
        text = text[text.find(tokenizer.cls_token) :]

        total_dict = {'abstract': " ".join(example['abstract']), 'output': text}

        out_path = os.path.join(args.output_dir, "f_{}.json".format(global_step))
        with open(out_path, 'w') as f:
            json.dump(total_dict, f)

        global_step += 1


if __name__ == "__main__":
    main()
