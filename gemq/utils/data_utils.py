import itertools
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, default_data_collator
from transformers.testing_utils import CaptureLogger
from datasets import load_dataset, Dataset, DatasetDict

from gemq.utils.model_utils import NAME_TO_MODEL, ModelType


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model, use_fast=False):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=use_fast)
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model, use_fast=False):
    traindata = load_dataset(
        "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
    )
    valdata = load_dataset(
        "allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation"
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=use_fast)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model="", use_fast=False):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, use_fast)
    if "c4" in name:
        return get_c4_new(nsamples, seed, seqlen, model, use_fast)


def build_calib_loader(dataset: str, tokenizer, max_block_size: int, n_blocks_for_stat: int, batch_size: int, num_workers: int, seed: int = 41):
    DATASETS = {
        "c4": lambda: load_dataset("json", data_files={"train": "data/c4-train.00000-of-01024.json"}),
        "math": lambda: load_dataset("json", data_files={"train": "data/math_pretrain_style.json"}),
    }
    
    all_set = DATASETS[dataset]()

    block_size = tokenizer.model_max_length
    if block_size > max_block_size:
        print(
            "The chosen tokenizer supports a `model_max_length` that is longer than the default `max_block_size` value"
            f" of {max_block_size}. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
            " override this default with `--max_block_size xxx`."
        )
        block_size = max_block_size

    if n_blocks_for_stat > 0:
        calib_set = all_set["train"].shuffle(seed=seed).select(
            range(min(n_blocks_for_stat * 16, len(all_set["train"]))))
    else:
        print("n_blocks_for_stat <= 0, using the whole dataset.")
        calib_set = all_set["train"].shuffle(seed=seed)

    print(f"Calibration dataset: {calib_set}")
    text_column_name = "text" if "text" in calib_set.features else list(
        calib_set.features)[0]

    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    tokenized_calib_set = calib_set.map(
        tokenize_function,
        batched=True,
        remove_columns=list(calib_set.features),
    )

    def group_texts(examples):
        concatenated_examples = {
            k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    lm_calib_set = tokenized_calib_set.map(
        group_texts,
        batched=True,
    )

    if n_blocks_for_stat > 0:
        assert len(lm_calib_set) > n_blocks_for_stat
        lm_calib_set = lm_calib_set.select(range(n_blocks_for_stat))

    calib_loader = DataLoader(
        lm_calib_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=default_data_collator
    )

    return calib_loader


def get_calib_loader(tokenizer, args):
    """
    This is a unified interface to get calibration dataloaders for different datasets
    in both model stats collection and quantization.
    """
    if args.calib_dataset == "wikitext2":
        calib_loader, _ = get_loaders(
            args.calib_dataset,
            args.nsamples,
            args.seed,
            seqlen=args.seqlen,
            model=args.model, # model dir
            use_fast=args.use_fast
        )

    elif args.calib_dataset in ["c4", "math"]:
        loader = build_calib_loader(
            args.calib_dataset,
            tokenizer,
            args.seqlen,
            args.nsamples,
            args.batch_size,
            num_workers=4,
            seed=args.seed
        )
        # unify the dataloader format
        calib_loader = []
        for i, batch in enumerate(loader):
            calib_loader.append((batch["input_ids"], None))  # (batch_size, seqlen)

    elif "+" in args.calib_dataset:
        datasets = args.calib_dataset.split("+")
        calib_loader = []
        for ds in datasets:
            assert ds in ["c4", "math"], f"Dataset {ds} not supported in combined calibration datasets."
            loader = build_calib_loader(
                ds,
                tokenizer,
                args.seqlen,
                args.nsamples,
                args.batch_size,
                num_workers=4,
                seed=args.seed
            )
            # unify the dataloader format
            for i, batch in enumerate(loader):
                calib_loader.append((batch["input_ids"], None))  # (batch_size, seqlen)

    else:
        raise NotImplementedError(f"Calibration dataset {args.calib_dataset} not implemented.")

    return calib_loader
