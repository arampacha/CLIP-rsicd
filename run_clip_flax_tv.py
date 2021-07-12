#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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
"""
Pre-training/Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import json
import jsonlines
import shutil
import numpy as np

import datasets
from datasets import Dataset, load_dataset
from flax import training
from tqdm import tqdm

import torch
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import (
    # added for image augmentation
    ToPILImage,
    RandomCrop,
    ColorJitter,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomResizedCrop,
    ToTensor,
    # /added for image augmentation
    CenterCrop, 
    ConvertImageDtype, 
    Normalize, 
    Resize
)
from torchvision.transforms.functional import InterpolationMode

import jax
import jax.profiler
import jax.numpy as jnp
import optax
import transformers
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from flax.serialization import to_bytes, from_bytes
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    FlaxCLIPModel,
    CLIPProcessor,
    CLIPTokenizerFast,
    HfArgumentParser,
    TrainingArguments,
    is_tensorboard_available,
    IntervalStrategy
    
)
from transformers.testing_utils import CaptureLogger

from importlib.util import find_spec

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )
    save_optimizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to store full train state including optimizer."},
    )
    repo_path_or_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the modelhub repo directory"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to local folder containing data files."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    text_column_name: Optional[str] = field(
            default='text',
            metadata={"help": "Column containing main text data."},
    )
    augment_images: Optional[bool] = field(
        default=False,
        metadata={ "help": "Augment input training images" }
    )
    captions_per_image: Optional[int] = field(
        default=5,
        metadata={"help": "Number of captions per image to use when creating train dataset."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt", "jsonl"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt", "jsonl"], "`validation_file` should be a csv, a json or a txt file."


# We use torchvision for faster image pre-processing.
# We need to ensure faster processing speed as it can become a bottleneck on TPU
class Transform(torch.nn.Module):
    def __init__(self, image_size, augment_images):
        super().__init__()
        if augment_images:
            crop_size = int(image_size * 0.8)
            self.transforms = torch.nn.Sequential(
                # image augmentation transforms
                RandomCrop(crop_size),
                ColorJitter(),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomResizedCrop(crop_size, scale=(0.8, 1.2), ratio=(1.0, 1.0)),
                # /image augmentation transforms
                Resize([image_size], interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
                ConvertImageDtype(torch.float),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            )
        else:
            self.transforms = torch.nn.Sequential(
                Resize([image_size], interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
                ConvertImageDtype(torch.float),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x

class ImageTextDataset(VisionDataset):
    """
    Dtaset for loading image-text data for tasks like CLIP training, Image Captioning.

    Args:
        root: (string): The root path where the dataset is stored
        file_path: (string): Path to the file containing the image_paths and associated captions.
            The expected format is jsonlines where each line is a json object containing to keys.
            `filename`: The path to the image.
            `captions`: An `array` of captions.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        file_path: str,
        captions_per_image=5,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        with jsonlines.open(file_path, "r") as reader:

            self.captions = []
            self.image_paths = []

            for example in reader:
                self.captions.extend(example["captions"][:captions_per_image])
                self.image_paths.extend([example["filename"]] * captions_per_image)

    def _load_image(self, idx: int):
        path = f"{self.root}/{self.image_paths[idx]}"
        return read_image(path, mode=ImageReadMode.RGB)

    def _load_target(self, idx):
        return self.captions[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.captions)


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn

# utils
def mb_item(x):
    return x.item() if hasattr(x, "item") else x

def make_batch(samples):
    batch = {k:jnp.array(v) for k,v in samples.items()}
    return batch

#checkpoint functions
# def save_checkpoint(model, save_dir, state, with_opt:bool=True, push_to_hub:bool=False):
#     state = jax_utils.unreplicate(state)
#     logger.info(f"SAVING CHECKPOINT IN {save_dir}...")
#     save_dir = f"{save_dir}/ckpt-{mb_item(state.step)-1}"
#     model.save_pretrained(
#         save_dir,
#         params=state.params,
#         push_to_hub=push_to_hub,
#         commit_message=f"Saving weights and logs at step {mb_item(state.step)-1}",
#     )
#     if with_opt:
#         with open(os.path.join(save_dir, "opt_state.msgpack"), "wb") as f:
#             f.write(to_bytes(state.opt_state))
#         with open(os.path.join(save_dir, "training_state.json"), "w") as f:
#             json.dump({"step": state.step.item()}, f)
#     logger.info("checkpoint saved")
        
# def restore_checkpoint(save_dir, state):
#     logger.info(f"RESTORING CHECKPOINT FROM {save_dir}...")
#     with open(os.path.join(save_dir, "flax_model.msgpack"), "rb") as f:
#         params = from_bytes(state.params, f.read())

#     with open(os.path.join(save_dir, "opt_state.msgpack"), "rb") as f:
#         opt_state = from_bytes(state.opt_state, f.read())

#     with open(os.path.join(save_dir, "training_state.json"), "r") as f:
#         training_state = json.load(f)
#     step = training_state["step"]

#     logger.info("checkpoint restored")
#     return state.replace(step=step, params=params, opt_state=opt_state), step

def rotate_checkpoints(ckpt_dir:str, save_total_limit:int):
    "Removes older checkpoints so that `save_total_limit` checkpoints are kept"
    # TODO: what to remove is decided using step number only, we might want to improve that
    ckpts = [str(x) for x in Path(ckpt_dir).glob("ckpt-*")]
    # sort checkpoints by step
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split('-')[-1]))
    ckpts_to_delete = ckpts_sorted[:-save_total_limit]
    for ckpt in ckpts_to_delete:
        logger.info(f"Deleting older checkpoint [{ckpt}] due to save_total_limit ({save_total_limit})")
        shutil.rmtree(ckpt)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # if model_args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
    #     )
    # elif model_args.model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
    #     )
    # else:
    #     raise ValueError(
    #         "You are instantiating a new tokenizer from scratch. This is not supported by this script."
    #         "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    #     )

    processor = CLIPProcessor.from_pretrained(model_args.model_name_or_path)
    # tokenizer = CLIPTokenizerFast.from_pretrained(model_args.model_name_or_path)
    tokenizer = processor.tokenizer
    if model_args.model_name_or_path:
        model = FlaxCLIPModel.from_pretrained(
            model_args.model_name_or_path, config=config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )
    else:
        model = FlaxCLIPModel.from_config(
            config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )

    config = model.config
    # Initialize torchvision transforms and jit them for faster processing
    # preprocess = Transform(config.vision_config.image_size)
    preprocess = Transform(config.vision_config.image_size, data_args.augment_images)
    preprocess = torch.jit.script(preprocess)

    eval_preprocess = Transform(config.vision_config.image_size, False)
    eval_preprocess = torch.jit.script(eval_preprocess)

    # Initialize the image-text dataset
    train_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.train_file,
        captions_per_image=data_args.captions_per_image,
        transform=preprocess,
    )

    eval_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.validation_file,
        captions_per_image=1,
        transform=eval_preprocess,
    )

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )
    
    # Use collate function to tokenizer the text and convert the processed images to numpy
    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples]).numpy()
        captions = [example[1] for example in examples]
        inputs = tokenizer(captions, max_length=64, padding="max_length", return_tensors="np")

        batch = {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        return batch

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count() * training_args.gradient_accumulation_steps
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=data_args.preprocessing_num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=data_args.preprocessing_num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # enable wandb tracking
    has_wandb = find_spec("wandb") is not None 
    if jax.process_index() == 0 and has_wandb and ("wandb" in training_args.report_to):
        try:
            import wandb
            wandb.init(
                name=training_args.run_name,
                entity="wandb", 
                project="hf-flax-clip-rsicd",
                sync_tensorboard=True
            )
            wandb.config.update(training_args)
            wandb.config.update(model_args)
            wandb.config.update(data_args)
        except ImportError as e:
            print(e)
            has_wandb = False

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    # Note that this mask is specifically adapted for FlaxGPT2.
    # For other models, one should correct the layer norm parameter naming
    # accordingly.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in [("ln_1", "scale"), ("ln_2", "scale"), ("ln_f", "scale")])
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # create optimizer
    if training_args.adafactor:
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=linear_decay_lr_schedule_fn,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=linear_decay_lr_schedule_fn,
            b1=training_args.adam_beta1,
            b2=training_args.adam_beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            mask=decay_mask_fn,
        )
    if training_args.gradient_accumulation_steps > 1:
        optimizer = optax.MultiSteps(optimizer, training_args.gradient_accumulation_steps)
    grad_accum_steps = training_args.gradient_accumulation_steps

    # Setup train state
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer, dropout_rng=dropout_rng)
    
    if training_args.resume_from_checkpoint:
        state = restore_checkpoint(training_args.resume_from_checkpoint, state)
        resume_step = mb_item(state.step)
    else:
        resume_step = 0

    def cross_entropy(logits, axis):
        logprobs = jax.nn.log_softmax(logits, axis=axis)
        nll = jnp.diag(logprobs)
        ce = -jnp.mean(nll)
        return ce

    def clip_loss(similarity):
        loss = (cross_entropy(similarity, axis=0) + cross_entropy(similarity, axis=1)) / 2
        return loss

    # Define gradient update step fn
    def train_step(state, batch):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = clip_loss(logits)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step // grad_accum_steps)}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics

    # Define eval fn
    def eval_step(params, batch):
        logits = model(**batch, params=params, train=False)[0]
        loss = clip_loss(logits)

        # summarize metrics
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch")

    # Replicate the train state on each device
    state = state.replicate()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed and grad_accum) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    if not training_args.skip_memory_metrics:
        server = jax.profiler.start_server(9999)

    train_time = 0
    train_metrics = []
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # Generate an epoch by shuffling sampling indices from the train dataset
        steps_per_epoch = len(train_dataset) // train_batch_size
        # train
        steps_trained_progress_bar = tqdm(range(steps_per_epoch), desc="Training...", position=1,
                                          leave=False, initial=(resume_step // grad_accum_steps))
        for step, batch in enumerate(train_loader):
            cur_step = epoch * (len(train_dataset) // train_batch_size) + step
            # skip to the step from which we are resuming
            if cur_step < resume_step:
                continue
            
            batch = shard(make_batch(batch))
            state, train_metric = p_train_step(state, batch)
            train_metrics.append(train_metric)
            if step % grad_accum_steps == 0:
                steps_trained_progress_bar.update(1)

            if cur_step % (training_args.logging_steps * grad_accum_steps)== 0 and cur_step > 0:
                # Save metrics
                train_metric = unreplicate(train_metric)
                train_time += time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)
                if has_wandb and jax.process_index() == 0 and ("wandb" in training_args.report_to):
                    # TODO: add accumulation of metrics
                    _metrics = {k if k=="learning_rate" else f"train_{k}":mb_item(v.mean()) for k, v in train_metric.items()}
                    wandb.log({"training_step":cur_step, **_metrics}, commit=True)

                epochs.write(
                    f"Step... ({cur_step} | Loss: {train_metric['loss'].mean()}, Learning Rate: {train_metric['learning_rate'].mean()})"
                )

                train_metrics = []

            # if (cur_step % (training_args.eval_steps * grad_accum_steps) == 0 and
            #     cur_step > 0 and 
            #     model_args.eval_strategy == "steps"):
            #     # ======================== Evaluating ==============================
            #     eval_metrics = []
            #     eval_steps = len(eval_dataset) // eval_batch_size
            #     eval_iter = iter(eval_loader)
            #     for batch in tqdm(eval_loader, desc="Evaluating...", position=2, leave=False):
            #         # Model forward
            #         batch = shard(make_batch(batch))
            #         metrics = p_eval_step(state.params, batch)
            #         eval_metrics.append(metrics)

            #     # normalize eval metrics
            #     eval_metrics = get_metrics(eval_metrics)
            #     eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

            #     # Print metrics and update progress bar
            #     desc = f"Step... ({cur_step} | Eval Loss: {eval_metrics['loss']})"
            #     epochs.write(desc)
            #     epochs.desc = desc

            #     # Save metrics
            #     if has_tensorboard and jax.process_index() == 0:
            #         # cur_step = epoch * (len(train_dataset) // train_batch_size)
            #         write_eval_metric(summary_writer, eval_metrics, cur_step)
            #     if has_wandb and jax.process_index() == 0 and ("wandb" in training_args.report_to):
            #         _metrics = {f"eval_{k}":mb_item(v) for k, v in eval_metrics.items()}
            #         wandb.log({"eval_step":cur_step, **_metrics})

        # we can add an argument to select eval strategy; for now its done every epoch
        if True:
            # ======================== Evaluating ==============================
            eval_metrics = []
            eval_steps = len(eval_dataset) // eval_batch_size
            for batch in tqdm(eval_loader, desc="Evaluating...", position=2, leave=False):
                # Model forward
                batch = shard(make_batch(batch))
                metrics = p_eval_step(state.params, batch)
                eval_metrics.append(metrics)

            # normalize eval metrics
            eval_metrics = get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

            # Print metrics and update progress bar
            desc = f"Step... ({cur_step} | Eval Loss: {eval_metrics['loss']})"
            epochs.write(desc)
            epochs.desc = desc

            # Save metrics
            if has_tensorboard and jax.process_index() == 0:
                # cur_step = epoch * (len(train_dataset) // train_batch_size)
                write_eval_metric(summary_writer, eval_metrics, cur_step)
            if has_wandb and jax.process_index() == 0 and ("wandb" in training_args.report_to):
                _metrics = {f"eval_{k}":mb_item(v) for k, v in eval_metrics.items()}
                wandb.log({"eval_step":cur_step, **_metrics})

        # save checkpoint after each epoch
        if jax.process_index() == 0 and training_args.save_strategy == IntervalStrategy.EPOCH:
            save_dir = f"{training_args.output_dir}/ckpt-{epoch}"
            model.save_pretrained(
                save_dir,
                params=unreplicate(state.params),
                push_to_hub=False, # training_args.push_to_hub, # we don't push intermediate steps
                commit_message=f"Saving weights and logs at epoch {epoch}",
                repo_name_or_path=training_args.output_dir
            )
            if model_args.save_optimizer:
                save_checkpoint(training_args.output_dir, unreplicate(state), cur_step, keep=training_args.save_total_limit, overwrite=True)
            if training_args.save_total_limit is not None:
                rotate_checkpoints(training_args.output_dir, training_args.save_total_limit)
    
    # save model after training is over
    model.save_pretrained(
        training_args.output_dir,
        params=unreplicate(state.params),
        push_to_hub=training_args.push_to_hub,
        commit_message=f"Saving weights and logs at step {cur_step}",
        repo_name_or_path=training_args.output_dir
    )




if __name__ == "__main__":
    main()
