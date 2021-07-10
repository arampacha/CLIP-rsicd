# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""TODO: Add a description here."""


import json
import numpy as np
from PIL import Image
from collections import defaultdict
import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLs = {
    'first_domain': "https://huggingface.co/great-new-dataset-first_domain.zip",
    'second_domain': "https://huggingface.co/great-new-dataset-second_domain.zip",
}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class NewDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("0.1.0")

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')

    def _info(self):
        features = datasets.Features(
            {
                'filename': datasets.Value(dtype='string'),
                'imgid': datasets.Value(dtype='int64'),
                'tokens': datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value(dtype='string'), length=-1), length=5),
                'sentences': datasets.Sequence(datasets.Value(dtype='string'), length=5),
                'split': datasets.Value(dtype='string'),
                'sentids': datasets.Sequence(feature=datasets.Value(dtype='int64'), length=5),
                'image': datasets.Array3D(shape=(224, 224, 3), dtype='uint8')
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # my_urls = _URLs[self.config.name]
        # data_dir = dl_manager.download_and_extract(my_urls)
        data_dir = self.config.data_dir
        with open(f"{data_dir}/dataset_rsicd.json") as f:
            ds = json.load(f)
        _items = defaultdict(list)
        for e in ds["images"]:
            _items[e["split"]].append(e)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"items":_items["train"], "data_dir":data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"items":_items["test"], "data_dir":data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"items":_items["val"], "data_dir":data_dir},
            ),
        ]

    def _generate_examples(self, items, data_dir):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        for _id, item in enumerate(items):
            image = np.asarray(Image.open(f"{data_dir}/RSICD_images/{item['filename']}"))
            sentences = item.pop('sentences')
            sample = {"image":image, 
                      "sentences":[s["raw"] for s in sentences], 
                      "tokens":[s["tokens"] for s in sentences], 
                      **item}
            yield _id, sample

