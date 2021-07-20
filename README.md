# CLIP-rsicd

This repository contains code for fine-tuning a [CLIP transformer model](https://huggingface.co/transformers/model_doc/clip.html#transformers.CLIPModel) with image caption pairs from the [RSICD dataset](https://github.com/201528014227051/RSICD_optimal). The work was done as part of the [Flax/JAX community week](https://github.com/huggingface/transformers/blob/master/examples/research_projects/jax-projects/README.md#quickstart-flax-and-jax) organized by Hugging Face and Google's Flax, JAX, and Cloud teams ([announcement](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/7104)).


## Our model

We have two versions of our model available for use on HuggingFace Models.

* [flax-community/clip-rsicd (version 1)](https://huggingface.co/flax-community/clip-rsicd)
* [flax-community/clip-rsicd-v2 (version 2)](https://huggingface.co/flax-community/clip-rsicd-v2)

corresponding to models `bs128x8-lr1e-4-imgaugs-textaugs/ckpt-4` and `bs128x8-lr5e-6-adam/ckpt-1` (see Evaluation below). Both models can be used in the same way as the original CLIP model. Please refer to the [Hugging Face documentation for CLIPModel](https://huggingface.co/transformers/model_doc/clip.html#clipmodel) for details.


## Demo

You can try out our model using text to image and image to image retrieval. [Check out Demo](https://huggingface.co/spaces/sujitpal/clip-rsicd-demo).


## Team

* Arto (@arampacha)
* Dev Vidhani (@devvidhani)
* Goutham Venkatesh (@goutham794)
* Mayank Bhaskar (@cataluna84)
* Ritobroto Ghosh (@ghosh-r)
* Sujit Pal (@sujitpal)


## Augmentation Strategy

Because our dataset was fairly small, we used both image augmentation and text augmentation to fine-tune our model. Image augmentation was done inline using built in transforms from Pytorch's Torchvision package. Text augmentations were done offline via backtranslation using the [ROMANCE models from Helsinki-NLP](https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE).


## Evaluation Results

A subset of the image test set had file names that indicated that the image belonged to one of 30 image categories in the RSICD dataset. Evaluation was done by comparing the CLIP encoding of each image with CLIP encodings of each of 30 synthetic caption sentences of the form `"An aerial photograph of {category}"`, and the checking to see that the correct category was found within the first k ranked predictions, for k=1, 3, 5, and 10.

The `baseline` model represents the pre-trained `openai/clip-vit-base-patch32` CLIP model. This model was fine tuned using captions and images from the RSICD dataset, and resulted in significant boosts in performance, as shown below.


| Model-name                               | k=1   | k=3   | k=5   | k=10  |
| ---------------------------------------- | ----- | ----- | ----- | ----- |
| baseline                                 | 0.572 | 0.745 | 0.837 | 0.939 |
| bs128x8-lr1e-4-augs/ckpt-2               | 0.819 | 0.950 | 0.974 | 0.994 |
| bs128x8-lr1e-4-imgaugs/ckpt-2            | 0.812 | 0.942 | 0.970 | 0.991 |
| bs128x8-lr1e-4-imgaugs-textaugs/ckpt-4   | 0.843 | 0.958 | 0.977 | 0.993 |
| bs128x8-lr5e-5-imgaugs-textaugs/ckpt-8   | 0.831 | 0.959 | 0.977 | 0.994 |
| bs128x8-lr5e-5-imgaugs/ckpt-4            | 0.746 | 0.906 | 0.956 | 0.989 |
| bs128x8-lr5e-5-imgaugs-textaugs-2/ckpt-4 | 0.811 | 0.945 | 0.972 | 0.993 |
| bs128x8-lr5e-5-imgaugs-textaugs-3/ckpt-5 | 0.823 | 0.946 | 0.971 | 0.992 |
| bs128x8-lr5e-5-wd02/ckpt-4               | 0.820 | 0.946 | 0.965 | 0.990 |
| [bs128x8-lr5e-6-adam/ckpt-1]((https://huggingface.co/flax-community/clip-rsicd-v2)) | **0.883** | **0.968** | **0.982** | **0.998** |

