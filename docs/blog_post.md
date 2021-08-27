---
title: Fine tuning CLIP with Remote Sensing (Satellite) images and captions
thumbnail: /blog/assets/clip-rsicd/thumbnail.png
---

# Fine tuning CLIP with Remote Sensing (Satellite) images and captions

_by Artashes Arutiunian (@arampacha), Dev Vidhani (@devvidhani), Goutham Venkatesh (@goutham794), Mayank Bhaskar (@cataluna84), Ritobrata Ghosh (@ghosh-r), and Sujit Pal (@sujitpal)_


In July this year, [Hugging Face](https://huggingface.co/) organized a [Flax/JAX Community Week](https://github.com/huggingface/transformers/blob/master/examples/research_projects/jax-projects/README.md#quickstart-flax-and-jax), during which the community was invited to submit projects to train Hugging Face transformer models in the areas of Natural Language Processing (NLP) and Computer Vision (CV) on Tensor Processing Units (TPU) using [Flax](https://github.com/google/flax), a neural network library and ecosystem for [JAX](https://github.com/google/jax), a linear algebra library (like numpy) which can do automatic differentiation ([Autograd](https://github.com/hips/autograd)) and can compile down to [XLA](https://www.tensorflow.org/xla). [Google Cloud](https://cloud.google.com/) were co-sponsors of this event.

They organized 3 days of lectures during which speakers from Hugging Face and Google Cloud talked about TPUs, JAX, Flax, and how to use them to train Hugging Face transformer models (recordings for [day #1](https://www.youtube.com/watch?v=fuAyUQcVzTY), [day #2](https://www.youtube.com/watch?v=__eG63ZP_5g), and [day #3](https://www.youtube.com/watch?v=ZCMOPkcTu3s) available). This was followed by around 10 days of actual work, during which time Google Cloud provided each team with a GCP instance with a TPU. Each team was expected to train one or more Hugging Face models using JAX/Flax and share them with the community via their [model hub](https://huggingface.co/models). In addition, teams were asked to provide a demo [Hugging Face spaces](https://huggingface.co/spaces) showcasing the capabilities of their model. Approximately 100 teams participated in the event, and it resulted in 170 models and 36 demo spaces.

Our team, like probably many others, is a distributed one, spanning 12 time zones. Our common thread is that we all belong to the [TWIML Slack Channel](https://twimlai.slack.com/), where we came together based on a shared interest in Artificial Intelligence (AI) and Machine Learning (ML) topics. 

We decided that we would fine tune the [CLIP Network from OpenAI](https://openai.com/blog/clip/) with satellite images and captions from the [RSICD dataset](https://github.com/201528014227051/RSICD_optimal). The CLIP network learns visual concepts by being trained with image and caption pairs in a self-supervised manner, by using text paired with images found across the Internet. During inference, the model can predict the most relevant image given a text description, and the most relevant text description given an image. CLIP is powerful enough to be used in zero-shot manner on everyday images. However, we felt that satellite images were sufficiently different from everyday images, that it would be useful to fine-tune CLIP with them. Our intuition turned out to be correct, as the evaluation results (described below) shows.

Our project ended up placing third at the event, which was a very pleasant surprise. We are incredibly humbled and gratified that the judges saw the potential of applications that could benefit from models such as ours. We are very grateful to the organizers for teaching us about Flax/JAX and for providing the resources to create and showcase our model. We are also very thankful to the other participants and organizers for sharing their knowledge and insights so freely, we have each benefited enormously from these interactions. You can read about our project on our [project page](https://github.com/arampacha/CLIP-rsicd), download our [trained model](https://huggingface.co/flax-community/clip-rsicd-v2) to use for inference, or see it in action on our [demo](https://huggingface.co/spaces/sujitpal/clip-rsicd-demo).

In this post, we describe details of our training and evaluation process, and our plans for future work on this project.

## Training

### Dataset

### Model

### Data Augmentation

### Parameters and Plots

## Evaluation

### Metrics

### Demo

## Future Work

## Conclusion




