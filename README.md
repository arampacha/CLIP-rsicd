# CLIP-rsicd

This repository contains code for fine-tuning a [CLIP transformer model](https://huggingface.co/transformers/model_doc/clip.html#transformers.CLIPModel) with image caption pairs from the [RSICD dataset](https://github.com/201528014227051/RSICD_optimal). The work was done as part of the [Flax/JAX community week](https://github.com/huggingface/transformers/blob/master/examples/research_projects/jax-projects/README.md#quickstart-flax-and-jax) organized by Hugging Face and Google's Flax, JAX, and Cloud teams ([announcement](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/7104)).

See our [project proposal](https://discuss.huggingface.co/t/fine-tune-clip-on-satellite-images-captions/7612) on the Hugging Face Discussion Board.

## Applications

We think our model will be useful in industries that depend on remote sensing or satellite imagery. Our model provides the ability to quickly search large collections of such images for specific features. Some examples of applications that would benefit from such functionality are listed below.

* National Defense and anti-terrorism initiatives -- analysis of satellite imagery or drone footage to quickly find specific features and address them before they become bigger problems.
* Climate Change initiatives -- ability to spot and address wildfires and refugee migration patterns before they become unmanageable; ability to spot things that might otherwise go unnoticed, such as island-size plastic garbage floating in oceans; ability to systematically track deforestation and desertification over time; etc.
* Consumer Applications -- can help provide new functionality such as "make your own vacation" feature for map providers (Google Maps, Apple Maps). User selects how far they are willing to travel and the features they are looking for at their destination, for example, beaches and hiking trails, or mountains and snow, etc., and the application returns a ranked list of destinations within the desired travel radius.

Finally, this project is also a demonstration of how effective fine-tuned CLIP models can be for specialized domains. The search strategies that we demonstrate for our fine-tuned CLIP model -- text to image, image to image, and text feature in image -- would work equally well in other domains, such as for medical images. Thus, fine-tuned CLIP models such as ours have the potential to become digital assistants to humans in any industry that have to deal with large number of images.

## Our model

We have two versions of our model available for use on HuggingFace Models.

* [flax-community/clip-rsicd (version 1)](https://huggingface.co/flax-community/clip-rsicd)
* [flax-community/clip-rsicd-v2 (version 2)](https://huggingface.co/flax-community/clip-rsicd-v2)

Both fine-tuned models listed above can be used in the same way as the original CLIP model. The CLIP model learns to project images and text onto a common embedding space such that similar (image, image), (text, image), and (text, text) pairs appear closer together in this space than dissimilar pairs. The Hugging Face CLIP API offers methods to retrieve the embeddings from text and image inputs, and rank a set of text descriptions on how similar they are to an image, or a set of images on how similar they are to a text description. 

Our model cards have **code templates** that can help get you started using these models for inference.

In addition, the [Hugging Face documentation for CLIPModel](https://huggingface.co/transformers/model_doc/clip.html#clipmodel) provides more details on its use.

## Demo

Our demo uses our fine-tuned CLIP model to provide the following functionality. For the first two services, we have previously encoded the images from the RSICD test split with our fine-tuned CLIP model and stored these encodings in an [NMSLib](https://github.com/nmslib/nmslib) Approximate Nearest Neighbor based retrieval.

* Text to Image Search -- user enters a text feature describing some natural or man-made geographical feature, for example, "beach", "mountain", "school", or "baseball field". The query is encoded with our fine-tuned CLIP model and matched against the NMSLib index of image encodings. The top-10 ranked list of images with vectors that have high cosine similarity to the query vector are returned.
* Image to Image Search -- user enters an image filename from the RSICD test set. This image is encoded and matched against the NMSLib index of image encodings. The top-10 ranked list of images with vectors that have the highest cosine similarity to the query image vector are returned.
* Finding Text Feature in Image -- an arbitary image and a feature to find in the image are provided to the model. The model partitions the image into patches, and sends the batch of patch images and the feature to the model. The model returns a ranked list of patches for the feature, where highly ranked patches are more likely to contain the feature being asked for.

[Check out our Demo](https://huggingface.co/spaces/sujitpal/clip-rsicd-demo) (_only accessible to Hugging Face Spaces beta participants currently_)


## Training Details

The model was trained using Flax/JAX on TPU-v3-8. Flax/JAX models can be trained on GPU and CPU as well, although the latter is probably not practical in this case. On TPU, we used a batch size of 1024 (128 for each TPU device) and on GPU we used a batch size of 32. Best training results were observed using the [Adafactor](https://arxiv.org/abs/1804.04235) and [Adam](https://arxiv.org/abs/1412.6980) optimizers with a learning rate of 5e-5 and a linear learning rate schedule

The script that we used for fine-tuning the CLIP models on a TPU VM provided by the Google Cloud Platform (GCP) is [run_clip_flax_tv.py](https://github.com/arampacha/CLIP-rsicd/blob/master/run_clip_flax_tv.py). The Evaluation Results reported below are for models trained using this script.

We have provided a [Colab Notebook](https://colab.research.google.com/github/arampacha/CLIP-rsicd/blob/master/nbs/Finetuning_CLIP_with_HF_and_jax.ipynb) containing a similar script that you can use to reproduce our training on Colab (GPU).

### Dataset

The Remote Sensing Image Caption Dataset ([RSICD](https://github.com/201528014227051/RSICD_optimal)) is a collection of about 10,000 images collected from Google Earth, Baidu Map, MapABC, and Tianditu and provided to the research community for advancement of remote sensing captioning via [Exploring Models and Data for Remote Sensing Image Caption Generation](https://arxiv.org/abs/1712.07835) (Lu et al, 2017). The images are provided as (224, 224) RGB images at various resolutions. Each image has upto 5 captions associated with it.

The [UCM dataset](https://mega.nz/folder/wCpSzSoS#RXzIlrv--TDt3ENZdKN8JA) is based on the UC Merced Land Use Dataset. It consists of 2100 images belonging to 21 classes (100 images per class). The dataset provides 5 captions for each image. The images are (256, 256) RGB images with pixel resolution of 0.3048m.

The [Sydney dataset](https://mega.nz/folder/pG4yTYYA#4c4buNFLibryZnlujsrwEQ) contains images of Sydney, Australia from Google Earth. The dataset consists of 613 images belonging to 7 classes. Images are (500, 500) RGB images with pixel resolution 0.5m. The dataset provides 5 captions for each image.

### Augmentation Strategy

Because our dataset was fairly small, we used both image augmentation and text augmentation to regularize our dataset and prevent overfitting.

Image augmentation was done inline using built in transforms from Pytorch's Torchvision package. The transformations used were Random Cropping, Random Resizing and Cropping, Color Jitter, and Random Horizontal and Vertical flipping.

Text augmentations to image captions were done offline via backtranslation using the [Marian MT](https://huggingface.co/transformers/model_doc/marian.html) family of translation models, specifically the [ROMANCE models from Helsinki-NLP](https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE). Each augmentation corresponded to backtranslation through a different pair of language models.

This Weights and Biases report describes [the impact of Image and Text Augmentations on the Training Regime](https://wandb.ai/wandb/hf-flax-clip-rsicd/reports/Fine-tuning-CLIP-on-RSICD--Vmlldzo4NzMyOTg) of our fine-tuned CLIP Models.


## Evaluation Results

We used a subset of the RSICD test set with file names that specified that the image belonged to one of 30 image categories. Evaluation was done by comparing the CLIP encoding of each image with CLIP encodings of each of 30 synthetic caption sentences of the form `"An aerial photograph of {category}"`. Categories corresponding to captions with the top k scores (for k=1, 3, 5, and 10) were compared with the "label" category indicated by the image name. The score is 1 if the top-k predicted classes contained the label category (for k=1, 3, 5, and 10), otherwise the score is 0. The scores are averaged over the entire set of evaluation images and reported for various values of k, as shown below.

The `baseline` model represents the pre-trained `openai/clip-vit-base-patch32` CLIP model. This model was fine tuned using captions and images from the RSICD dataset, and resulted in significant boosts in performance, as shown below.


| Model-name                               | k=1   | k=3   | k=5   | k=10  |
| ---------------------------------------- | ----- | ----- | ----- | ----- |
| baseline                                 | 0.572 | 0.745 | 0.837 | 0.939 |
| bs128x8-lr1e-4-augs/ckpt-2               | 0.819 | 0.950 | 0.974 | 0.994 |
| bs128x8-lr1e-4-imgaugs/ckpt-2            | 0.812 | 0.942 | 0.970 | 0.991 |
| [bs128x8-lr1e-4-imgaugs-textaugs/ckpt-4](https://huggingface.co/flax-community/clip-rsicd)<sup>2</sup>   | 0.843 | 0.958 | 0.977 | 0.993 |
| bs128x8-lr5e-5-imgaugs-textaugs/ckpt-8   | 0.831 | 0.959 | 0.977 | 0.994 |
| bs128x8-lr5e-5-imgaugs/ckpt-4            | 0.746 | 0.906 | 0.956 | 0.989 |
| bs128x8-lr5e-5-imgaugs-textaugs-2/ckpt-4 | 0.811 | 0.945 | 0.972 | 0.993 |
| bs128x8-lr5e-5-imgaugs-textaugs-3/ckpt-5 | 0.823 | 0.946 | 0.971 | 0.992 |
| bs128x8-lr5e-5-wd02/ckpt-4               | 0.820 | 0.946 | 0.965 | 0.990 |
| [bs128x8-lr5e-6-adam/ckpt-1](https://huggingface.co/flax-community/clip-rsicd-v2)<sup>1</sup> | **0.883** | **0.968** | **0.982** | **0.998** |


_1 - our best model, 2 - our second best model_

## Team

* Arto (@arampacha)
* Dev Vidhani (@devvidhani)
* Goutham Venkatesh (@goutham794)
* Mayank Bhaskar (@cataluna84)
* Ritobroto Ghosh (@ghosh-r)
* Sujit Pal (@sujitpal)


