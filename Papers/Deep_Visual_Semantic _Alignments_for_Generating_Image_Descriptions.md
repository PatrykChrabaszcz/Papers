# Deep Visual-Semantic Alignments for Generating Image Descriptions
### Authors
Andrej Karpathy, Li Fei-Fei
### Arxiv Date 
Dec 2014

### Published 
CVPR 2015


### Abstract
We present a model that generates natural language descriptions of images and their regions. 
Our approach leverages datasets of images and their sentence descriptions to learn about the inter-modal correspondences between language and visual data. 
Our alignment model is based on a novel combination of Convolutional Neural Networks over image regions, bidirectional Recurrent Neural Networks over sentences, and a structured objective that aligns the two modalities through a multimodal embedding. 
We then describe a Multimodal Recurrent Neural Network architecture that uses the inferred alignments to learn to generate novel descriptions of image regions.
We demonstrate that our alignment model produces state of the art results in retrieval experiments on Flickr8K, Flickr30K and MSCOCO datasets. 
We then show that the generated descriptions significantly outperform retrieval baselines on both full images and on a new dataset of region-level annotations. 

### Summary
- Current models perform well on object classification tasks.
- Step towards generating dense description of images.
- Models have to be rich enoough to model the context and use language to describe it.
- Datasets are constructed such that for each image we have lot of sentences describing different parts of the image.
- They make the code available cs.stanford.edu/people/karpathy/deepimagesen
- Sentence descriptions reference objects on an image and their properties.
- They use Region Convolutional Neural Network (RCNN) to detect objects in every image.
- They compute representation with CNN of full image and 19 top boxes from RCNN. Representation has 4096 numbers.
- Then they transform this representation into multimodal embedding space (1000-1600 dimensions)
- Each image represented as 20 vectors 
- They use BRNN (with relu) to compute word representations. BRNN takes a sequence of words and transforms each word to the same space as image representation.
- They use word embedding on the input to the BRNN (300 dimensional word2vec, they keep it fixed to not overfit)
- In practive we can train word embedding from random initialization, only little change in final performance
- Word representation (output from BRNN) will take into account also words around given word. However, they notice that given word has the most influence.
- They use 300-600 neurons in the BRNN network
- Having image vectors and word vectors in the same embedding space we can define a measure of similarity
- Similarity between sentence and image is a sum of dot products (thresholded at 0) of every possible combination of word and image vectors.
- They mention also additional Multiple Instance Learning objective (?)
- They simplify objective, they sum dot products of words with only the best image region instead of all image regions
- This simplified objective improves performance

# TODO Finish from 3.1.4



### From the paper
- We develop a deep neural network model that infers the latent alignment between segments of sentences and the region of the image that they describe. Our  model  associates  the  two  modalities  through  a common,  multimodal  embedding  space  and  a  structured objective.  We validate the effectiveness of this approach  on  image-sentence  retrieval  experiments  in which we surpass the state-of-the-art.
- We introduce a multimodal Recurrent Neural Network architecture that takes an input image and generates its description in text.  Our experiments show that the generated sentences significantly outperform retrieval-based baselines, and produce sensible qualitative predictions. We then train the model on the inferred correspondences and evaluate its performance on a new dataset of region-level annotations.
- Together with their additional Multiple Instance Learning objective, this score carries the interpretation that a sentence fragment aligns to a subset of the image regions whenever the  dot  product  is  positive.
- We  found  that  the  following reformulation simplifies the model and alleviates the need for additional objectives and their hyperparameters (...) Here, every word st aligns to the single best image region.
- This objective encourages aligned image-sentences pairs to have a higher score than misaligned pairs, by a margin


### Referenced Work
Understanding and generating simple image descriptions.
Generating sentences from images.
Deep fragment embeddings for bidirectional image sentence mapping.
Rich feature hierarchies for accurate object detection and semantic segmentation (RCNN Network)
Bidirectional recurrent neural networks
Distributed representations of words and phrases and their compositionality (word2vec)

