### Long short-term memory
- Main LSTM paper

### Learning phrase representations using rnn encoder-decoder for statistical machine translation.
- Main GRU paper


### An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling (Mar 2018) (Arxiv) Bai et al. (Carnegie Mellon University)
- Claim that sequence modeling should be addressed using CNNs instead of RNNs.
- Say that recent work showed that CNNs outperformed RNNs on audio synthesis, word-level language modeling, and machine translation.
- Tasks: mnist and p-mnist, polyphonic music modeling (JSB Chorales), word and character-level language modeling (Penn TreeBank, Wikitext-103, LAMBADA, text-8), synthetic stress tests (the adding problem, copy-memory)
- Their architecture is called TCN (Temporal Convolutional Network)
- Code at https://github.com/locuslab/TCN
- Uses casual convolutions (no information leakage from the future)
- Uses residual layers and dilated CNNs (exponential dilation: 1, 2, 4, ... )
- Combines simplicity, autoregressive prediction, and very long memory
- Simpler than WaveNet
- Produces outputs of the same length as inputs.
- Skip connection has optional 1x1 convolution (when feature sizes do not match)
- They use weight normalization and spacial dropout (dropping out whole channel at once).
- Says that CNN uses less memory for training compared to RNN (usually)
- For each prediction it has to process the full sequence again (Disadvantage compared to RNN) 
- They train with Adam, learning_rate 0.002 
- Found that gradient cliping helps and they pick the maximum norm from [0.3, 1]
- For RNNs they do a grid search over set of parameters: optimizer, recurrent_drop [0.05,0.5], learning rate, gradient clipping, and initial forget-gate bias
- When comparing the models they keep the number of parameters fixed.
- Proves on copy-memory task than TCN has better memory than GRU and LSTM
- They predict that RNNs will fail the battle with CNNs :).

From the paper:
- "We conduct a systematic evaluation of generic convolutional and recurrent architectures for sequence modeling."
- "Our results indicate that a simple convolutional architecture outperforms canonical recurrent networks such as LSTMs across a diverse range of tasks and datasets, while demonstrating longer effective memory."
- "We show that despite the theoretical ability of recurrent architectures to capture infinitely long history, TCNs exhibit substantially longer memory, and are thus more suitable for domains where a long history is required."
- "While these combinations show promise" {about recent upgrades to RNNs and CNNs} "our study here focuses on a comparison of generic convolutional and recurrent architectures."
- "Our aim is to distill the best practices  in  convolutional  network  design  into  a  simple architecture that can serve as a convenient but powerful starting point."
- "(...) build very long effective history sizes (.. ) using a combination of very deep networks (augmented with residual layers) and dilated convolutions."
- "TCN is much simpler than WaveNet (no skip connections across layers, conditioning, context stacking, or gated activations)."
- "Within a residual block, the TCN has two layers of dilated causal convolution and non-linearity, for which we used the rectified linear unit."
- "(...) convolutions can be done in parallel since the same filter is used in each layer.Therefore, in both training and evaluation, a long input sequence can be processed as a whole in TCN, instead of sequentially as in RNN."
- "TCN thus avoids the problem of exploding/vanishing gradients, which is a major issue for RNNs"
- "The experimental results indicate that TCN models substantially outperform generic recurrent architectures such as LSTMs and GRUs."
- "(...) showed that the “infinite memory” advantage of RNNs is largely absent in practice. TCNs exhibit longer memory than recurrent architectures with the same capacity."
- "Due to the comparable clarity and simplicity of TCNs, we conclude that convolutional networks should be regarded as a natural starting point and a powerful toolkit for sequence modeling"

### Diagonal RNNs in symbolic music modeling (Apr 2017) Subakan et al. (University of Illinois) (Adobe Systems)

- Use diagonal matrices for recurrent connections.
- When comparing different cells they sample 60 configurations and compare between top 6 (10%).
- They have a github with the code https://github.com/ycemsubakan/diagonal_rnns
- They report improvements when using diagonal RNNs on 3 out of 4 datasets.
- Only used for symbolic music datasets.
- Diagonal networks use fewer parameters.

From this paper:
- "The novelty is simple: We use diagonal recurrent matrices instead of full"
- "we empirically show that in symbolic music modeling, using a diagonal recurrent matrix in RNNs results in significant improvement in terms of convergence speed and test likelihood."
- "We see that using diagonal recurrent matrices results in an improvement in test likelihoods in almost all cases we have explored in this paper."

### Dilated Recurrent Neural Networks (Oct 2017) (NIPS 2017) Chang, Zhang, Han et al. (IMB) 
https://arxiv.org/abs/1710.02224
- Publicly available tensorflow code https://github.com/code-terminator/DilatedRNN
- Run experiments on: 
- Claims that Dilated RNNs improve over base RNNs 
- Claims that dilations make vanilla RNNs get state-of-the art
- Should be beneficial for learning on a very long sequences


From this paper:
- "We introduce a new dilated recurrent skip connection as the key building block of the proposed architecture. These alleviate gradient problems and extend the range of temporal dependencies like conventional recurrent skip connections, but in the dilated version require fewer parameters and significantly enhance computational efficiency."
- "We stack multiple dilated recurrent layers with hierarchical dilations to construct a DILATED RNN, which learns temporal dependencies of different scales at different layers."
- "We present the mean recurrent length as a new neural memory capacity measure that reveals the performance difference between the previously developed recurrent skip-connections and the dilated version."
- " We also verify the optimality of the exponentially increasing dilation distribution used in the proposed architecture."
TODO: Finish this paper

### TODO:
Clockwork RNN 
Phased LSTM
Hierarchical Multiscale RNN


### Learning to skim text

- Chang about this paper: "proposed learning-
based RNNs with the ability to jump (skim input text) after seeing a few timestamps worth of data;
although the authors showed that the modified LSTM with jumping provides up to a six-fold speed
increase, the efficiency gain is mainly in the testing phase."

### Feudal networks for hierarchical reinforcement learning
- Chang about this paper: "(...) can be viewed as a special case
of our model, which contains only one dilated recurrent layer with fixed dilation. The main purpose
of their model is to reduce the temporal resolution on time-sensitive tasks. Thus, the Dilated LSTM
is not a general solution for modeling at multiple temporal resolutions"


### Deep Speech 2: End-to-End Speech Recognition in English and Mandarin (Dec 2015) (ICML 2016) Amodei et al. (Baidu)
- they increase the training speed 7x compared to their previous approach
- ASR system - Automatic Speech Recognition
- They reduce error rates for English model by 43% compared to the previous system
- They focus on the  model  architecture,  large  labeled  training datasets, and computational scale.
- They train with CTC loss
- English system trained on 12000 hours of speech.
- They train one model on 8 or 16 GPUs. (They use synchronous SGD)
- They put a lot of effort into making it parallelizable and fast 
- On some benchmarks they are better than humans
- Up to 7 GRU cells on top of the CNN cells



- "We  show  that  an  end-to-end  deep  learning  approach  can  be  used  to  recognize either English or Mandarin Chinese speech."
- "We explore architectures with up to 11 layers including many bidirectional recurrent layers and convolutional layers."
- The RNN model is composed of several layers of hidden units.  The architectures we experiment with consist of one or more convolutional layers, followed by one or more recurrent layers, followed by one or more fully connected layers."
- "In contrast " {to the Laurent's work}", we demonstrate that when applied to very deep networks of simple RNNs on large data sets, batch normalization
substantially improves final generalization error while greatly accelerating training.


### Deep Residual Learning for Image Recognition (Dec 2015) He et al. (CVPR 2016) (Microsoft)
- Resnet allows for training much deeper networks.
- First place on Imagenet classification 2015, with 3.57% top 5 error (ensemble).
- Problem with convergence with many layers already addressed before by proper initialization and use of batch normalization.
- With plain network more layers will give better performance up to some point, from that point adding more layers degrades performance on both 
- Even though bigger models can act as shallow ones (If we use identity layers), they are still harder to optimize and at the end have lower train accuracy.
- Use shortcut identity connections to skip layers and implement residual learning.
- Imagenet model has 152 layers.
- Perform linear projection if number of features does not match
- They do not observe advantages if we only skip 1 layer (They test with blocks that have 2 or 3 layers in one res block)
- They do skip the last layer, they do not skip the first layer

- "We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions."
- "Instead  of  hoping  each  few  stacked  layers  directly  fit  a desired  underlying  mapping,  we  explicitly  let  these  layers fit a residual mapping."
- "Our extremely deep residual nets are easy to optimize, but the counterpart “plain” nets (that simply stack layers) exhibit higher training error when the depth increases."
- "Our deep residual nets can easily enjoy accuracy gains from greatly increased depth, producing results substantially better than previous networks."

TODO: Finish

### Dilated Recurrent Neural Networks (Oct 2017) Chang et al. (NIPS 2017) (IBM)
- Learning long sequences is complex, because: complex dependencies, vanishing/exploding gradients, efficient parallelization.
- Uses dilated recurrent connections 
- Good for tasks with very long dependencies 
- They make the code available https://github.com/code-terminator/DilatedRNN
- Analogous to the dilated CNN
- It is cell independent
- They use exponentialy increasing dilations 


- "Learning with recurrent neural networks (RNNs) on long sequences is a notori-
ously difficult task." 

### Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
- Compare GRU LSTM and tanh RNN

### On the Properties of Neural Machine Translation: Encoder-Decoder Approaches
- Original GRU paper

### Learning  Long-Term  Dependencies  with  Gradient Descent is Difficult (1994) Bengio et al. (University of Montreal)
- RNN good for sequences however practical difficulties when relations span across long horizons
- Show how standard gradient descent performance decreases as sequence size increases 
- Propose some alternatives to standard gradient based optimization
- Amount of time information is kept in RNN is not fixed, it depends on RNN weights and on the input data
- Standard training is backpropagation through time
- Forward propagation algorithms (More computationaly expensive and local in time) can be applied online and produce a partial gradient after each time step 
- Early experiments with RNN indicate that parameters settle in sub optimal regions where they only capture short dependencies.
- They have some theoretical results that either system is resistant to noise in the input or is efficiently trained by gradient descent.
- Toy problem where only first L values determine sequence class. Sequences arbitrarly long. Other values irrelavant and should be ignored.

From this paper: "Theanalysis  shows  thatwhentryingtosatisfyconditions1)and2)above,  the  magnitudeofthe  derivativeofthe  stateofa  dynamical  system  at  timetwith  respectto  the  stateattime0decreases  exponentially  astincreases. "


TODO: Finish

### On the difficulty of training recurrent neural networks
- Analyze how singular values relate to vanishing/exploding gradient problem.
- For some artificial problems that show that there are high walls in the loss surface that might cause high gradients and big jumps during optimization

From the paper:
- "There are two widely known issues with prop-erly  training  recurrent  neural  networks,  thevanishingand  theexplodinggradient  prob-lems  detailed  in  Bengioet  al."
- "We propose a gradient norm
clipping strategy to deal with exploding gra-
dients and a soft constraint for the vanishing
gradients  problem.   We  validate  empirically
our hypothesis and proposed solutions in the
experimental section."
- "While  in  principle  the  recurrent  network  is  a  simpleand  powerful  model,  in  practice,  it  is  hard  to  trainproperly.  Among the main reasons why this model isso unwieldy are thevanishing  gradientandexplodinggradientproblems described in Bengioet al.(1994)."
- Introduced in Bengioet al.(1994), theexploding gradientsproblem refers to the large increase in the normof the gradient during training.  Such events are due tothe explosion of the long term components, which can grow exponentially more than short term ones
- " The
vanishing gradients
problem refers to the opposite be-
haviour, when long term components go exponentially
fast to norm 0, making it impossible for the model to
learn correlation between temporally distant events."
- "Using an L1 or L2 penalty on the recurrent weights canhelp with exploding gradients.  Assuming weights areinitialized to small values, the largest singular valueλ1ofWrecis probably smaller than 1.  The L1/L2 termcan ensure that during trainingλ1stays smaller than1,  and  in  this  regime  gradients  can  not  explode. This  approach  limits  the  model  to  singlepoint  attractor  at  the  origin,  where  any  informationinserted in the model dies out exponentially fast.  Thisprevents  the  model  to  learn  generator  networks,  norcan it exhibit long term memory traces"
- "one  mechanism  to  deal
with the exploding gradient problem is to rescale their
norm whenever it goes over a threshold"
- "One  good
heuristic for setting this threshold is to look at statis-
tics on the average norm over a sufficiently large num-
ber  of  updates.   In  our  experience  values  from  half
to ten times this average can still yield convergence,
though convergence speed can be affected."
- " We  put  forward  a  hypothesis  stat-
ing  that  when  gradients  explode  we  have  a  cliff-like
structure in the error surface and devise a simple so-
lution based on this hypothesis, clipping the norm of
the exploded gradients"




### Finding structure in time

### Learning  representations  by  back-propagating errors.

### Generalization of backpropagation with application to a recurrent gas market model.


### On the state of the art of evaluation in neural language models (Jul 2017) Melis et al. (Deep Mind)
- Lot of SOTA but different evaluation procedures
- They reevaluate experiments with different architectures and regularizations
- Standard LSTM architectures, when properly regularised, outperform more recent models
- They discuss the problem that it would be better to properly evaluate models, describe their sensitivity to hyperparameters etc.. But this is also costly so we need to agree on some standardized procedures. 
- They test NAS, LSTM and Recurrent Highway Networks
- They use variational dropout from Gal or recurrent dropout from 
- They use datasets:  Penn Treebank, Wikitext-2 ,  Enwik8
- batch size of 64, truncated backpropagation with 35 time steps, forward states from previous minibatch
- start with zero state
- They use Adam with beta 1 set to 0 (More like RMSProp)
- Some experiments show that MC dropout would bring very small improvements (100x more costly) so they dont use it
- They use Google Vizier for HP Tuning
- They optimize: learning rate, input embedding ratio, input dropout,state dropout,output dropout,weight decay
- intra-layer dropout additionaly for RNNs.
- They parameterize over number of parameters, hidden size is derived from it
- Their experiments see no difference between Gal and Semeniuta dropouts.
- They check how the tuner overfits (Retrain with the same HP setting)

From the paper:
- "Even with this small set, thousands of evaluations are required to reach convergence"


### Sequence to sequence learning with neural networks (Sep 2014) (NIPS 2014) Sutskever et al. Google
- DNNs could not be used to map sequences to squences before
- They use datasets:  WMT’14 English to French
- Say their approach is related to "Recurrent continuous translation models."
- Use simple left to right beam search decoder
- They improve over SMT baseline
- They can handle long sequences because they reverse the order of input sequence (This was one of the key technical contributions)
- Resulting model is quite invariant to word order and active/passive voice
- Simple technique to do sequence to sequence modeling. Map input sequence into fixed vector and then decode that vector
- Last hidden state is used as this fixed vector representation
- Then the next LSTM is conditioned on that hidden state
- Different LSTMs for encoding and decoding (Possible to train on different language pairs)
- Find that Deep LSTMs work better so they use 4 layers
- If we want to translate a, b, c into A, B, C then its better to feed the input data as c, b, a and keep outputs as A, B, C. They have some hypthesis why it would work but important thing is that it works in practice
- Setup : 4layer LSTM 1000 neurons, 1000 dim word embedding, 160 000 input vocabulary, 80 000 output vocabulary
- They use naive softmax on the output
- SGD with learning rate 0.7 no momentum, halving every half epoch after first 5 epochs
- They have gradient norm clipping rule
- They batch sentences to get similar length sentences in the same batch (2x speed up compared to the naive approach)
- Parallelize implementation over 8 GPUs and train for 10 days
- They visualize sentence vectors in 2D using PCA 


- "We were able to do well on long sentences because we reversed the order of words in the source sentence but not the arget sentences in the training and test set. By doing so, we introduced many short term dependencies that made the optimization problem much simpler (see sec. 2 and 3.3).  As a result, SGD could learn LSTMs that had no trouble with long sentences.  The simple trick of reversing the words in the source sentence is one of the key technical contributions of this work."
- "By reversing the words in the source sentence, the average distance between corresponding words in the source and target language is unchanged.  However, the first few words in the source language are now very close to the first few words in the target language, so the problem’s minimal time lag is greatly reduced.  Thus, backpropagation has an easier time “ establishing communication” between the source sentence and the target sentence, which in turn results in substantially improved overall performance.
- "LSTMs trained on reversed source sentences did much better on long sentences than LSTMs  trained on the raw source sentences (see sec. 3.7), which suggests that reversing the input sentencesresults in LSTMs with better memory utilization."
- "Most importantly, we demonstrated that a simple, straightforward and a relatively unoptimized approach can outperform an SMT system, so further work will likely lead to even greater translation accuracies. These results suggest that our approach will likely do well on other challenging sequence to sequence problems."


### Recurrent continuous translation models.

Sutskever about this: "(...) were the first to map the entire input sentence to vector"
Sutskever about this: "Our work is closely related to Kalchbrenner and Blunsom [18], who were the first to map the input sentence into a vector and then back to a sentence, although they map sentences to vectors using convolutional neural networks, which lose the ordering of the words"



### Learning phrase representations using RNN encoder-decoder for statistical machine translation
Sutskever about this: "was used only for rescoring hypotheses produced by a phrase-based system"
Sutskever about this: "Used an LSTM-like RNN architecture to map sentences into vectors and back, although their
primary focus was on integrating their neural network into an SMT system."

### Overcoming  the curse of sentence length for neural machine translation using automatic segmentation.

Sutskever says about their observations compared to the observations in the paper: "Surprisingly, the LSTM did not suffer on very long sentences, despite the recent experience of other researchers with related architecture"

Sutskever about this:  "Likewise, Pouget-Abadie et al. [26] attempted to address the memory problem of Cho et al. [5] by translating pieces of the source sentence in way that produces smooth translations, which is similar to a phrase-based approach. We suspect that they could achieve similar improvements by simply training their networks on reversed source sentences."


### Generating sequences with recurrent neural networks

Sutskever about this : introduced a novel differentiable attention mechanism that allows neural networks to focus on different parts of their input


### Neural machine translation by jointly learning to align and translate.

Sutskever about this: "an elegant variant of this idea" {Meaning attention mechanism} "was successfully applied to machine translation"
Sutskever about this: "Bahdanau et al. [2] also attempted direct translations with a neural network that used an attention mechanism to overcome the poor performance on long sentences experienced by Cho et al. [5] and achieved encouraging results"

### Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks.

Sutskever about this: " The Connectionist Sequence Classification is another popular technique for mapping sequences to sequences with neural networks, but it assumes a monotonic alignment between the inputs and the outputs"


### Recurrent neural network based language model.

Sutskever says: "(...) the simplest and most effective way  of applying an  RNN-Language Model (..) to an MT task is by rescoring the n-best lists of a strong MT baseline, which reliably improves translation quality."

### Joint language and translation modeling with recurrent neural networks

Sutskever says: "More recently, researchers have begun to look into ways of in
cluding information about the source
language into the NNLM. Examples of this work include Auli et
al. [1], who combine an NNLM
with a topic model of the input sentence, which improves resc
oring performance"


### Fast and robust neural network joint models for statistical machine translation

Sutskever says: "(...) followed a similar approach, " {Meaning similar to the paper "Joint language and translation modeling with recurrent neural networks"} "but they incorporated their NNLM into the decoder of an MT system
and used the decoder’s alignment information to provide the
NNLM with the most useful words in
the input sentence.  Their approach was highly successful an
d it achieved large improvements over
their baseline."


### Multilingual distributed representations without word alignment.


Sutskever about this: "End-to-end training is also the focus of Hermann et al. [12], whose model represents the inputs and outputs by feedforward networks, and map them to similar points in space. However, their approach cannot generate translations directly: to get a translation, they need to do a look up for closest vector in the pre-computed database of sentences, or to rescore a sentence."
