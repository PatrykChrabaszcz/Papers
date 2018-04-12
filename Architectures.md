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




