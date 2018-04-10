### Improving predictive  inference under  covariate  shift  by  weighting  the  log-likelihood function
Others about it :
Ioffe references this paper when he mentions "covariate shift"


### Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (Feb 2015) Ioffe et al. (Google)
- Distribution of layer activations changes during training -> hard to train 
- Without BN it is hard to train saturating nonliniearities 
- Some notes about the paper https://gist.github.com/shagunsodhani/4441216a298df0fe6ab0
- Covariate shift -> change in the input distribution to a learning system
- It is well established that networks converge faster if the inputs have been whitened (ie zero mean, unit variances) and are uncorrelated and internal covariate shift leads to just the opposite.
- BN normalizes layers outputs
- With BN you can use larger learning rates and be less careful with initialization.
- They achieve SOTA on ImageNet
- If distribution of layer inputs does not change then network does not need to readjust to compensate for the change in the distribution of x. 
- Prevents optimizer to get stuck in the saturated regime (for sigmoid and tahn activations)
- Claims that it also acts as a regularizer.
- Whitening is expensive since we would have to compute covariance matrices. 
- Use additional parameters to assure that BN can represent identity function
- In their experiments with BN they: increase the learning rate, remove dropout, reduce L2 regularization, accelerate learning rate decay, remove local response normalization, provide more shuffling to the training examples, distort images less.
- It does not decorrelate the activations due to the computationally costly matrix inversion


From the paper:
- "When the input distribution to a learning system changes, it is said to experience covariate shift (Shimodaira, 2000)"
- "If,  however,  we  could ensure that the distribution of nonlinearity inputs remains more stable as the network trains, then the optimizer would be less  likely  to  get stuck in  the  saturated  regime,  and the training would accelerate."
- "We refer to the change in the distributions of internal nodes of a deep network, in the course of training, as Internal Covariate Shift." 
- "by reducing the dependence of gradients on the scale of the parameters or of their initial values. This allows us to use much higher learning rates without the risk of divergence."
- "Furthermore, batch  normalization  regularizes  the  model  and  reduces  the  need  for Dropout (Srivastava et al., 2014).  Finally, Batch Normalization makes it possible to use saturating nonlinearities by preventing the network from getting stuck in the saturated modes."
- "Batch Normalization also makes training more resilient to the parameter scale. Normally, large learning rates may increase the scale of layer parameters, which then amplify the gradient during backpropagation and lead to the model explosion. However, with Batch Normalization, backpropagation through a layer is unaffected by the scale of its parameters"
- "Whereas Dropout (Srivastava et al., 2014) is typically used to reduce overfitting, in a batch-normalized network we found that it can be either removed or reduced in strength."
- "Interestingly, increasing the learning rate further (BN-x30) causes the model to train somewhat slower initially, but allows it to reach a higher final accuracy.

### Recurrent Batch Normalization (Mar 2016) (ICLR 2017) Cooijmans et al. (Université de Montréal)
https://arxiv.org/abs/1603.09025
- They evaluate on sequence classification, language modeling and question answering.
- They claim that batch-normalized LSTM consistently leads to faster convergence and improved generalization.
- Their findings about recurrent BN are counter to the ones from: "Batch normalized RNNs"
- RNN BN requires proper initialization
- Claims that BN LSTM outperforms LSTM
- RNNs inherently hard to train because of vanishing gradients
- They apply batch normalization before nonlinearities (They also normalize Wh and Wx separately)
- They do not apply batch normalization on the state c
- They remove biases when they apply batch norm (those are irrelevant when we subtract the mean anyway)
- Averaging statistics over time for batch normalization degrades performance, keep separate statistics per timestep. 
- For longer sequences during test time we can repeat statistics from the maximum timestep seen during training
- They initialize BN gamma to 0.1. They show that if it is initialized to 1 then derivatives of tahn are lower than 1 and this results in a vanishing gradients problem and it does not train that good. BN beta is initialized to 0 (standard approach)
- For MNIST they initialize LSTM recurrent weight matrices with identity matrices, other matrices are initialized with orthogonal matrices. 
- They add some gaussian noise because at the beginning all pixels in MNIST are black. They say it works better than making data dependet hiddent state.
- Better than LSTM on pMNIST, equal on MNIST
- They use gradient clipping of 1.0
- On Penn TreeBank they use orthogonal weight matrix initialization for all matrices (little bit different than for MNIST)
- For text datasets they train on sequences of length 100 (Penn BankTree), 180 (text-8)
- "drastically improves training" on question answering dataset.
- With attentive reader model they share statistics of BN (between timesteps) for forward connections but not for recurrent
- They had to manipulate padding a little bit for varying length sequences in Attentive Reader model.
- Very good results on QA dataset
From the paper:
- "We propose a reparameterization of LSTM that brings the benefits of batch normalization to recurrent neural networks. Whereas previous works only apply batch normalization to the input-to-hidden transformation of RNNs, we demonstrate that it is both possible and beneficial to batch-normalize the hidden-to-hidden transition, thereby reducing internal covariate shift between time steps."
- "(...) we describe a reparameterization of LSTM (Section 3) that involves batch normalization and demonstrate that it is easier to optimize and generalizes better."
- "(...) we (...) show that proper initialization of the batch normalization parameters is crucial to avoiding vanishing gradient."
- "(...) show (...) that our LSTM reparameterization consistently outperforms the LSTM baseline across tasks, in terms of both time to convergence and performance."
- "(...) we leverage batch normalization in both the input-to-hidden and the hidden-to-hidden transformations."
- "In order to leave the LSTM dynamics intact and preserve the gradient flow through ct, we do not apply batch normalization in the cell update."
- "However, we find that simply averaging statistics over time severely degrades performance."
- "In our formulation, we normalize the recurrent term Wh ht−1 and the input term Wx xt separately. Normalizing these terms individually gives the model better control over the relative contribution of the terms using the γh and γx parameters."
- "Consequently, we recommend using separate statistics for each timestep to preserve information of the initial transient phase in the activations."
- "During training we estimate the statistics across the minibatch, independently for each timestep. At
test time we use estimates obtained by averaging the minibatch estimates over the training set."
- "(...) our  proposed  BN-LSTM  trains  faster  and  generalizes  better  on  a  variety  of  tasks  including language modeling and question-answering.  We have argued that proper initialization of the batch normalization parameters is crucial, and suggest that previous difficulties (Laurent et al., 2016; Amodei et al., 2015) were due in large part to improper initialization. Finally, we have shown our model to apply to complex settings involving variable-length data, bidirectionality and highly nonlinear attention mechanisms."


### Batch normalized recurrent neural networks (Oct 2015) (ICASSP 2016) Laurent et al. (Universit́e de Montŕeal) 
https://arxiv.org/abs/1510.01378

- Claim that BN does not work when applied to recurrent connections.
- Claim that with forward connection it gives faster convergence but does not give better generalization.
- BN more challenging for RNNs but several variants seem to offer benefits. 
- They show how BN can speed up the training.
- They tried to apply BN to h_t before nonlinearity. Did not work, they decided to apply BN on W*x_t.  
- Little bit different than deep speech, BN on input only after multiplication with Wx (Before nonlinearity)
- For some applications they compute BN over 1 training sample (only over batch axis), for other they compute BN over batch and time axis.
- Evaluate on Wall Street Journal (WSJ), Penn TreeBank (PTB)
- Show that Batch Normalization netowrks train faster but also overfit more.
- Some parts of the network were not normalized and because of that they could not use higher leraning rates

From this paper:io
- "In this paper, we show that applying batch normalization to the hidden-to-hidden transitions of our RNNs doesn’t help the training procedure." 
- "(...) batch normalization is only applied after multiplication with the input-to-hidden weight matrices Wx·."


Others about this paper:
- Cooijmans says when he cites this paper : "(...) batch normalization (...) is proven to be difficult to apply in recurrent architectures"
- Coijmans about this paper: " RNNs are deeper in the time direction, and as such batch normalization would be most beneficial when applied horizontally.   However, Laurent et al. (2016) hypothesized that applying batch normalization in this way hurts training because of exploding gradients due to repeated rescaling."
- Coijmans about this: "We suspect that the previous difficulties with recurrent batch normalization reported in Laurent et al. (2016); Amodei et al. (2015) are largely due to improper initialization of the batch normalization parameters, and γ in particular."

### Bridging  the  gaps  between  residual  learning,  recurrent  neural networks and visual cortex

Others about this paper
- Coijmans: "(...) smultaneously investigated batch normalization in recurrent neural networks, albeit only for very short sequences (10 steps)."


### Ivestigation of recurrent neural network architectures and learning methods for spoken language understanding.

Others about it:
- Bluche claims that in this paper they demonstrated: "(...) that dropout can significantly increase the generalization capacity in architectures with recurrent layers."


### Regularization and nonlinearities for neural language models: when are they needed?
Others about it:
- Gal claims that here authors reason that noise added in the recurrent connections of an RNN leads to model instabilities, and they only add dropout to the decoding part.

### Layer normalization (Jul 2016) (Arxiv) Ba et al.(University of Toronto)
- Batch Normalization is dependent on the mini batch size
- BN/LN should be applied before non linearity 
- In RNNs compute normalization statistics separately at each timestep 
- Can be used with batch size 1
- No problems like for recurrent BN when it has to store separate mean statistics for each timestep 
- They discuss how weight/batch/layer normalizations relate to each other and what kind of invariances each brings.
- They discuss Riemannian manifold
- They test LN on 6 tasksk  image-sentence  ranking,  question-answering,  contextual  language  modelling,  generative
modelling,  handwriting sequence generation and MNIST classification
- On Microsoft COCO LN model requires only 60% time compared to the model without LN 
- Train some models for a month
- LN not really suited for CNNs yet

From this paper:
- "However, the effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks."
- "Unlike batch normalization, layer normalization performs exactly the same computation at training and test times"
- "It is also straightforward to apply to recurrent neural networks by computing the normalization statistics separately at each time step"
- "Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks.  Empirically, we show that layer normalization can substantially reduce the training time compared with previously published techniques."
- "We show that layer normalization works well for RNNs and improves both the training time and the generalization performance of several existing RNN models."
- "(...) “covariate shift” problem can be reduced by fixing the mean and the variance of the summed inputs within each layer."
- "Unlike batch normalization, layer normaliztion does not impose any constraint on the size of a mini-batch and it can be used in the pure online regime with batch size 1"
- "But when we apply batch normalization to an RNN in the obvious way, we need to to compute and store separate statistics for
each time step in a sequence"
- "(...) its normalization terms depend only on the summed inputs to a layer at the current time-step.  It also has only one set of gain and bias parameters shared over all time-steps"
- "In a layer normalized RNN, the normalization terms make it invariant to re-scaling all of the summed inputs to a layer, which results in much more stable hidden-to-hidden dynamics"



Others about this paper:
- Cooijmans: "(...) independently developed a variant of batch normalization that is also applicable to recurrent neural networks and delivers similar improvements as our method."

### Recurrent neural network regularization (8 Sep 2014) (ICLR 2015) Zaremba et al. 
- Authors claim that it is the first approach for using dropout in RNNs (LSTM).
- Dropout applied only to the non recurrent connections.
- Applies dropout on the RNN input and output (L+1 dropouts per timestep).
- Dropout regularized model gives a similar performance to an ensemble of 10 non-regularized models. 
- Same approach as "Dropout improves recurrent neural networks for handwriting recognition".
- Applied for Language Modeling and Speech Recognition.


From the paper:
- "Unfortunately, dropout (...) does not work well with RNNs."
- "Bayer et al. (2013) claim that conventional dropout does not work well with RNNs because the recurrence amplifies noise, which in turn hurts learning."
- "Standard dropout perturbs the recurrent connections, which makes it difficult for the LSTM to learn to store information for long periods of time."
- "By not using dropout on the recurrent connections, the LSTM can benefit from dropout regularization without sacrificing its valuable memorization ability."
- "The main idea is to apply the dropout operator only to the non-recurrent connections."

Questions: 
- Is dropout mask different for different timesteps? Probably yes.
	
Others about this paper:
- Gal says: "In comparison, Zaremba’s dropout variant replaces zx with the time-dependent ztx which is sampled anew every time step."
- Subakan uses it in their work on Diagonal RNNs.


### Hybrid speech recognition with deep bidirectional LSTM
- Presents noise injection as referenced by Moon in his dropout paper.

Others about this paper:
- Mono says: "In weight noise injection, zero mean Gaussian noise is added to the weights when computing the gradient."


### Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations


Others about this paper:
Merity: "(...) updates to the hidden state may fail to occur for randomly selected neurons"


### Regularizing and Optimizing LSTM Language Models (Aug 2017) (Arxiv) Merity et al. (Salesforce)
- SOTA on Penm Treebank and WikiText-2.
- Adding neural cache to the model futher improves the performance.
- They use  averaged SGD (ASGD) for training. Their method is called NT-ASGD and auto-tunes T.
- Use random sequence length but compare it with very naive implementation where sequence starting point is not randomized. 
- When using longer sequences they use bigger (Need to look at the code to confirm because it is not clearly stated in the paper) learning rate to make the influence of short and long sequences equal.
- Uses embedding dropout as Gal.
- Uses weight tying.
- Uses Temporal Activation Regularization TAR (also Merity's idea) to penalize the model when it produces large changes in hidden state between timesteps.
- AR penalizes hidden activations that are significantly larger than 0.
- AR and TAR only applied to the final LSTM output.
- They use gradient cliping (0.25).
- For each technique they use they also check the performance when this technique is disabled. The most influencial one is hidden to hidden weight decay.

From the paper:
- "We propose the weight-dropped LSTM which uses DropConnect on hidden-to-hidden weights as a form of recurrent regularization."
- "(...) we introduce NT-ASGD, a variant of the averaged stochastic gradient method, wherein the averaging trigger is determined using a non-monotonic condition as opposed to being tuned by the user."
- "The weight-dropped LSTM applies recurrent regularization through a DropConnect mask on the hidden-to-hidden recurrent weights. Other strategies include the use of randomized-length backpropagation through time (BPTT), embedding dropout, activation regularization (AR), and temporal activation regularization (TAR)."
- "As no modifications are required of the LSTM implementation these regularization strategies are compatible with black box libraries, such as NVIDIA cuDNN, which can be many times faster than naïve LSTM implementations."
- "In the context of word-level language modeling, past work has empirically found that SGD outperforms other methods" {Adam, SGD with Momentum etc.} " in not only the final loss but also in the rate of convergence."
- "We propose a variant of ASGD where T is determined on the fly through a non-monotonic criterion and show that it achieves better training outcomes compared to SGD."
- "As the dropout operation is applied once to the weight matrices, before the forward and backward pass, the impact on training speed is minimal and any standard RNN implementation can be used, including inflexible but highly optimized black box LSTM implementations such as NVIDIA’s cuDNN LSTM."
- "As the same weights are reused over multiple timesteps, the same individual dropped weights remain dropped for the entirety of the forward and backward pass"
- "(...) we use variational dropout for all other dropout operations, specifically using the same dropout mask for all inputs and outputs of the LSTM within a given forward and backward pass. Each example within the mini-batch uses a unique dropout mask, rather than a single dropout mask being used over all examples, ensuring diversity in the elements dropped out."
- "In addition, L2 decay can be used on the individual unit activations and on the difference in outputs of an RNN at different time steps; these strategies labeled as activation regularization (AR) and temporal activation regularization (TAR) respectively."
- "In past work, pointer based attention models have been shown to be highly effective in improving language modeling "

### Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks (Feb 2016) (NIPS 2016) Salimans et al. (OpenAI)
- Divides weight parameter into two parameters: magniture and direction.
- Claims that this  improves the conditioning of the optimization problem.
- Claims that this speeds up convergence of stochastic gradient descent.
- Can be applied to RNNs as well.
- Faster than Batch Normalization.
- Experiments on supervised image recognition, generative modelling, and deep reinforcement learning.
- From the optimization perspective model parametrization might be very important.
- Inspired by Batch Normalization but deterministic .
- Magnitude can be parametrized also in a log scale, but they do not find to to be beneficial (and it trains slower).
- They provide the code in Theano https://github.com/TimSalimans/weight_norm
- This parametrization is more robust to different settings of learning rate.
- Important to properly initialize parameters. They set v from normal distribution with 0.05 stdv and then compute g and b such that for one of the minibatches they get normaly distributed activations in each layer (They give derived equations).
- This initialization is not easily applicable to RNNs, for RNNs they use standard methods.
- They reduce the gradient noise by adding using mean-only batch normalization, they claim that it also gives better test accuracies.


From this paper:
- "(...) a reparameterization of the weight vectors in a neural network that decouples the length of those weight vectors from their direction."
- "This shows that weight normalization accomplishes two things: it scales the weight gradient by g / ||v|| , and it projects the gradient away from the current weight vector. Both effects help to bring the covariance matrix of the gradient closer to identity and benefit optimization."
- "Weight normalization can thus be viewed as a cheaper and less noisy approximation to batch normalization. Although exact equivalence does not usually hold for deeper architectures, we still find that our weight normalization method provides much of the speed-up of full batch normalization."


# Deep speech 2: End-to-end speech recognition in english and mandarin

- Coijmans about this: "We suspect that the previous difficulties with recurrent batch normalization reported in Laurent et al. (2016); Amodei et al. (2015) are largely due to improper initialization of the batch normalization parameters, and γ in particular."
