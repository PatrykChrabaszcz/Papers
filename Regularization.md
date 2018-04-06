### Ivestigation of recurrent neural network architectures and learning methods for spoken language understanding.

Others about it:
- Bluche claims that in this paper they demonstrated: "(...) that dropout can significantly increase the generalization capacity in architectures with recurrent layers."


### Regularization and nonlinearities for neural language models: when are they needed?
Others about it:
- Gal claims that here authors reason that noise added in the recurrent connections of an RNN leads to model instabilities, and they only add dropout to the decoding part.



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

# Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks (Feb 2016) (NIPS 2016) Salimans et al. (OpenAI)
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


