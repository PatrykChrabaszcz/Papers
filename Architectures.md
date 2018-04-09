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




