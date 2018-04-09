### Optimizing Performance of Recurrent Neural Networks on GPUs (Apr 2016) Appleyard et al. (NVidia) 
https://arxiv.org/pdf/1604.01946.pdf
- RNNs can be trained for weeks or months
- Current RNN implementations only use basic optimizations for GPU
- Three optimization stages Firstly optimizing a single cell, secondly a single layer, and thirdly the entire network
- Naive way to implement LSTM will use only around 10% of GPU
- Simply merging separate multiplications into single operation doubles the performance
- Processing i * w_i and h * W_h in parallel doubles again the performance
- Fusing all all point wise operations
- It is possible to parallelize input matrix multiplication for different timesteps
- Pre transposing of the weight matrix gives additional improvements
- Over 11x faster over naive implementation when using multiple layers.
- They provide CUDA code https://github.com/parallel-forall/code-samples/blob/master/posts/rnn/LSTM.cu

- "it is possible to reformulate a group of four matrix multiplications into a single matrix multiplication of four times the size"

### Automatic differentiation in PyTorch (NIPS 2017) Paszke et al. (University of Warsaw, Facebook)
