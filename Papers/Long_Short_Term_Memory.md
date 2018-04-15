# Long Short-term Memory
###


### Authors
Sepp Hochreiter, Jurgen Schmidhuber


### Abstract 
Learning to store information over extended time intervals by recurrent backpropagation takes a very long time, mostly because of insufficient, decaying error backflow. We briefly review Hochreiter's (1991) analysis of this problem, then address it by introducing a novel, efficient, gradient-based method called long short-term memory (LSTM). Truncating the gradient where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000 discrete-time steps by enforcing constant error flow through constant error carousels within special units. Multiplicative gate units learn to open and close access to the constant error flow. LSTM is local in space and time; its computational complexity per time step and weight is O(1). Our experiments with artificial data involve local, distributed, real-valued, and noisy pattern representations. In comparisons with real-time recurrent learning, back propagation through time, recurrent cascade correlation, Elman nets, and neural sequence chunking, LSTM leads to many more successful runs, and learns much faster. LSTM also solves complex, artificial long-time-lag tasks that have never been solved by previous recurrent network algorithms. 

### Summary
- LSTM tries to address the problem of "decaying error back flow" (vanishing gradients)
- Existing methods for RNNs do not work better than feed forward networks with fixed time horizon
- With BPTT and RTRL error signals blow up (explode) or vanish
- LSTM can work even witht 1000 time steps
- They reference some different gradienet based variants but say that they all suffer from the same problems as BPTT and RTRL
- They reference a lot of work on previous approaches
- LSTM uses multiplicative units to protect error flow from unwanted perturbations
- They show that weight guessing (Random Search) works better than some approaches proposed before. It meas that benchmark problems were too simple.
- BPTT is sensitive to recent distractions, hard to learn long term relations

### References

Back Propagation Through Time BPTT Wiliams and Zipser 1992, Werbos 1988
Real Time Recurrent Learning RTRL Robinson and Fallside 1987
