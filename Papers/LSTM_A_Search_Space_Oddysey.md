### LSTM_a_Search_Space_Oddysey

Klaus Greff, Rupesh Kumar Srivastava, Jan Koutník, Bas R. Steunebrink, Jürgen Schmidhuber

### Abstract 
Several  variants  of  the  Long  Short-Term  Memory (LSTM)  architecture  for  recurrent  neural  networks  have  been proposed   since   its   inception   in   1995.   In   recent   years,   these networks  have  become  the  state-of-the-art  models  for  a  variety of machine learning problems. This has led to a renewed interest in  understanding  the  role  and  utility  of  various  computational components of typical LSTM variants. In this paper, we present the  first  large-scale  analysis  of  eight  LSTM  variants  on  three representative tasks: speech recognition, handwriting recognition, and  polyphonic  music  modeling.  The  hyperparameters  of  all LSTM  variants  for  each  task  were  optimized  separately  using random  search,  and  their  importance  was  assessed  using  the powerful fANOVA framework. In total, we summarize the results of  5400  experimental  runs  ( ≈ 15 years  of  CPU  time),  which makes  our  study  the  largest  of  its  kind  on  LSTM  networks. Our  results  show  that  none  of  the  variants  can  improve  upon the  standard  LSTM  architecture  significantly,  and  demonstrate the  forget  gate  and  the  output  activation  function  to  be  its most  critical  components.  We  further  observe  that  the  studied hyperparameters are virtually independent and derive guidelines for  their  efficient  adjustment.


- First  large-scale  analysis  of  eight  LSTM  variants, each differs from vanilla LSTM by a small change
- Three representative tasks: speech recognition, handwriting recognition, and  polyphonic  music  modeling.
- Optimize parameters using random search
- 5400 experimental runs
- Asses the importance of parameters with fanova
- None of the variants can improve significantly over vanilla LSTM
- Forget  gate  and  the  Output  activation  function  are the most  critical  components
- Hyperparameters they study are independent
- In initial LSTM paper only the cell state gradient was back-propageted, recurrent gradient (state h) was truncated
- In some previous work that introduced peephole connections output activation was ommited as there was no evidence that it improves performance
- Not designed for SOTA, rather for fair comparison and simple experiments
- Tune Hyperparameters individually for each LSTM variant
- They use the same validation set for early stopping and for validation performance optimization
- Describe features used for raw audio data (LSTM does not process raw signals)
- Train with SGD with Nesterov style momentum
- Random search improves hyperparameter importance analysis
- 200 trials for each random search (lstm variant x datasets)
- Gradient clipping of -1, 1 hurts performance, but this might be the result of the clip range (-1, 1)
- Welch’s t-test to measure if  mean test set performance of each variant was significantly different from that of the baseline
- Removing output activation or forget gate hurts performance significantly
- Without output activation it could grow too large 
- Input/Forget gates coupling did not have significant effect, same for removing peepholes
- Removing the input gate (NIG), the output gate (NOG), and the input activation function (NIAF) led to a significant reduction  in  performance  on  speech  and  handwriting  recognition. No significant effect on music modeling
- Good examples of using fANOVA
- Learning rate has high range of good values -> Advice: when trying to find good values for the model, simply start with high value and divide it by 10 until no improvement is seen 
- Input noise increases triaining time and hurts performance 
- No variants are significantly better, but some variants are simpler (no peephole connections, tied forget and input gates) and have less parameters
- The forget gate and the output activation are importent. Output  activation  function  is  needed  to  prevent  the
unbounded  cell  state  to  propagate  through  the  network  and destabilize learning.
- Learning rate is the most important hyperparameter

### Hyperparameters
- hidden size
- learning rate
- momentum
- input noise standard deviation



### Examined Variants:
- No forget gate
- No input gate
- No output gate
- No input activation
- No output activation
- Coupled input and forget gate f=1-i
- No peepholes
- Full gate recurrence (9 additional recurrent weight matrices (between all gates))





### References
Paper that introduced forget gate:
Learning  to  forget:  Continual  prediction  with  LSTM.

For precise timing peepholes were introduced:
Recurrent nets that time and count

Full BPTT:
Framewise phoneme classification with bidirectional LSTM and other neural network architectures
- One of the advantages of full bptt is that finite differences can be used to check implementation

Evolving LSTM structures:
- Evolving  memory  cell  structures  for sequence learning

Downprojection of LSTM output for reducing the number of parameters:
Long  short-term  memory  recurrent  neural  network  architectures  for  large  scale  acoustic  modeling

Trainable scaling parameter for the slope of the gate activation functions
Fast and robust training of recurrent neural networks for offline  handwriting  recognition. 

Dynamic Context Memory, adding recurrent connections between  the  gates  of  a  single  block
Enhancing Recurrent Neural Networks for Gradient-Based Sequence Learning
