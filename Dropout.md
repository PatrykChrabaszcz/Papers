### Improving neural networks by preventing co-adaptation of feature detectors (3 Jul 2012) (Arxiv) (University of Toronto) Hinton et al. 
- Set of possible weights that perform well on the train data is huge, but not many of them will perform well on the test data.
- Can be seen as a low cost method for building an ensemble of networks that share some of the weights.
- Instead of L2 normalization they put an upper bound L2 norm constraint of the incoming weight wector for each hidden unit (unit == layer ?).
- They fix dropout probability to 0.5 and propose it as a good default value.
- Went from 160 test errors to 110 errors on MNIST using dropout 0.5 on hidden activations and 0.2 on the input.


From the paper:  
- "Using a constraint" {limit on max L2 norm} "rather than a penalty" {they mean L2 regularization penalty} "prevents weights from growing very large no matter how large the proposed weight-update is. This makes it possible to start with a very large learning rate which decays during learning, thus allowing a far more thorough search of the weight-space than methods that start with small weights and use a small learning rate."
- "It is also possible to adapt the individual dropout probability of each hidden or input unit by comparing the average performance on a validation set with the average performance when the unit is present.  This makes the method work slightly better."
- "For datasets in which the required input-output mapping has a number of fairly different regimes, performance can probably be further improved by making the dropout probabilities be a learned function of the input, thus creating a statistically efficient “mixture of experts” in which there are combinatorially many experts, but each parameter gets adapted on a large fraction of the training data."
- "Dropout can be seen as an extreme form of bagging in which each model is trained on a single case and each parameter of the model is very strongly regularized by sharing it with the corresponding parameter in all the other models. This is a much better regularizer than the standard method of shrinking parameters towards zero."
- "One reason why dropout gives major improvements over backpropagation is that it encourages each individual hidden unit to learn a useful feature without relying on specific other hidden units to correct its mistakes."


### Dropout: A simple way to prevent neural networks from overfitting (Nov 2013) (JMLR 2014) (University of Toronto) Srivastava et al.
- Seems that optimal probability for dropping hidden neurons is close to 0.5 while for the input is closer to 0.0 than 0.5.
- About "Fast Dropout Trianing": proposed a method for speeding up dropout by marginalizing dropout noise.
- Dropout network works better than plain network but worse than bayesian network (small datasets).
- Clams that dropout helps also in convolutional layers.
- Shows figures with CNN filters that are less noisy and more independent when using dropout.
- For FC networks shows that only few neurons have high activations while most of them stay at 0 .
- Compares weight averaging to MC sampling, given enough samples MC gets a little bit better, but weight averagining is still a good approximation.
- Proposes another dropout type: use Gaussian distribution instead of Bernoulli.
- When using dropout learning rate should be 10-100x bigger or momentum should be 0.99 instead of 0.9.


From the paper:  
- "With unlimited computation, the best way to “regularize” a fixed-sized model is to average the predictions of all possible settings of the parameters, weighting each setting by its posterior probability given the training data."
- "One particular form of regularization was found to be especially useful for dropout constraining the norm of the incoming weight vector at each hidden unit to be upper bounded by a fixed constant c" {typical values for c 3-4} ". (...). This is also called max-norm regularization since it implies that the maximum value that the norm of any weight can take is c."
- "In dropout, each model is  weighted equally, whereas in a Bayesian neural network each model is weighted taking into account the prior and how well the model fits the data, which is the more correct approach."
- "We found that as a side-effect of doing  dropout, the activations of the  hidden  units become sparse, even when no sparsity inducing regularizers are present."
- "As the size of the data set is increased, the gain from doing dropout increases up to a point and then declines." {MNIST experiment}
- "It can be seen that around k=50, the Monte-Carlo method becomes as good as the approximate method. Thereafter, the Monte-Carlo method is slightly better than the approximate method but well within one standard deviation of it. This suggests that the weight scaling method is a fairly good approximation of the true model average."
- "One of the drawbacks of dropout is that it increases training time. A dropout network typically takes 2-3 times longer to train than a standard neural network of the same architecture."


### Dropout improves recurrent neural networks for handwriting recognition (5 Nov 2013) (ICFHR 2014) (A2iA) Pham et al.
- Authors claim it is a first work that applies dropout to RNNs.
- Architecture used is a hybrid of CNN and RNN.
- For CNNs (low amount of weights) dropout samples more models than dropconnect.
- Claims that in some previous work CNN dropout had to be smaller than 0.5, otherwise performance decreases.
- Claims up to 20% improved performance when dropout applied only to the last LSTM layer.
- Claims that dropout acts as L2 regularization but this hyperparameter is easier to tune.


From the paper:  
- "In  our  approach,  dropout  is  carefully  used  in  the network so that it does not affect the recurrent connections."
- "(...) dropout is applied only to feed-forward connections and not to recurrent connections."
- "In our experiments, however, we find out that ReLU can not give good performance in LSTM cells, hence we keep tanh for the LSTM cells and sigmoid for the gates."
- "We found that dropout at 3 LSTM layers is generally helpful, however the training time is significantly longer both in term of the number of epochs before convergence and the CPU time for each epoch."
- "(...) hypothesis that dropout encourages the units to emit stronger activations. Since some units were randomly dropped during training, stronger activations might make the units more independently helpful, given the complex contexts of other hidden activations."
- "The word recognition networks with dropout at the topmost layer significantly reduces the CER and WER by 10-20%, and the performance can be further improved by 30-40% if dropout is applied at multiple LSTM layers."
- "Extensive experiments also provide evidence that dropout behaves similarly to weight decay, but the dropout hyper-parameter is much easier to tune than those of weight decay.


Others about it:
- Gal about this paper: "They conclude that dropout in recurrent layers disrupts the RNN’s ability to model sequences, and that dropout should be applied to feed-forward connections and not to recurrent connections."


### Understanding dropout
- Some theory behind dropout.
TODO


### On fast dropout and its applicability to recurrent networks

Others about it:
Gal about this paper: "They reason that with dropout, the RNN’s dynamics change dramatically, and that dropout should be applied to the non-dynamic parts of the model – connections feeding from the hidden layer to the output layer"


### Regularization of Neural Networks using DropConnect
TODO

### Where to Apply Dropout in Recurrent Neural Networks for Handwriting Recognition? (ICDAR 2015) Bluche et al.
- Same people as "Dropout improves rnn for handwriting recognition".
- Investigates bidirectional LSTM. 
- Checks three possible ways to add dropout to the model (input, hidden recurrent, hidden forward).


From the paper:
- "In this paper, we show that further improvement can be achieved by implementing dropout differently, more specifically by applying it at better positions relative to the LSTM units."
- "A false rumour stated that dropout in combination  with parameterized convolutions should not work well, but some attempts demonstrated that it actually is a source of improvement."
- "Moreover, adding dropout before, inside and after the LSTM at the same time is not as good as choosing the right position."
- "(...) among all relative positions to the LSTM, when dropout is applied to every LSTM, placing it after was the worst choice in all six configurations."
- "It seems like dropout is best close to the inputs and outputs of the network."

Others about this paper:
- Gal: "They provide mixed results, not showing significant improvement on existing technique."


### A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (16 Dec 2015) (NIPS 2016)
- Dropout as an approximate Bayesian Inference.
- Dropout only on inputs and outputs still leads to overfitting.
- Proposes variational RNN that uses the same dropout at each timestep.
- Improve Zaremba's SOTA results on Penn Treebank.
- Show that MC dropout with 1000 samples is better than weights averaging.
- Also proposes to use dropout on embeddings for word models.
- Dropout makes it possible to use bigger networks.
- Ensembling Variational RNNs futher improves performance.

From the paper:
- "(...) dropout can be interpreted as a variational approximation to the posterior of a Bayesian neural network (NN)."
- "Empirical results have led many to believe that noise added to recurrent layers (connections between RNN units) will be amplified for long sequences, and drown the signal. Consequently, existing research has concluded that the technique should be used with the inputs and outputs of the RNN alone."
- "In the new dropout variant, we repeat the same dropout mask at each time step for both inputs, outputs, and recurrent layers (drop the same network units at each time step). This is in contrast to the existingad hoc techniques where different dropout masks are sampled at each time step for the inputs and outputs alone (no dropout is used with the recurrent connections since the use of different masks with these connections leads to deteriorated performance."
- "(...) we show that it is possible to derive a variational inference based variant of dropout which successfully regularises such parameters" {meaning recurrent} ", by grounding our approach in recent theoretical research."
- "Implementing our approximate inference is identical to implementing dropout in RNNs with the same network units dropped at each time step , randomly dropping inputs, outputs, and recurrent connections. This is in contrast to existing techniques, where different network units would be dropped at different time steps, and no dropout would be applied to the recurrent connections."
- "A common approach for regularisation is to reduce model complexity (necessary with the non-regularised LSTM). With the Variational models however, a significant reduction in perplexity is achieved by using larger models."
- "Yet it seems that with no embedding dropout, a higher dropout probability within the recurrent layers leads to overfitting! This presumably happens because of the large number of parameters in the embedding layer which is not regularised. Regularising the embedding layer with dropout probability pE=0.5 we see that a higher recurrent layer dropout probability indeed leads to increased robustness to overfitting, as expected."
- "(...) we assess the importance of weight decay with our dropout variant. Common practice is to remove weight decay with naive dropout. Our results suggest that weight decay plays an important role with our variant (it corresponds to our prior belief of the distribution over the weights)."

Others about this paper:
- Merity calls it "Variational Dropout"


### Recurrent dropout without memory loss

Others about this paper:
- Merity sais it is reminescent to Zoneout 
- Merity said: "Another approach is to regularize the network through limiting updates to the RNN’s hidden state. (...) authors drop updates to network units, specifically the input gates of the LSTM, in lieu of the units themselve."


### RnnDrop: A Novel Dropout for RNNs in ASR (Dec 2015) (ASRU 2015) Moon et al. (Samsung)
- Claims that previous dropout only on inputs and outputs is worse than "weight noise injection" regularization.
- Dropout mask for a sequence drawn once for all timesteps.
- Different dropout masks for different sequences within minibatch.
- Gal claims it is parallel to their work about variational RNN.
- Used for phoneme  recognition (TIMIT) and speech recognition (Wall Street Journal WSJ).


From the paper:
- "(...) there have been some attempts to apply the dropout, a popular regularization scheme for the feed forward neural networks, to RNNs, but they do not perform as well as other regularization scheme such as weight noise injection."
- "(...) rnnDrop, which drops out hidden nodes instead of some connections. That is, rnnDrop drops both the non-recurrent and recurrent connectionsthat are connected to the dropped nodes. More importantly,the dropout mask is randomly selected for each input sequenceand fixed throughout the sequence. "
- "Although DBLSTM networks trained with rnnDrop showed superior performances, it took much longer to train them."

Others about this paper:
- Gal: "They randomly drop elements in the LSTM’s internal cell ct and use the same mask at every time step. This is the closest to our proposed approach (although fundamentally different to the approach we suggest)."
- Gal when reproducing results from this work: "Adding our embedding dropout, the model performs much better, but still underperforms compared to applying dropout on the inputs and outputs alone."


### Recurrent dropout without memory loss (Mar 2016) (Arxiv) Semenitua 
TODO
