
### Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups (2012) (IEEE Signal Processing Magazine ) Hinton et al. (Toronto, IBM, Microsoft, Google)
- Previous models use HMMs and Gaussian Mixture Models
- Deep Networks for acustic modeling outperform Gaussian Mixture Models
- In the previous approach what worked well was concatenation of Mel  Frequency  Cepstral  Coefficients  (MFCCs)  or  Perceptual  L
inear  Predictive  coefficients  (PLPs) with their first and second order temporal differences. 
- This previous approach is heavily engineered and discards lot of things (mostly irrelavant, at least that is what people think)
- GMM HMM first is trained generatively to maximize probablity of generating observed data
- GMM HMM can then be discriminatively fine-tuned (Don't go into details on that)
- Can be futher augmented using features learned by neural network
- Speech signal lays on a low dimensional manifold, there have to be models that are better suited than GMM HMM for this problem
- Because of more computational power and better train algorithms many reasearch groups have shown that Neural Networks work better than GMMs on acustic modeling
- In the first phase they start with one layer and train with generative objective and add more layers
- In the second phase they initialize DNN with learned parameters and train to predict GMM HMM states.
- Experiments on the TIMIT database
- They use sigmoid activations, SGD with momentum, L2 norm or early stopping 
- Overfitting can be a severe problem, they propose to use generative training, they finetune later during discriminative learning
- Claim that generative pretraining reduces overfitting.
- Two ways to generate data, directed model (generate latent, then generate data based on latent, separate parameters to define latent model and conditional output model) and undirected model (ties together latent and output with one set of parameter and uses energy minimization as an objective
- They provide an algorithm to train RBM
- RBM similar to MRF (but it has bipartile connectivity graph, does not share weights between units, subset of variables are unobserved)
- Sample from the model - Gibbs Sampling, Contrastive Divergence (TODO: Look it up)
- To model real value data they use Gaussian-Bernoulli RBM (GRBM)
- Stacked RBMs -> Deep Belief Network
- They initialize DNN from DBN
- TIMIT is a small dataset, and lot of benchmars are already tested on it, good starting point
- Compared to GMMs, DNNs work well with correlated features as inputs. DNNs performed better with the data that was not suited for GMMs compared to the orignal GMMs data.
- They derive another train objective from Maximum Mutual Information. It takes into acount sequence of inputs and outputs sequence as well
- They use some hand engineered initialize/train procedure to gain 5% boosts in performance
- They list a set of differences between DBN and GMMs. DBN - Product of experts, GMM - Sum of experts. Different nature of nonlinearity, DNNs can exploit multiple frames of input cooeficients while GMMs require independent features. SGD vs EM.
- For large dataset they replace monophone HMM with triphone HMM
- They test on Bing-Voice-Search speech recognition task, Switchboard speech recognition task, Google Voice Input speech recognition task, YouTube speech recognition task, English-Broadcast-News speech recognition task
- Good results of DNN HMM are due to: direct modeling of tied triphone states, effective exploration of neighboring frames, strong modeling power of deep networks
- Engineered feature less helpful for big models
- Speed up by setting weights to 8 bits
- Claim that generative pretraining imporoves performance sometime by a big margin
- On larger minibatches instead of SGD one cana use: non linear conjugate gradient, LBFGS, Hessian Free methods
- Big issue of DNNs compared to GMMs -> Poor parallelization

From the paper:
- "Instead of designing feature detectors to be good for discriminating between classes, we can start by designing them to be good at modeling the structure in the input data."
- " DNN-HMMs  consistently outperform  GMM-HMMs  that  are  trained  on  the  same  amount  of  data,  sometimes  by  a  large  margin.  For  some tasks, DNN-HMMs also outperform GMM-HMMs that are trained on much more data."
