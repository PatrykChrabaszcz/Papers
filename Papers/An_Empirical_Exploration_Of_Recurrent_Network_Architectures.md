## An Empirical Exploration of Recurrent Network Architectures



- Try to determine whether LSTM are optimial or some better architectures exist
- Evaluated 10 000 different architectures (Evolutionary architecture search), 230 000 hyperparameter configurations
- Found architecture that outperforms GRU and LSTM on some but not all tasks
- Adding forget bias of 1 to the LSTM brings its performance closer to the GRU
- GRUs outperform LSTMs on most of the tasks, however when droput was used LSTMs were better
- They measure the performance of many LSTM components
- Input gate is important, output gate is not important, forget gate is extremally significant on all problems except language modeling
- Vanishing gradients make it such that gradient component in the direction that emphasizes short term dependencies is much larger than in the direction that emphasizes longer ones.
- Different practitioners use different LSTM variants
- They found out that GRU outperforms LSTM on nearly all tasks except language modeling. LSTM nearly matches GRU if forget bias initialized to 1.
- Architecture they found outperformed LSTM on all tasks, and outperformed by a small margin GRU. Different architectures 
- Transfer good hyperparameter settings from parent into offspring, add some random hyperparameters as well.
- Evaluate on 3 problems, Arithmetics, XML Modeling and Penn Tree-Bank
- Add 4th Music dataset to measure generalization properties of architecture
- Train with sequences of length 32 and batch size 20 (use truncated BPTT if neccessary)
- Fixed learninig rate that will be decayed when no improvement over validation set detected
- For each dataset they have different learinign rate decay schedules
- They observe only slight performance gains for larger models
- They present three best architectures, similar to GRU but outperform it 
- Some people critisized that evolved architectures are too similar to parent GRU and LSTM
- If there are better than LSTM/GRU architectures then they are not trivial to find

### Search procedure
- They store results for evaluated hyperparameters inside cache (What about noise in the evaluation?)

- Keep a list of 100 best architectures, performance of each is measured as preformance with best hyperparameter settings (encountered so far)
- Initialize this list with LSTM and GRU architectures (At the beginning they train those 2 architectures with all possible hyperparameter values)
One of the following steps is performed in each algorithm iteration:
a)  - Select random architecture and evaluate on 20 random hyperparameters (drawn from prior distribution), evaluate for each of the tasks
    - Assign score as relative improvement over the best GRU architecture in this task. Keep mimimum over tasks -> Forces architectures to work well on all tasks
b)  - Chose architecture from the list of 100 best architectures and mutate it. 
    - After mutation check on toy problem if it is able to get 95% performance 
    - If yes then run on the first task, and if there it gets at least 90% of the best GRU performance (20 random hyperparameters) then run it on all tasks.
    - If it gets good performance save it into the list of 100 best architectures.

### Mutation
- They treat cell as a graph and different layers as nodes.
- Draw random probability of mutation [0-100%], and then apply random mutation to each node with this probability.
- If node is activation function then replace it with other activation function from predefined set
- If node is element-wise operator replace it with other randomly choosen element-wise operator
- Insert random activation between the node and parent
- Remove the node
- Replace the node with randomly choosen ancestor (Shared weights ?)
- Replace the node with sum,product or difference of ancestors of the current node and some other random node.
- Reject incompatible architectures (repeat sampling)

### Hyperparameters
- Initialization scale
- Learning rate
- Max gradient norm
- Number of layers
- Number of parameters in the model


### Findings
- GRU outperformed LSTM on all tasks except language modeling
- Evolved architecture matched performance of GRU on language modeling and outperformed it on all other tasks
- When dropout was allowed LSTM outperform all architectures on PTB
- LSTM with large forget bias outperformed LSTM and GRU on almost all tasks
- Forget gate turns out to be the greatest importance on some datasets (relatively unimportant for PTB)
- Second most significant gate is input gate
- Output gate seems not to be that important
- Those whole analysis does not hold when dropout is allowed (Then different architectures seem to be better)
- They were unable to find architectures that outperform LSTM and GRU on all tasks


### References

Initialization of forget bias was already proposed by:
Learning to forget, continual prediction with LSTM Gres et al.

Work that found GRUs to be better than LSTMs on various tasks:
Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling 

Architecture Search for RNN has been introduced previously by 
Evolving Memory Cell Structures for Sequence Learning Bayer et al. 
  - Attempt to address the same problem
  - They used much fewer experiments and small models
  - They were able to find architectures that outperform LSTM on some synthetic problems with long dependencies
  
Silmultaneous work that reached similar conclusions regarding importance of differet gates of LSTM:
LSTM a Search Space Oddysey Greff et al. 


