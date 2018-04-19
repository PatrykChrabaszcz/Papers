## LipNet: End-to-End Sentence-level Lipreading
https://arxiv.org/abs/1611.01599
Nov 2016

Deep Mind Oxford


Lipreading is the task of decoding text from the movement of a speaker’s mouth. Traditional approaches separated the problem into two stages: designing or learning visual features, and prediction.  More recent deep lipreading approaches are end-to-end trainable (Wand et al., 2016; Chung & Zisserman, 2016a).  However, existing  work  on  models  trained  end-to-end  perform  only  word  classification, rather than sentence-level sequence prediction.  Studies have shown that human lipreading performance increases for longer words (Easton & Basala, 1982), indicating the importance of features capturing temporal context in an ambiguous communication  channel.   Motivated  by  this  observation,  we  present  LipNet,  a model that maps a variable-length sequence of video frames to text, making use of spatiotemporal convolutions, a recurrent network, and the connectionist temporal classification loss, trained entirely end-to-end.  To the best of our knowledge, LipNet is the first end-to-end sentence-level lipreading model that simultaneously learns spatiotemporal visual features and a sequence model. On the GRID corpus, LipNet achieves 95.2% accuracy in sentence-level, overlapped speaker split task, outperforming experienced human lipreaders and the previous 86.4% word-level state-of-the-art accuracy (Gergen et al., 2016).

- Maps video frames to text
- Previous end-to-end approaches operated on world level, this one operates on sentence level
- Use GRID corpus dataset
- Humans are poor for lip reading, up to 21% +- 11% accuracy for a limited set of 30 words.
- System has to extract spaciotemporal relations (motion and position of lips)
- LipNet is the first end-to-end sentence-level lipreading model
- Uses spatiotemporal convolutional neural network (STCNNs), RNNs and connectionist temporal classification loss (CTC)
- Achieves 95.2 % sentece level word accuracy (Previous 86.4%)
- Generalizes to new speakers with 88.6% accuracy
- Hearing impared people get 52,3% accuracy
- They visualize silency maps
- Erroneous predictions occur within visemes, since context is sometimes insufficient for disambiguation.


