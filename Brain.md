### Hierarchical internal representation of spectral features in deep convolutional networks trained for EEG decoding (Nov 2017) (BCI 2018) Kay et al. (University of Freiburg)


### Deep learning with convolutional neural networks for EEG decoding and visualization

### Deep learning with convolutional neural networks for decoding and visualization of EEG pathology (Aug 2017) (Arxiv) Schirrmeister et al. (University of Freiburg)

- Apply Convnets to Temple University Hospital EEG Abnormal Corpus dataset.
- Use two basic architectures: shallow and deep.
- Get 85% accuracy compared to the previous 79%.
- Automated search found some surprising architectures, for example networks with max pooling as an only non-linearity.
- Downsamples the data to 100Hz and trains on sequences of length 600 (6 seconds). 
- Removes 1st minute of each recording (contains artifacts).
- Clips the data at 800uV.
- Uses crop training. 
- Provides github code https://github.com/robintibor/auto-eeg-diagnosis-example
- Evaluates the effect of using shorter recordings (ony the first minute, only the first 2 minutes, ...).
- Use SMAC to optimize the architecture (filter lengths, strides and types of nonlinearities).
- Each configuration in SMAC was trained for 3,5h
- To visualize predictions they simply perturbe input signals with some frequencies and compute correlation between perturbation and prediction.
- For both normal and abnormal classes they checked which words appeared more frequently in the wrongly predicted class.
- They have a nice visualization of confusion matrices.
- Better than previous baseline even when taking 1 prediction from only 6 seconds per recording. 
- However baseline was evaluated on an old version of the dataset 
- The best accuracies obtained when training on long recordings but testing only on the first minute.
- "small amount", "age", "sleep" appeared frequently in a wrongly classified cases. 


From the paper:
- "Visualizations of the ConvNet decoding behavior showed that they used spectral power changes in the delta (0-4 Hz) and theta (4-8 Hz) frequency range, possibly alongside other features"
- "Baseline results on this dataset have already been reported by TUH using a convolutional neural network (ConvNet) with multiple fully connected layers that uses precomputed EEG bandpower-based features as input and reached 78.8% accuracy"
- "(...) our shallow ConvNet is specifically tailored to decode band-power features."
- "(...) the first minute of the recordings was always excluded because it appeared to be more prone to artifact contamination than the later time windows."
- "This implies that predictions with high accuracies can be made from just 6 seconds of EEG data."
- "Note that the baseline was evaluated on an older version of the Corpus that has since been corrected to not contain the same patient in training and test recordings among other things."
- "Noticeably, as expected, accuracies slightly decreased with increasing recording time." {They mean test recordings during evaluation} " However, the decrease is below 0.5% and thus should be interpreted cautiously:"
