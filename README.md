# tomographic_binning

This repository contains Jupyter notebooks used for data analysis for the paper Improved Tomographic Binning of the 3x2pt Lens Sample: Neural Network Classifiers and Optimal Bin Assignments.

nn_class.py contains updates to the neural network classifier defined here: https://github.com/adam-broussard/nnclassifier/tree/master. The updates switch the definition of "goodness of fit" to be whether or not the galaxy was sorted into the correct bin (the Misclassification NNC).

file_generator_unrep.ipynb and n_bin_NNC_files.ipynb sort data into training, validation and application samples for running TPZ and the NNCs.

binning_optimizer_unrep_cosmodc2 and binning_optimizer_unrep_buzzard are the analysis notebooks for CosmoDC2 and Buzzard respectively. They determine the optimal bin edge selection, the optimal retained fraction for the NNC selected samples, and plot the sample selections and resulting binnings for the equal number and optimal bin cases. They also compute SNRs for each binning choice.
