# Medical imaging project
My solution to a data challenge:

*train_input/moco_features: the features of the tiles of the slides of all patients of the training set

*test_input/moco_features: the features of the tiles of the slides of all patients of the test set

*supplementary_data: folder containing the metadata about each tile (zoom, coordinates of the tile within the slide, medical center ID, patient ID, slide ID...)

*train_output.csv: the ground truth labels of the train set (whether the mutation is present or not)

*model.py: the definition of the neural networks used to do the classification task (here, the one that matters is GatedAttentionDomainAdaptation)

*crossval_domain_adaptation.py: define a function that performs 3-fold cross-validation of the model and creates a CSV file with the ROC-AUC and the loss on each fold during training

*crossval_domain_adaptation.ipynb: calls the function defined in crossval_domain_adaptation.py

*domain_adaptation_v2_fc256_gat256_fc128_fc64_fc1_batch8_lr1eminus5_epochs40_tanh_no_dropout.csv: the CSV file with the ROC-AUC and the loss on each fold during training which is the output obtained when running crossval_domain_adaptation.ipynb

*train_val_curves_domain_adaptation.ipynb: notebook with plots of the learning curves of the model during cross-validation

*test_domain_adaptation.py: runs the model on the test dataset
