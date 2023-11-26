from __future__ import print_function

from pathlib import WindowsPath
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


import random
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


from model import GatedAttentionDomainAdaptation

# Training settings
parser = argparse.ArgumentParser(description="DeepMIL PIK3CA")
parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    metavar="N",
    help="number of epochs to train (default: 20)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0005,
    metavar="LR",
    help="learning rate (default: 0.0005)",
)
parser.add_argument(
    "--reg", type=float, default=10e-5, metavar="R", help="weight decay"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--batch_size", type=int, default=1, metavar="BS", help="batch size (default: 1)"
)
parser.add_argument(
    "--csv_name", type=str, default='', metavar="CSV", help="name of the CSV with losses"
)
parser.add_argument(
    "--theta", type=float, default=1., metavar="theta", help="weight of the domain loss"
)
args = parser.parse_args()
args.cuda =  torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print("\nGPU is ON!")

# Build train and test sets

# load the training and testing data sets
train_features_dir = "train_input" / "moco_features"
test_features_dir = "test_input" / "moco_features"
df_train = pd.read_csv("supplementary_data" / "train_metadata.csv")
# concatenate y_train and df_train
y_train = pd.read_csv("train_output.csv")
df_train = df_train.merge(y_train, on="Sample ID")

X_train_mean = []
y_train = []
centers_train = []
patients_train = []

for sample, label, center, patient in tqdm(
    df_train[["Sample ID", "Target", "Center ID", "Patient ID"]].values
):
    # load the coordinates and features (1000, 3+2048)
    _features = np.load(train_features_dir / sample)
    # get coordinates (zoom level, tile x-coord on the slide, tile y-coord on the slide)
    # and the MoCo V2 features
    coordinates, features = _features[:, :3], _features[:, 3:]  # Ks
    # slide-level averaging
    X_train_mean.append(np.mean(features, axis=0))
    y_train.append(label)
    centers_train.append(center)
    patients_train.append(patient)

# convert to numpy arrays
X_train_mean = np.array(X_train_mean)
y_train = np.array(y_train)
centers_train = np.array(centers_train)
patients_train = np.array(patients_train)




X_mean = X_train_mean
centers = centers_train

preprocessing = {}
for center in np.unique(centers):
    mean = np.mean(X_mean[centers==center], axis=0)
    std = np.std(X_mean[centers==center], axis=0)
    preprocessing[center] = {'mean': mean, 'std': std}

def validation(val_center):
    df_train_ = df_train[df_train['Center ID']!=val_center]
    df_val_ = df_train[df_train['Center ID']==val_center]

    X_train = []
    y_train = []
    centers_train = []
    patients_train = []
    class_weights = []

    for sample, label, center, patient in df_train_[
        ["Sample ID", "Target", "Center ID", "Patient ID"]
    ].values:
        # load the coordinates and features (1000, 3+2048)
        _features = np.load(train_features_dir / sample)
        # get coordinates (zoom level, tile x-coord on the slide, tile y-coord on the slide)
        # and the MoCo V2 features
        coordinates, features = _features[:, :3], _features[:, 3:]  # Ks
        features = (features - preprocessing[center]['mean']) / preprocessing[center]['std']
        # NO slide averaging, X_train will be a 3d tensor
        X_train.append(features)
        y_train.append(label)
        centers_train.append(int(center[2])-1)
        patients_train.append(1/len(df_train_[df_train_['Patient ID']==patient]))
        class_weights.append(1/len(df_train_[df_train_['Target']==label]))

    X_test = []
    y_test = []
    centers_test = []

    for sample, label, center, patient in df_val_[
        ["Sample ID", "Target", "Center ID", "Patient ID"]
    ].values:
        # load the coordinates and features (1000, 3+2048)
        _features = np.load(train_features_dir / sample)
        # get coordinates (zoom level, tile x-coord on the slide, tile y-coord on the slide)
        # and the MoCo V2 features
        coordinates, features = _features[:, :3], _features[:, 3:]  # Ks
        features = (features - preprocessing[center]['mean']) / preprocessing[center]['std']
        # NO slide averaging, X_train will be a 3d tensor
        X_test.append(features)
        y_test.append(label)
        centers_test.append(int(center[2])-1)
            
    """
    print("\n Training DS ")
    print("Training DS contains {} patients".format(len(X_train)))
    print("with labels: " + str([int(i) for i in y_train]))
    print(
        "Ratio of {} negative to {} positive cases.".format(
            len([i for i in y_train if i == 0]),
            len([i for i in y_train if i == 1]),
        )
    )

    # random minority oversampling
    while len([i for i in range(len(y_train)) if y_train[i] == 1]) < len(
        [i for i in range(len(y_train)) if y_train[i] == 0]
    ):
        new_sample_index = random.choice([i for i in range(len(y_train)) if y_train[i] == 1])
        X_train.append(X_train[new_sample_index])
        y_train.append(y_train[new_sample_index])

    print("\nModified Training DS after RANDOM MINORITY OVERSAMPLING")
    print("Training DS contains {} patients".format(len(X_train)))
    print("with labels: " + str([int(i) for i in y_train]))
    print(
        "Ratio of {} negative to {} positive cases.".format(
            len([i for i in y_train if i == 0]),
            len([i for i in y_train if i == 1]),
        )
    )
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    class_weights = np.array(class_weights)

    centers_train = np.array(centers_train)
    centers_test = np.array(centers_test)

    patients_train = np.array(patients_train)

    # load the data from `df_test` (~ 1 minute) if we use the real test set
    train_set = data_utils.TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
        torch.from_numpy(class_weights).float(),
        torch.from_numpy(centers_train).float(),
        torch.from_numpy(patients_train).float(),
    )  # create the dataset.

    test_set = data_utils.TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).float(),
        torch.from_numpy(centers_test).float(),
    )  # create the dataset.

    print("Load Train and Test Set")
    loader_kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}

    train_loader = data_utils.DataLoader(dataset=train_set,batch_size=args.batch_size, shuffle=True, **loader_kwargs)

    test_loader = data_utils.DataLoader(dataset=test_set,batch_size=args.batch_size, shuffle=True, **loader_kwargs)

    print("Init Model")

    model = GatedAttentionDomainAdaptation()
    if args.cuda:
        model.cuda()
    
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg
    )

    """
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr
    )
    """
    return model, optimizer, train_loader, test_loader

def train(epoch,model,optimizer,train_loader,val_loader):
    """
    Train the DeepMIL model
    """
    model.train()
    train_losses = []
    valid_losses = []
    train_loss = 0.0
    train_error = 0.0
    train_array_probas = np.array([])
    train_array_true_labels = np.array([])
    start_steps = epoch * len(train_loader)
    total_steps = args.epochs * len(train_loader)
    for batch_idx,((data, bag_label, class_weight,center,patient_weight),(data_val,bag_label_val,center_val)) in enumerate(zip(train_loader,val_loader)):
        p = float(batch_idx + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-10*p)) - 1
        if args.cuda:
            data, bag_label, class_weight,center,patient_weight = data.cuda(), bag_label.cuda(), class_weight.cuda(),center.cuda(),patient_weight.cuda()
        data, bag_label, class_weight,center,patient_weight = Variable(data), Variable(bag_label), Variable(class_weight),Variable(center), Variable(patient_weight)
        labels_one_hot = torch.FloatTensor(data.shape[0], 5)
        labels_one_hot.zero_()
        center = center.type(torch.int64)
        labels_one_hot.scatter_(1, center.view(-1, 1), 1)
        if args.cuda:
            data_val, bag_label_val, center_val = data_val.cuda(), bag_label_val.cuda(),center_val.cuda()
        data_val, bag_label_val, center_val = Variable(data_val), Variable(bag_label_val), Variable(center_val)
        labels_one_hot_val = torch.FloatTensor(data_val.shape[0], 5)
        labels_one_hot_val.zero_()
        center_val = center_val.type(torch.int64)
        labels_one_hot_val.scatter_(1, center_val.view(-1, 1), 1)
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        class_loss, domain_loss_train, _ = model.calculate_objective(data, bag_label, labels_one_hot, constant)
        class_loss_val, domain_loss_val, _ = model.calculate_objective(data_val, bag_label_val, labels_one_hot_val, constant)
        domain_loss = domain_loss_train+domain_loss_val
        loss = class_loss+args.theta*domain_loss
        loss = loss.mean() #in case batch size!=1 we want the mean of the log-likelihoods and not the tensor of the log-likelihoods of each tile
        train_loss += loss.data
        #weighted_loss = torch.mul(loss, class_weight)
        #loss = weighted_loss
    
        error, predicted_label = model.calculate_classification_error(data, bag_label, constant)
        probas, _, _, _ = model(data, constant)
        probas = probas.cpu().data.numpy()[:,0]

        train_array_probas = np.concatenate((train_array_probas,probas))
        train_array_true_labels = np.concatenate((train_array_true_labels,bag_label.cpu().data.numpy()))
        train_error += error
        # backward pass
        
        loss.backward()
        # step
        optimizer.step()
        train_losses.append(loss.item())
    """
    for batch_idx,(data, bag_label, center) in enumerate(val_loader):
        if args.cuda:
            data, bag_label, center = data.cuda(), bag_label.cuda(),center.cuda()
        data, bag_label, center = Variable(data), Variable(bag_label), Variable(center)
        labels_one_hot = torch.FloatTensor(data.shape[0], 5)
        labels_one_hot.zero_()
        center = center.type(torch.int64)
        labels_one_hot.scatter_(1, center.view(-1, 1), 1)
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        class_loss, domain_loss, _ = model.calculate_objective(data, bag_label, labels_one_hot, constant)
        loss = args.theta*domain_loss
        loss = loss.mean() #in case batch size!=1 we want the mean of the log-likelihoods and not the tensor of the log-likelihoods of each tile
        train_loss += loss.data
        loss.backward()
        optimizer.step()
    """    
    # validate the model
    model.eval()
    array_probas = np.array([])
    array_true_labels = np.array([])
    for data, bag_label, centers in val_loader:
        if args.cuda:
            data, bag_label, centers = data.cuda(), bag_label.cuda(), centers.cuda()
        data, bag_label, centers = Variable(data), Variable(bag_label), Variable(centers)
        labels_one_hot = torch.FloatTensor(data.shape[0], 5)
        labels_one_hot.zero_()
        centers = centers.type(torch.int64)
        labels_one_hot.scatter_(1, centers.view(-1, 1), 1)
        constant=1.
        loss, _, _ = model.calculate_objective(data, bag_label, labels_one_hot, constant)
        loss = loss.mean()
        valid_losses.append(loss.item())
        probas, _, _, _ = model(data,constant)
        probas = probas.cpu().data.numpy()[:,0]
        array_probas = np.concatenate((array_probas,probas))
        array_true_labels = np.concatenate((array_true_labels,bag_label.cpu().data.numpy()))
    train_roc_auc = roc_auc_score(train_array_true_labels,train_array_probas)  
    valid_roc_auc = roc_auc_score(array_true_labels,array_probas)
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    return train_loss,valid_loss, train_roc_auc, valid_roc_auc


def test(model,test_loader):
    """
    Test DeepMIL model
    """
    model.eval()
    test_loss = 0.0
    test_error = 0.0
    array_probas = np.array([])
    array_true_labels = np.array([])

    for batch_idx, (data, bag_label, centers) in enumerate(test_loader):
        if args.cuda:
            data, bag_label, centers = data.cuda(), bag_label.cuda(), centers.cuda()
        data, bag_label, centers = Variable(data), Variable(bag_label), Variable(centers)
        constant=1.
        labels_one_hot = torch.FloatTensor(data.shape[0], 5)
        labels_one_hot.zero_()
        centers = centers.type(torch.int64)
        labels_one_hot.scatter_(1, centers.view(-1, 1), 1)
        loss, _, _ = model.calculate_objective(data, bag_label, labels_one_hot, constant)
        loss = loss.mean()
        test_loss += loss.data
        error, predicted_label = model.calculate_classification_error(data, bag_label, constant)
        probas, _, _, _ = model(data,constant)
        probas = probas.cpu().data.numpy()[:,0]
        array_probas = np.concatenate((array_probas,probas))
        array_true_labels = np.concatenate((array_true_labels,bag_label.cpu().data.numpy()))

        test_error += error

        if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
            bag_level = (
                bag_label.cpu().data.numpy(),
                (predicted_label.cpu().data.numpy()[:,0]).astype(int)
            )
            #print(f"\nTrue Bag Label, Predicted Bag Label:{bag_level}")
            #print(f"\nPredicted probabilities:{probas}")

    test_error /= len(test_loader)
    test_loss /= len(test_loader)
    
    """
    print(
        "\nTest Set, Loss: {:.4f}, Test error: {:.4f}".format(
            test_loss.cpu().numpy(), test_error
        )
    )
    print(f"\nLabels:{array_true_labels}")
    print(f"\nPredictions:{array_probas}")
    """

    print(f"\nROC AUC:{roc_auc_score(array_true_labels,array_probas)}")
    
    return roc_auc_score(array_true_labels,array_probas)


if __name__ == "__main__":
    roc_auc_scores = []
    df = pd.DataFrame()
    for val_center in ['C_1','C_2','C_5']:
        model, optimizer, train_loader, val_loader = validation(val_center)
        print("Start Training")

        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = [] 
        list_train_roc_auc = []
        list_valid_roc_auc = []
        for epoch in range(1, args.epochs + 1):
            train_loss,valid_loss,train_roc_auc,valid_roc_auc = train(epoch,model,optimizer,train_loader,val_loader)
            """
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            """
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            list_train_roc_auc.append(train_roc_auc)
            list_valid_roc_auc.append(valid_roc_auc)
        print("Start Testing")
        current_roc_auc_score = test(model,val_loader)
        roc_auc_scores.append(current_roc_auc_score)
        df['train_loss_'+val_center] = pd.Series(avg_train_losses)
        df['val_loss_'+val_center] = pd.Series(avg_valid_losses)
        df['train_roc_auc_'+val_center] = pd.Series(list_train_roc_auc)
        df['val_roc_auc_'+val_center] = pd.Series(list_valid_roc_auc)
    df.to_csv(args.csv_name+'.csv')
    print(f"Mean ROC AUC score:{np.mean(roc_auc_scores)}")
