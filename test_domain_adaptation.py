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
    "--file_name", type=str, default='benchmark', metavar="F", help="file name (default: benchmark)"
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
data_dir = WindowsPath("c:/Users/Wissam/Nextcloud/Documents/MVA/owkin_challenge/")

# load the training and testing data sets
train_features_dir = data_dir / "train_input" / "moco_features"
test_features_dir = data_dir / "test_input" / "moco_features"
df_train = pd.read_csv(data_dir / "supplementary_data" / "train_metadata.csv")
df_test = pd.read_csv(data_dir  / "supplementary_data" / "test_metadata.csv")
# concatenate y_train and df_train
y_train = pd.read_csv(data_dir  / "train_output.csv")
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


X_test_mean = []
centers_test = []

# load the data from `df_test` (~ 1 minute)
for sample, center in tqdm(df_test[["Sample ID", "Center ID"]].values):
    _features = np.load(test_features_dir / sample)
    coordinates, features = _features[:, :3], _features[:, 3:]
    X_test_mean.append(np.mean(features, axis=0))
    centers_test.append(center)

X_test_mean = np.array(X_test_mean)
centers_test = np.array(centers_test)

X_mean = np.concatenate([X_train_mean, X_test_mean])
centers = np.concatenate([centers_train, centers_test])

preprocessing = {}
for center in np.unique(centers):
    mean = np.mean(X_mean[centers==center], axis=0)
    std = np.std(X_mean[centers==center], axis=0)
    preprocessing[center] = {'mean': mean, 'std': std}

def build_dataloaders():
    X_train = []
    y_train = []
    centers_train = []

    for sample, label, center, patient in df_train[
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


    X_test = []
    samples_test = []
    centers_test = []

    for sample, center, patient in df_test[
        ["Sample ID", "Center ID", "Patient ID"]
    ].values:
        # load the coordinates and features (1000, 3+2048)
        _features = np.load(test_features_dir / sample)
        # get coordinates (zoom level, tile x-coord on the slide, tile y-coord on the slide)
        # and the MoCo V2 features
        coordinates, features = _features[:, :3], _features[:, 3:]  # Ks
        features = (features - preprocessing[center]['mean']) / preprocessing[center]['std']
        # NO slide averaging, X_train will be a 3d tensor
        X_test.append(features)
        samples_test.append(int(sample[3:6]))
        centers_test.append(int(center[2])-1)

    centers_train = np.array(centers_train)
    centers_test = np.array(centers_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    y_train = np.array(y_train)

    samples_test = np.array(samples_test)


    # load the data from `df_test` (~ 1 minute) if we use the real test set
    train_set = data_utils.TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
        torch.from_numpy(centers_train).float(),
    )  # create the dataset.

    test_set = data_utils.TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(samples_test).float(),
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
    return model, optimizer, train_loader, test_loader


def train(epoch,model,optimizer,train_loader,test_loader):
    """
    Train the DeepMIL model
    """
    model.train()
    train_loss = 0.0
    train_error = 0.0
    start_steps = epoch * len(train_loader)
    total_steps = args.epochs * len(train_loader)
    for batch_idx,((data, bag_label,center),(data_val,samples_val,center_val)) in enumerate(zip(train_loader,test_loader)):
        p = float(batch_idx + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-10*p)) - 1
        if args.cuda:
            data, bag_label,center = data.cuda(), bag_label.cuda(),center.cuda()
        data, bag_label, center = Variable(data), Variable(bag_label), Variable(center)
        labels_one_hot = torch.FloatTensor(data.shape[0], 5)
        labels_one_hot.zero_()
        center = center.type(torch.int64)
        labels_one_hot.scatter_(1, center.view(-1, 1), 1)
        if args.cuda:
            data_val, samples_val, center_val = data_val.cuda(), samples_val.cuda(),center_val.cuda()
        data_val, samples_val, center_val = Variable(data_val), Variable(samples_val), Variable(center_val)
        labels_one_hot_val = torch.FloatTensor(data_val.shape[0], 5)
        labels_one_hot_val.zero_()
        center_val = center_val.type(torch.int64)
        labels_one_hot_val.scatter_(1, center_val.view(-1, 1), 1)
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        class_loss, domain_loss_train, _ = model.calculate_objective(data, bag_label, labels_one_hot, constant)
        domain_loss_val = model.calculate_domain_loss(data_val, labels_one_hot_val, constant)
        domain_loss = domain_loss_train+domain_loss_val
        loss = class_loss+args.theta*domain_loss
        loss = loss.mean() #in case batch size!=1 we want the mean of the log-likelihoods and not the tensor of the log-likelihoods of each tile
        train_loss += loss.data
        # backward pass
        loss.backward()
        # step
        optimizer.step()




def test(model,test_loader):
    """
    Test DeepMIL model
    """
    model.eval()

    array_probas = np.array([])
    array_samples = np.array([])

    for batch_idx, (data,sample,center) in enumerate(test_loader):
        if args.cuda:
            data,sample,center = data.cuda(),sample.cuda(),center.cuda()
        data, sample,center = Variable(data),Variable(sample),Variable(center)
        constant=1.
        probas, _, _, _ = model(data,constant)
        probas = probas.cpu().data.numpy()[:,0]
        array_probas = np.concatenate((array_probas,probas))
        array_samples = np.concatenate((array_samples,sample.cpu().data.numpy()))
    df = pd.DataFrame()
    df['sample'] = array_samples
    df['Target'] = array_probas
    df['Sample ID'] = df['sample'].apply(lambda nb:"ID_"+str(int(nb)).zfill(3)+".npy")
    df2 = df.sort_values("Sample ID")[["Sample ID","Target"]]
    return df2


if __name__ == "__main__":
    model, optimizer, train_loader, test_loader = build_dataloaders()
    print("Start Training")
    for epoch in range(1, args.epochs + 1):
        train(epoch,model,optimizer,train_loader, test_loader)
    print("Start Testing")
    benchmark = test(model,test_loader)
    benchmark.to_csv(args.file_name+'.csv',index=False)