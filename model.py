import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.nb_features = 2048
        self.L = 256
        self.D = 256

        self.fc1 = nn.Sequential(nn.Linear(self.nb_features, self.L))

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())

        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())

        self.attention_weights = nn.Sequential(nn.Linear(self.D, 1))

        self.fc2 = nn.Sequential(nn.Linear(self.L, 128), nn.Tanh())

        self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.Tanh())

        self.classifier = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        # x is of size N x nb_features
        # N:number of tiles corresponding to the WSI
        # nb_features:number of features per tile
        x = self.fc1(x)  # NxL
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 2)  # KxN
        A = F.softmax(A, dim=2)
        M = torch.matmul(A, x)  # KxL
        M = M.squeeze(1)
        M = self.fc2(M)  # Kx128
        M = self.fc3(M)  # Kx64
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.flatten().eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5).flatten()
        neg_log_likelihood = -1.0 * (
            Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob)
        )  # negative log bernoulli

        return neg_log_likelihood, A
    

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)
    
class GatedAttentionDomainAdaptation(nn.Module):
    def __init__(self):
        super(GatedAttentionDomainAdaptation, self).__init__()
        self.nb_features = 2048
        self.L = 256
        self.D = 256

        self.fc1 = nn.Sequential(nn.Linear(self.nb_features, self.L))

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())

        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())

        self.attention_weights = nn.Sequential(nn.Linear(self.D, 1))

        self.fc2 = nn.Sequential(nn.Linear(self.L, 128), nn.Tanh())

        self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.Tanh())

        self.center_classifier = nn.Sequential(nn.Linear(64,32),nn.Tanh(),nn.Linear(32,5))

        self.classifier = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x, constant):
        # x is of size N x nb_features
        # N:number of tiles corresponding to the WSI
        # nb_features:number of features per tile
        x = self.fc1(x)  # NxL
        #x = self.bn1(x)
        #x = self.fc1bis(x)
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        #A = self.attention_weights(A_V)
        A = torch.transpose(A, 1, 2)  # KxN
        #A = F.softmax(A*1000/A.shape[2], dim=2)  # softmax over N
        A = F.softmax(A, dim=2)
        M = torch.matmul(A, x)  # KxL
        M = M.squeeze(1)
        M = self.fc2(M)  # Kx128
        M = self.fc3(M)  # Kx64
        Y_prob = self.classifier(M)
        input = GradReverse.grad_reverse(M, constant)
        Y_prob_center = self.center_classifier(input)
        prob_center = F.softmax(Y_prob_center,dim=1)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A, prob_center

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y, constant):
        Y = Y.float()
        _, Y_hat, _, _ = self.forward(X, constant)
        error = 1.0 - Y_hat.flatten().eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y, centers, constant):
        Y = Y.float()
        Y_prob, _, A, prob_center = self.forward(X, constant)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5).flatten()
        prob_center = torch.clamp(prob_center, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (
            Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob)
        )  # negative log bernoulli
        domain_criterion = nn.CrossEntropyLoss()
        domain_loss = domain_criterion(prob_center,centers)
        return neg_log_likelihood, domain_loss, A
    def calculate_domain_loss(self,X,centers,constant):
        Y_prob, _, A, prob_center = self.forward(X, constant)
        prob_center = torch.clamp(prob_center, min=1e-5, max=1.0 - 1e-5)
        domain_criterion = nn.CrossEntropyLoss()
        domain_loss = domain_criterion(prob_center,centers)
        return domain_loss
class Chowder(nn.Module):
    def __init__(self):
        super(Chowder, self).__init__()
        self.nb_features = 2048
        self.L = 200
        self.D = 100

        self.fc1 = nn.Linear(self.nb_features, 1)

        #self.fc2 = nn.Sequential(nn.Linear(200, self.L), nn.Sigmoid())

        #self.fc3 = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())

        self.classifier = nn.Sequential(nn.Linear(200, 1), nn.Sigmoid())

    def forward(self, x):
        # x is of size N x nb_features
        # N:number of tiles corresponding to the WSI
        # nb_features:number of features per tile
        x = self.fc1(x)  # NxL
 
        y1 = torch.topk(x,100,1)[0].squeeze()
        y2 = torch.topk(x,100,1,False)[0].squeeze()
        y = torch.concat((y1,y2),1)
        #y = self.fc2(y)
        #y = self.fc3(y)
        Y_prob = self.classifier(y)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat = self.forward(X)
        error = 1.0 - Y_hat.flatten().eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _ = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5).flatten()
        neg_log_likelihood = -1.0 * (
            Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob)
        )  # negative log bernoulli

        return neg_log_likelihood

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.nb_features = 2048
        self.L = 128
        self.D = 128

        self.fc1 = nn.Linear(self.nb_features, self.L)

        self.attention = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh(),nn.Linear(self.D,1))

        self.attention_weights = nn.Linear(self.D, 1)

        self.fc2 = nn.Sequential(nn.Linear(self.L, 128), nn.Tanh())

        self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.Tanh())

        self.classifier = nn.Sequential(nn.Linear(self.L, 1), nn.Sigmoid())

    def forward(self, x):
        # x is of size N x nb_features
        # N:number of tiles corresponding to the WSI
        # nb_features:number of features per tile
        x = self.fc1(x)  # NxL
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 2)  # KxN
        A = F.softmax(A, dim=2)  # softmax over N
        if not self.use_adaptive:
            M = torch.matmul(A, x)  # KxL
        else:
            mean_attention = torch.mean(A)
            thresh = nn.Threshold(
                mean_attention.item(), 0
            )  # set elements in the attention vector to zero if they are <= mean_attention of the cycle
            positive_attention = thresh(
                A.squeeze(0)
            )  # vector of [1,n] to [n] and then threshold
            pseudo_positive = torch.where(
                positive_attention > 0,
                torch.transpose(x, 1, 2),
                torch.tensor([0.0], device="cuda:3"),
            )  # select all elements of the hidden feature embeddings that have sufficient attention
            positive_attention = positive_attention.unsqueeze(
                0
            )  # reverse vector [n] to [1,n]
            negative_attention = torch.where(
                A.squeeze(0) <= mean_attention,
                A.squeeze(),
                torch.tensor([0.0], device="cuda:3"),
            )  # attention vector with zeros if elements > mean_attention
            pseudo_negative = torch.where(
                negative_attention > 0,
                torch.transpose(x, 1, 2),
                torch.tensor([0.0], device="cuda:3"),
            )  # select all elements of the hidden feature embeddings matching this new vector
            negative_attention = negative_attention.unsqueeze(0)

            x_mul_positive = torch.matmul(
                positive_attention, torch.transpose(pseudo_positive, 1, 0)
            )  # pseudo positive instances N-N_in Matrix Mult.
            x_mul_negative = self.lam * torch.matmul(
                negative_attention, torch.transpose(pseudo_negative, 1, 0)
            )  # pseudo negative instances N_in Matrix Mult modfied by lambda hyperparameter (increases weightdifferences between pos/neg)
            M = (
                x_mul_positive + x_mul_negative
            )  # see formula 2 of Li et al. MICCAI 2019

        M = M.squeeze(1)
        M = self.fc2(M)  # Kx128
        M = self.fc3(M)  # Kx64
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.flatten().eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5).flatten()
        neg_log_likelihood = -1.0 * (
            Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob)
        )  # negative log bernoulli

        return neg_log_likelihood, A
