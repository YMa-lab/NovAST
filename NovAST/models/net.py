import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATv2Conv, SAGEConv
from typing import List, Callable, Union, Any, TypeVar, Tuple
from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy
from tllib.modules.kernels import GaussianKernel

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return 10 * out    

class FCNet(nn.Module):
    def __init__(self, x_dim, num_cls):
        super(FCNet, self).__init__()
        self.linear = NormedLinear(x_dim, num_cls)

    def forward(self, data):
        x = data
        out = self.linear(x)
        return out, x, x

class AE(nn.Module):
    def __init__(self, x_dim: int, latent_dim: int, hidden_dims: List = None, dropout=0):
        super(AE, self).__init__()
        self.x_dim = x_dim 
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 128]
        modules_encoder = []
        for h_dim in hidden_dims:
                    modules_encoder.append(
                        nn.Sequential(
                            nn.Linear(x_dim, h_dim),
                            nn.ReLU(),
                            nn.Dropout(p=dropout)
                    ))
                    x_dim = h_dim
        self.encoder = nn.Sequential(*modules_encoder)
        self.linear = nn.Linear(x_dim, self.latent_dim)

        hidden_dims.reverse()
        modules_decoder = []
        for h_dim in hidden_dims:
                    modules_decoder.append(
                        nn.Sequential(
                            nn.Linear(latent_dim, h_dim),
                            nn.BatchNorm1d(h_dim),
                            nn.LeakyReLU(),
                            nn.Dropout(p=dropout)
                    ))
                    latent_dim = h_dim
        modules_decoder.append(nn.Linear(latent_dim, self.x_dim))
        self.decoder = nn.Sequential(*modules_decoder)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent representation
        """
        result = self.encoder(input)
        result = self.linear(result)

        return result
     
    def encode_gcn(self, input, edge_index):
        """
        Encodes the input by passing through the encoder network
        and returns the latent representation
        """
        result = self.encoder(input)
        result = self.gcn(result, edge_index)
        result = self.linear(result)

        return result
    
    def encode_gcn_front(self, input, edge_index):
        """
        Encodes the input by passing through the encoder network
        and returns the latent representation
        """
        result = self.gcn(input, edge_index)
        result = self.encoder(result)
        result = self.linear(result)

        return result
    
    def decode_recons(self, z):
        """
        Maps the given latent codes
        onto the sequence space.
        """
        result = self.decoder(z)
        return result
    
    def decode_labels(self, z, y):
        """
        Maps the given latent codes
        onto the pseudo label based
        on similarities
        """
        similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
        num_classes = torch.max(y) + 1
        y_one_hot = F.one_hot(y, num_classes=num_classes).float()
        # similarity_matrix (N, N) * y_one_hot (N, num_classes) -> class_sums (N, num_classes)
        class_sums = similarity_matrix @ y_one_hot
        softmax_output = F.softmax(class_sums, dim=1)
        pseudo_label = torch.argmax(softmax_output, dim=1)
        return [softmax_output, pseudo_label]

    def forward(self, data, y=None):
        x = data
        z = self.encode(x)
        output = self.decode_recons(z)
        if y != None:
            softmax_output, pseudo_label = self.decode_labels(z, y)
        else:
            softmax_output = pseudo_label = None
        return  [output, z, softmax_output, pseudo_label]
    
    def mmf_loss(self, features, labels, k=10, margin=1.0):
        """
        Min-Max Feature (MMF) loss for semi-supervised learning. Adopted from ECCV 2018 Paper "Transductive Semi-Supervised Deep Learning
        using Min-Max Features"

        Parameters:
        - features: Tensor of shape (N, D), where N is the batch size and D is the feature dimension.
        - labels: Tensor of shape (N,), containing class labels for each sample in the batch.
        - margin: Float, the predefined margin for separating different class features.

        Returns:
        - loss: MMF loss value.
        """
        # calculate pairwise Squared Euclidean distance distance matrix
        distances = torch.cdist(features, features, p=2) ** 2 
        # print('distance min: ' + str(distances.min()))
        # print('distance max: ' + str(distances.max()))
        # print('distance mean: ' + str(distances.mean()))

        # Sum of distances to the k-nearest neighbors (exclude self)
        k = min(k, features.shape[0] - 1) # to prevent the last batch has a sample size less than k
        knn_distances, _ = distances.topk(k=k + 1, dim=-1, largest=False) 
        proximity = knn_distances[:, 1:].sum(dim=1)

        # Normalize proximity to get confidence (1 - di/dmax)
        dmax = proximity.max()
        confidences = 1 - (proximity / dmax)
        
        # Indicator matrices for same class (delta_ij=1) and different class (delta_ij=0)
        label_equal = labels.unsqueeze(1) == labels.unsqueeze(0)
        label_diff = ~label_equal

        # Compute r_i * r_j for each pair
        confidence_weights = confidences.unsqueeze(1) * confidences.unsqueeze(0)

        # Minimize distance for same-class pairs, weighted by confidence
        within_class_loss = torch.sum(distances * label_equal.float() * confidence_weights)

        # Enforce margin for different-class pairs, weighted by confidence
        between_class_loss = torch.sum(F.relu(margin - distances) * label_diff.float() * confidence_weights)

        # Normalize by the number of pairs (for stability purpose)
        num_within_pairs = label_equal.sum().float()
        num_between_pairs = label_diff.sum().float()
        
        if num_within_pairs > 0:
            within_class_loss /= num_within_pairs
        if num_between_pairs > 0:
            between_class_loss /= num_between_pairs

        # Total MMF loss
        loss = within_class_loss + between_class_loss
        return loss
    
    def loss_function(self,
                      *args,
                      ) -> dict:
        """
        Computes the AE loss function.
        """
        labeled_recons = args[0]
        labeled_input = args[1]
        softmax_output = args[2]
        labeled_z = args[3]
        labeled_y = args[4]
        unlabeled_recons = args[5]
        unlabeled_input = args[6]
        gamma = args[7]
        alpha = args[8]
        beta = args[9]
        k = args[10]
        margin = args[11]
        
        recons_loss = F.mse_loss(labeled_recons, labeled_input) + F.mse_loss(unlabeled_recons, unlabeled_input)

        sim_loss = F.cross_entropy(softmax_output, labeled_y)

        mmf_loss = self.mmf_loss(labeled_z, labeled_y, k=k, margin=margin)

        loss = torch.mean(alpha*recons_loss + gamma*sim_loss + beta*mmf_loss)
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(),  "Similarity_Loss": sim_loss.detach(), "MMF_Loss": mmf_loss.detach()}
