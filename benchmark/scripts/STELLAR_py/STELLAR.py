import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .models.net import Encoder, FCNet
from .utils import entropy, MarginLoss
import numpy as np
from itertools import cycle
import copy
from torch_geometric.loader import ClusterData, ClusterLoader
import scanpy as sc
from anndata import AnnData
import os

class STELLAR:

    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        args.input_dim = dataset.unlabeled_data.x.shape[-1]
        self.model = Encoder(args.input_dim, args.num_heads)
        self.model = self.model.to(args.device)

    def train_supervised(self, args, model, device, dataset, optimizer, epoch):
        model.train()
        ce = nn.CrossEntropyLoss()
        sum_loss = 0

        labeled_graph = dataset.labeled_data
        labeled_data = ClusterData(labeled_graph, num_parts=args.num_parts_train, recursive=False) 
        labeled_loader = ClusterLoader(labeled_data, batch_size=1, shuffle=True, drop_last=True,
                                    num_workers=1)
        
        for batch_idx, labeled_x in enumerate(labeled_loader):
            labeled_x = labeled_x.to(device)
            optimizer.zero_grad()
            output, _, _ = model(labeled_x)
            
            loss = ce(output, labeled_x.y)
            
            optimizer.zero_grad()
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('Seed loss: {:.6f}'.format(sum_loss / (batch_idx + 1)))

    def est_seeds(self, args, model, device, dataset, clusters, num_seed_class):
        model.eval()
        entrs = np.array([])
        with torch.no_grad():
            labeled_graph, unlabeled_graph = dataset.labeled_data, dataset.unlabeled_data
            unlabeled_graph_cp = copy.deepcopy(unlabeled_graph)
            unlabeled_graph_cp = unlabeled_graph_cp.to(device)
            output, _, _ = model(unlabeled_graph_cp)
            prob = F.softmax(output, dim=1)
            entr = -torch.sum(prob * torch.log(prob), 1)
            entrs = np.append(entrs, entr.cpu().numpy())
        
        entrs_per_cluster = []
        for i in range(np.max(clusters)+1):
            locs = np.where(clusters == i)[0]
            entrs_per_cluster.append(np.mean(entrs[locs]))
        entrs_per_cluster = np.array(entrs_per_cluster)
        if num_seed_class > 0:
            novel_cluster_idxs = np.argsort(entrs_per_cluster)[-num_seed_class:]
        else:
            novel_cluster_idxs = []
        novel_label_seeds = np.zeros_like(clusters)
        largest_seen_id = torch.max(labeled_graph.y)
        
        for i, idx in enumerate(novel_cluster_idxs):
            novel_label_seeds[clusters == idx] = largest_seen_id + i + 1
        return novel_label_seeds

    def train_epoch(self, args, model, device, dataset, optimizer, m, epoch): 
        """ Train for 1 epoch."""
        model.train()
        bce = nn.BCELoss()
        ce = MarginLoss(m=-m)
        sum_loss = 0

        labeled_graph, unlabeled_graph = dataset.labeled_data, dataset.unlabeled_data
        labeled_data = ClusterData(labeled_graph, num_parts=self.args.num_parts_train, recursive=False)
        labeled_loader = ClusterLoader(labeled_data, batch_size=1, shuffle=True, drop_last=True,
                                    num_workers=1)
        unlabeled_data = ClusterData(unlabeled_graph, num_parts=self.args.num_parts_test, recursive=False)
        unlabeled_loader = ClusterLoader(unlabeled_data, batch_size=1, shuffle=True, drop_last=True,
                                    num_workers=1)
        unlabel_loader_iter = cycle(unlabeled_loader)

        for batch_idx, labeled_x in enumerate(labeled_loader):
            unlabeled_x = next(unlabel_loader_iter)
            unlabeled_ce_idx = torch.where(unlabeled_x.novel_label_seeds>0)[0]
            labeled_x, unlabeled_x = labeled_x.to(device), unlabeled_x.to(device)
            optimizer.zero_grad()
            labeled_output, labeled_feat, _ = model(labeled_x)
            unlabeled_output, unlabeled_feat, _ = model(unlabeled_x)
            labeled_len = len(labeled_output)
            batch_size = len(labeled_output) + len(unlabeled_output)
            output = torch.cat([labeled_output, unlabeled_output], dim=0)
            feat = torch.cat([labeled_feat, unlabeled_feat], dim=0)
            
            prob = F.softmax(output, dim=1)
            # Similarity labels
            feat_detach = feat.detach()
            feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
            cosine_dist = torch.mm(feat_norm, feat_norm.t())

            pos_pairs = []
            target = labeled_x.y
            target_np = target.cpu().numpy()
            
            for i in range(labeled_len):
                target_i = target_np[i]
                idxs = np.where(target_np == target_i)[0]
                if len(idxs) == 1:
                    pos_pairs.append(idxs[0])
                else:
                    selec_idx = np.random.choice(idxs, 1)
                    while selec_idx == i:
                        selec_idx = np.random.choice(idxs, 1)
                    pos_pairs.append(int(selec_idx))
            
            unlabel_cosine_dist = cosine_dist[labeled_len:, :]
            vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
            pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
            pos_pairs.extend(pos_idx)
            
            pos_prob = prob[pos_pairs, :]
            pos_sim = torch.bmm(prob.view(batch_size, 1, -1), pos_prob.view(batch_size, -1, 1)).squeeze()
            ones = torch.ones_like(pos_sim)
            bce_loss = bce(pos_sim, ones)
            ce_idx = torch.cat((torch.arange(labeled_len), labeled_len+unlabeled_ce_idx))
            target = torch.cat((target, unlabeled_x.novel_label_seeds))
            ce_loss = ce(output[ce_idx], target[ce_idx])
            entropy_loss = entropy(torch.mean(prob, 0))
            
            loss = 1 * bce_loss + 1 * ce_loss - 0.3 * entropy_loss
            
            optimizer.zero_grad()
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()

        print('Loss: {:.6f}'.format(sum_loss / (batch_idx + 1)))
        return loss, bce_loss, ce_loss, entropy_loss


    def pred(self):
        self.model.eval()
        preds = np.array([])
        confs = np.array([])
        feats = np.array([])
        out_feats = np.array([])
        with torch.no_grad():
            _, unlabeled_graph = self.dataset.labeled_data, self.dataset.unlabeled_data
            unlabeled_graph_cp = copy.deepcopy(unlabeled_graph)
            unlabeled_graph_cp = unlabeled_graph_cp.to(self.args.device)
            output, feat, out_feat = self.model(unlabeled_graph_cp)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
            feats = np.append(feats, feat.cpu().numpy())
            out_feats = np.append(out_feats, out_feat.cpu().numpy())
        preds = preds.astype(int)
        mean_uncert = 1 - np.mean(confs)
        return mean_uncert, preds, feats, out_feats

    def train(self):
        unlabel_x = self.dataset.unlabeled_data.x
        adata = AnnData(unlabel_x.numpy())
        sc.pp.neighbors(adata)
        sc.tl.louvain(adata, 1)
        clusters = adata.obs['louvain'].values
        clusters = clusters.astype(int)

        seed_model = FCNet(x_dim = self.args.input_dim, num_cls=torch.max(self.dataset.labeled_data.y)+1)
        seed_model = seed_model.to(self.args.device)
        seed_optimizer = optim.Adam(seed_model.parameters(), lr=1e-3, weight_decay=0)
        for epoch in range(20):
            self.train_supervised(self.args, seed_model, self.args.device, self.dataset, seed_optimizer, epoch)
        print(f'num_seed_class is {self.args.num_seed_class}')
        novel_label_seeds = self.est_seeds(self.args, seed_model, self.args.device, self.dataset, clusters, self.args.num_seed_class)
        self.dataset.unlabeled_data.novel_label_seeds = torch.tensor(novel_label_seeds)
        # Set the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.stellar_lr, weight_decay=self.args.stellar_wd)
        
        total_losses = []
        DCT_losses = []
        RCT_losses = []
        R_losses = []
        
        for epoch in range(self.args.stellar_epochs):
            mean_uncert, _, _, _ = self.pred()
            self.train_epoch(self.args, self.model, self.args.device, self.dataset, optimizer, mean_uncert, epoch)
        
        