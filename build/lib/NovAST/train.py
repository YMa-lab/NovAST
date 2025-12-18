import os
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torch.cuda.amp import autocast
from . import models

class NovAST_train:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        args.input_dim = dataset.unlabeled_data.x.shape[-1]
        self.model = models.AE(x_dim = args.input_dim, latent_dim = args.latent_dim, hidden_dims=args.hidden_dims, dropout=args.dropout)
        self.model = self.model.to(self.args.device)
        self.savedir = os.path.join(args.savedir, args.dataset)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.args.device.type == 'cuda'))

    def pred_supervised(self, model, data, device):
        model.eval()
        all_labels = []
        with torch.no_grad():
            dataloader = DataLoader(data, batch_size=4, shuffle=False,num_workers=1)
            for batch_idx, z_batch in enumerate(dataloader):
                z_batch = z_batch.to(device)
                out, _, _ = model(z_batch)
                softmax_output = F.softmax(out, dim=1)
                labels = torch.argmax(softmax_output, dim=1)
                all_labels.append(labels.cpu())
            label = torch.cat(all_labels, dim=0)
        return label

    def train_epoch(self, args, model, device, dataset, optimizer, epoch):
        model.train()
        sum_loss = sum_recons = sum_sim = sum_mmf = 0

        labeled_loader = DataLoader(dataset.labeled_data, batch_size=self.args.batch_size, shuffle=True,
                                        num_workers=0)
        unlabeled_loader = DataLoader(dataset.unlabeled_data, batch_size=self.args.batch_size, shuffle=True,
                                    num_workers=0)
        unlabel_loader_iter = cycle(unlabeled_loader)

        for batch_idx, (labeled_x, labeled_y) in enumerate(labeled_loader):
            labeled_x, labeled_y = labeled_x.to(device), labeled_y.to(device)
            unlabeled_x, unlabeled_y = next(unlabel_loader_iter)
            unlabeled_x, unlabeled_y = unlabeled_x.to(device), unlabeled_y.to(device)
            optimizer.zero_grad()
            with autocast(): # use AMP to speed up
                labeled_output, labeled_z, softmax_output, _ = model(labeled_x,labeled_y)
                unlabeled_output, _, _ , _ = model(unlabeled_x)
                loss_dict = model.loss_function(labeled_output, labeled_x, softmax_output, labeled_z, labeled_y, unlabeled_output, unlabeled_x, args.gamma, args.alpha, args.beta, args.mmf_k, args.mmf_margin)

            self.scaler.scale(loss_dict['loss']).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            sum_loss   += loss_dict["loss"].item()
            sum_recons += loss_dict["Reconstruction_Loss"].item()
            sum_sim    += loss_dict["Similarity_Loss"].item()
            sum_mmf    += loss_dict["MMF_Loss"].item()

        n_batches = batch_idx + 1
        avg_loss = sum_loss / n_batches
        avg_recons = sum_recons / n_batches
        avg_sim = sum_sim / n_batches
        avg_mmf = sum_mmf / n_batches

        # Print
        print(f"[Epoch {epoch}] Total: {avg_loss:.6f} | Recon: {avg_recons:.6f} | "
            f"Sim: {avg_sim:.6f} | MMF: {avg_mmf:.6f}")

        return loss_dict

    def pred(self, data):
        self.model.eval()
        all_outputs = []
        all_z = []

        with torch.no_grad():
            dataset = TensorDataset(data.x, data.y)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False,num_workers=1)
            
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.args.device)
                
                output, z, _, _ = self.model(x_batch)
                
                all_outputs.append(output.cpu())
                all_z.append(z.cpu())
            output = torch.cat(all_outputs, dim=0)
            z = torch.cat(all_z, dim=0)
        return [output, z]
    
    def train(self):
        total_losses = []
        recons_losses = []
        sim_losses = []
        mmf_losses = []
        
        ## train the vae with similarity loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        for epoch in range(self.args.epochs):
            train_fn = self.train_epoch
            loss_dict = train_fn(self.args, self.model, self.args.device, self.dataset, optimizer, epoch)
            total_losses.append(loss_dict['loss'].item())
            recons_losses.append(loss_dict['Reconstruction_Loss'].item())
            sim_losses.append(loss_dict['Similarity_Loss'].item())
            mmf_losses.append(loss_dict['MMF_Loss'].item())
        
        labeled_data, unlabeled_data = self.dataset.labeled_data, self.dataset.unlabeled_data
        pred_fn = self.pred
        _,z_train = pred_fn(labeled_data)
        _,z_test = pred_fn(unlabeled_data)
        
        model_cpu = self.model.to('cpu')
        
        return [z_train, z_test, total_losses, recons_losses, sim_losses, mmf_losses, model_cpu]