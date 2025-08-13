# -*- coding: utf-8 -*-
import torch.nn as nn
import torch


class MultiArteryVeinLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        weights = torch.FloatTensor([1, 1, 3, 5]).to(config.device)
        self.av_criterion = nn.CrossEntropyLoss(weight=weights).to(config.device)

        self.vessel_criterion = nn.BCELoss().to(config.device)

    def forward(self, prediction, label):
        vessel = torch.sigmoid(torch.sum(prediction, dim=1) - prediction[:, 0, ...])
        vessel_label = torch.where(label > 0, 1, 0).type(torch.FloatTensor).to(self.config.device)

        loss = self.av_criterion(prediction, label) + self.vessel_criterion(vessel, vessel_label)
        return loss


class SingleArteryVeinLoss(nn.Module):
    def __init__(self, config, beta=(5, 1, 4), ep=1e-6):
        super().__init__()

        pos_weight = torch.FloatTensor([5]).to(config.device)

        self.artery_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(config.device)
        self.vein_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(config.device)
        self.uncertain_criterion = nn.BCEWithLogitsLoss().to(config.device)

        self.bce_loss = nn.BCELoss().to(config.device)
        self.beta = beta

        self.ep = ep

    def dice_loss(self, prediction, label):
        intersection = 2 * torch.sum(prediction * label) + self.ep
        union = torch.sum(prediction) + torch.sum(label) + self.ep
        loss = 1 - intersection / union
        return loss

    def forward(self, prediction, label):
        loss = 0

        for i, criterion in enumerate([self.artery_criterion, self.uncertain_criterion, self.vein_criterion]):
            p, t = prediction[:, i, ...], label[:, i, ...]
            loss += self.beta[i] * criterion(p, t)
            loss += 0.1 * self.beta[i] * self.dice_loss(torch.sigmoid(p), t)

        return loss / 10


class SerialConnectedLoss(nn.Module):
    def __init__(self, config, ep=1e-6):
        super().__init__()
        self.ep = ep
        self.criterion = SingleArteryVeinLoss(config).to(config.device)

    def forward(self, predict, label):
        loss_multi = 0

        for idx, p in enumerate(predict[:-1]):
            # loss_multi += (1 / (len(predict) - idx)) * (self.criterion(p, label) + 0.1 * self.dice_loss(p, label))
            loss_multi += self.criterion(p, label)

        loss_multi /= len(predict)
        loss_multi += self.criterion(predict[-1], label)

        return loss_multi


class BCE3Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_vessels, vessels):
        pred_a = pred_vessels[:, 0, :, :]
        pred_v = pred_vessels[:, 1, :, :]
        pred_vt = pred_vessels[:, 2, :, :]

        gt_a = vessels[:, 0, :, :]
        gt_v = vessels[:, 1, :, :]
        gt_vt = vessels[:, 2, :, :]

        loss = 1.1 * self.loss(pred_a, gt_a)
        loss += 1.2 * self.loss(pred_v, gt_v)
        loss += 0.7 * self.loss(pred_vt, gt_vt)

        return loss

    def process_predicted(self, prediction):
        return torch.sigmoid(prediction.clone())


class RRLoss(nn.Module):
    """Recursive refinement loss.
    """

    def __init__(self, base_criterion=None):
        super().__init__()
        self.base_criterion = BCE3Loss()

    def forward(self, predictions, gt):
        loss_1 = self.base_criterion(predictions[0], gt)
        if len(predictions) == 1:
            return loss_1

        # Second loss (refinement) inspired by Mosinska:CVPR:2018.
        loss_2 = 1 * self.base_criterion(predictions[1], gt)
        if len(predictions) == 2:
            return loss_1 + loss_2
        for i, prediction in enumerate(predictions[2:], 2):
            loss_2 += i * self.base_criterion(prediction, gt)

        K = len(predictions[1:])
        Z = (1 / 2) * K * (K + 1)

        loss_2 *= 1 / Z

        loss = loss_1 + loss_2

        return loss
