"""TableNet Module."""

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.models import vgg19, vgg19_bn

EPSILON = 1e-15


class TableNetModule(pl.LightningModule):
    """Pytorch Lightning Module for TableNet."""

    def __init__(self, num_class: int = 1, batch_norm: bool = False):
        """Initialize TableNet Module.

        Args:
            num_class (int): Number of classes per point.
            batch_norm (bool): Select VGG with or without batch normalization.
        """
        super().__init__()
        self.model = TableNet(num_class, batch_norm)
        self.num_class = num_class
        self.dice_loss = DiceLoss()

    def forward(self, batch):
        """Perform forward-pass.

        Args:
            batch (tensor): Batch of images to perform forward-pass.

        Returns (Tuple[tensor, tensor]): Table, Column prediction.
        """
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        """Get training step.

        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.

        Returns: Tensor
        """
        samples, labels_table, labels_column = batch
        output_table, output_column = self.forward(samples)

        loss_table = self.dice_loss(output_table, labels_table)
        loss_column = self.dice_loss(output_column, labels_column)

        self.log('train_loss_table', loss_table)
        self.log('train_loss_column', loss_column)
        self.log('train_loss', loss_column + loss_table)
        return loss_table + loss_column

    def validation_step(self, batch, batch_idx):
        """Get validation step.

        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.

        Returns: Tensor
        """
        samples, labels_table, labels_column = batch
        output_table, output_column = self.forward(samples)

        loss_table = self.dice_loss(output_table, labels_table)
        loss_column = self.dice_loss(output_column, labels_column)

        if batch_idx == 0:
            self._log_images("validation", samples, labels_table, labels_column, output_table, output_column)

        self.log('valid_loss_table', loss_table, on_epoch=True)
        self.log('valid_loss_column', loss_column, on_epoch=True)
        self.log('validation_loss', loss_column + loss_table, on_epoch=True)
        self.log('validation_iou_table', binary_mean_iou(output_table, labels_table), on_epoch=True)
        self.log('validation_iou_column', binary_mean_iou(output_column, labels_column), on_epoch=True)
        return loss_table + loss_column

    def test_step(self, batch, batch_idx):
        """Get test step.

        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.

        Returns: Tensor
        """
        samples, labels_table, labels_column = batch
        output_table, output_column = self.forward(samples)

        loss_table = self.dice_loss(output_table, labels_table)
        loss_column = self.dice_loss(output_column, labels_column)

        if batch_idx == 0:
            self._log_images("test", samples, labels_table, labels_column, output_table, output_column)

        self.log('test_loss_table', loss_table, on_epoch=True)
        self.log('test_loss_column', loss_column, on_epoch=True)
        self.log('test_loss', loss_column + loss_table, on_epoch=True)
        self.log('test_iou_table', binary_mean_iou(output_table, labels_table), on_epoch=True)
        self.log('test_iou_column', binary_mean_iou(output_column, labels_column), on_epoch=True)
        return loss_table + loss_column

    def configure_optimizers(self):
        """Configure optimizer for pytorch lighting.

        Returns: optimizer and scheduler for pytorch lighting.

        """
        optimizer = optim.SGD(self.parameters(), lr=0.0001)
        scheduler = {
            'scheduler': optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=0.0001, steps_per_epoch=204, epochs=500, pct_start=0.1),
            'interval': 'step',
        }

        return [optimizer], [scheduler]

    def _log_images(self, mode, samples, labels_table, labels_column, output_table, output_column):
        """Log image on to logger."""
        self.logger.experiment.add_images(f'{mode}_generated_images', samples[0:4], self.current_epoch)
        self.logger.experiment.add_images(f'{mode}_labels_table', labels_table[0:4], self.current_epoch)
        self.logger.experiment.add_images(f'{mode}_labels_column', labels_column[0:4], self.current_epoch)
        self.logger.experiment.add_images(f'{mode}_output_table', output_table[0:4], self.current_epoch)
        self.logger.experiment.add_images(f'{mode}_output_column', output_column[0:4], self.current_epoch)


class TableNet(nn.Module):
    """TableNet."""

    def __init__(self, num_class: int, batch_norm: bool = False):
        """Initialize TableNet.

        Args:
            num_class (int): Number of classes per point.
            batch_norm (bool): Select VGG with or without batch normalization.
        """
        super().__init__()
        self.vgg = vgg19(pretrained=True).features if not batch_norm else vgg19_bn(pretrained=True).features
        self.layers = [18, 27] if not batch_norm else [26, 39]
        self.model = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.8),
                                   nn.Conv2d(512, 512, kernel_size=1),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.8))
        self.table_decoder = TableDecoder(num_class)
        self.column_decoder = ColumnDecoder(num_class)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): Batch of images to perform forward-pass.

        Returns (Tuple[tensor, tensor]): Table, Column prediction.
        """
        results = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                results.append(x)
        x_table = self.table_decoder(x, results)
        x_column = self.column_decoder(x, results)
        return torch.sigmoid(x_table), torch.sigmoid(x_column)


class ColumnDecoder(nn.Module):
    """Column Decoder."""

    def __init__(self, num_classes: int):
        """Initialize Column Decoder.

        Args:
            num_classes (int): Number of classes per point.
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.layer = nn.ConvTranspose2d(1280, num_classes, kernel_size=2, stride=2, dilation=1)

    def forward(self, x, pools):
        """Forward pass.

        Args:
            x (tensor): Batch of images to perform forward-pass.
            pools (Tuple[tensor, tensor]): The 3 and 4 pooling layer from VGG-19.

        Returns (tensor): Forward-pass result tensor.

        """
        pool_3, pool_4 = pools
        x = self.decoder(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, pool_4], dim=1)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, pool_3], dim=1)
        x = F.interpolate(x, scale_factor=2)
        x = F.interpolate(x, scale_factor=2)
        return self.layer(x)


class TableDecoder(ColumnDecoder):
    """Table Decoder."""

    def __init__(self, num_classes):
        """Initialize Table decoder.

        Args:
            num_classes (int): Number of classes per point.
        """
        super().__init__(num_classes)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )


class DiceLoss(nn.Module):
    """Dice loss."""

    def __init__(self):
        """Dice Loss."""
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        """Calculate loss.

        Args:
            inputs (tensor): Output from the forward pass.
            targets (tensor): Labels.
            smooth (float): Value to smooth the loss.

        Returns (tensor): Dice loss.

        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def binary_mean_iou(inputs, targets):
    """Calculate binary mean intersection over union.

    Args:
        inputs (tensor): Output from the forward pass.
        targets (tensor): Labels.

    Returns (tensor): Intersection over union value.
    """
    output = (inputs > 0).int()

    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)

    intersection = (targets * output).sum()

    union = targets.sum() + output.sum() - intersection

    result = (intersection + EPSILON) / (union + EPSILON)

    return result
