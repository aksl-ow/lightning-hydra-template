from typing import Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import joblib

import pytorch_lightning as pl
import torch.nn.functional as F

from classic_algos.nn import DeepMIL
from torchmetrics import AUROC

import torch

class DeepmilClassificationLitModel(pl.LightningModule):
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        d_model_attention: int = 128,
        temperature: float = 1.0,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
        lr: float = 1e-3,
        n_tiles: int = None,
        weight_decay: float = 0.0005
        
    ):
        super(DeepmilClassificationLitModel, self).__init__()
        # call this to save init params to the checkpoint
        self.save_hyperparameters()

        self.deepmil = DeepMIL(
            in_features=self.hparams.in_features,
            out_features=self.hparams.out_features,
            d_model_attention=self.hparams.d_model_attention,
            temperature=self.hparams.temperature,
            tiles_mlp_hidden=self.hparams.tiles_mlp_hidden,
            mlp_hidden=self.hparams.mlp_hidden,
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_activation=self.hparams.mlp_activation,
            bias=self.hparams.bias,
        )
        if self.hparams.out_features > 2:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        
        self.train_auroc = AUROC(num_classes=self.hparams.out_features)
        self.val_auroc = AUROC(num_classes=self.hparams.out_features)
        self.test_auroc = AUROC(num_classes=self.hparams.out_features)
    
    def process_step_outputs(self, step_outputs):
        avg_loss = torch.stack(
            [outputs["loss"] for outputs in step_outputs]
        ).mean()
        y_hat = torch.cat(
            [outputs["y_hat"] for outputs in step_outputs]
        ).numpy()
        y = torch.cat([outputs["y"] for outputs in step_outputs]).numpy()
        scores = torch.cat(
            [
                F.pad(
                    input=outputs["scores"],
                    pad=(
                        0,
                        self.hparams.n_tiles - outputs["scores"].shape[1],
                        0,
                        0,
                    ),
                    mode="constant",
                    value=float("-inf"),
                )
                for outputs in step_outputs
            ]
        ).numpy()
        coords = torch.cat(
            [
                F.pad(
                    input=outputs["coords"],
                    pad=(0, 0, 0,
                        self.hparams.n_tiles - outputs["coords"].shape[1],
                        0, 0,
                    ),
                    mode="constant",
                    value=float("-inf"),
                )
                for outputs in step_outputs
            ]
        ).numpy()
        wsi_paths = np.hstack(
            [outputs["wsi_path"] for outputs in step_outputs]
        )
        sample_ids = np.hstack(
            [outputs["sample_id"] for outputs in step_outputs]
        )
        patient_ids = (
            torch.cat([outputs["patient_id"] for outputs in step_outputs])
            .cpu()
            .numpy()
        )

        prediction = {
            "y_hat": y_hat,
            "y": y,
            "coords": coords,
            "wsi_paths": wsi_paths,
            "sample_ids": sample_ids,
            "patient_ids": patient_ids,
            "scores": scores,
        }
        return prediction

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        return self.deepmil.forward(x, mask)
    
    def step(self, batch: Any):
        x, mask, y, metadata = batch
        coords, x = x[:, :, 1:3], x[:, :, 3:]
        logits, scores = self.forward(x, mask)
        if self.hparams.out_features > 2:
            loss = self.criterion(logits, y.squeeze(-1).long())
            y_hat = torch.softmax(logits, -1)
        else:
            loss = self.criterion(logits, y)
            y_hat = torch.sigmoid(logits)
        
        outputs = {
            "y_hat": y_hat.cpu().detach(),
            "y": y.cpu().detach(),
            "scores": torch.squeeze(scores.cpu().detach(), -1),
            "coords": coords.cpu().detach(),
        }
        outputs.update(metadata)

        return y_hat, y, loss, outputs

    def training_step(self, batch, batch_idx):
        y_hat, y, loss, outputs = self.step(batch)

        # log train metrics
        auroc = self.train_auroc(y_hat, y.type(torch.long))
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in zip(["loss", "train_auroc"], [loss, auroc]):
            outputs[k] = v

        return outputs

    def training_epoch_end(self, training_step_outputs):
        prediction = self.process_step_outputs(step_outputs=training_step_outputs)
        predictions_dir = (
            Path(self.logger[0].save_dir)
            / self.logger[0].experiment_id
            / self.logger[0].run_id
            / "train"
            / "predictions"
        )
        predictions_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            prediction,
            predictions_dir / f"prediction-{self.logger[0].run_id}.joblib",
        )
    
    def validation_step(self, batch, batch_idx):
        y_hat, y, loss, outputs = self.step(batch)

        # log train metrics
        auroc = self.train_auroc(y_hat, y.type(torch.long))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in zip(["loss", "val_auroc"], [loss, auroc]):
            outputs[k] = v

        return outputs

    def validation_epoch_end(self, training_step_outputs):
        pass
    
    def test_step(self, batch, batch_idx):
        y_hat, y, loss, outputs = self.step(batch)

        # log train metrics
        auroc = self.train_auroc(y_hat, y.type(torch.long))
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in zip(["loss", "test_auroc"], [loss, auroc]):
            outputs[k] = v

        return outputs 

    def test_epoch_end(self, test_step_outputs):
        prediction = self.process_step_outputs(step_outputs=test_step_outputs)
        predictions_dir = (
            Path(self.logger[0].save_dir)
            / self.logger[0].experiment_id
            / self.logger[0].run_id
            / "test"
            / "predictions"
        )
        predictions_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            prediction,
            predictions_dir / f"prediction-{self.logger[0].run_id}.joblib",
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer