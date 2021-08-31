from .base_lightning_data_module import BaseLightningDataModule
from typing import Optional, List
from loguru import logger
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from sklearn.preprocessing import LabelEncoder
from classic_algos.utils.data import SlideFeaturesDataset
from classic_algos.nn import AutoEncoder
from classic_algos.utils.data.slides import fit_auto_encoder
from classic_algos.transforms import Encode


class HRDLightningDataModule(BaseLightningDataModule):
    def __init__(
        self,
        features_dir: str,
        clinic_data_file_train: str,
        wsi_dir: str,
        target_col: str,
        task: str,
        batch_size: int = 32,
        clinic_data_file_test: str = None,
        metadata_cols: List[str] = None,
        n_tiles: Optional[int] = None,
        transform: Optional[torch.nn.Module] = None,
        split_data: str = "by_slide",
        encode: bool = True,
        path_to_autoencoder: Optional[str] = None,
        autoencoder_save_file: Optional[str] = None
    ):
        super().__init__(
            features_dir=features_dir,
            clinic_data_file_train=clinic_data_file_train,
            wsi_dir=wsi_dir,
            target_col=target_col,
            task=task,
            batch_size=batch_size,
            clinic_data_file_test=clinic_data_file_test,
            metadata_cols=metadata_cols,
            n_tiles=n_tiles,
            transform=transform,
            split_data=split_data,
        )
        self.path_to_autoencoder = path_to_autoencoder
        self.transform = None
        self.encode = encode
        self.autoencoder_save_file = autoencoder_save_file

    def prepare_data(self):
        self.base_data_engineering()

        self.clinic_data_train = self.clinic_data_train.dropna(
            subset=["timeOfSample"]
        ).reset_index(drop=True)

        if self.split_data == "earliest_slide":
            self.clinic_data_train = (
                self.clinic_data_train.groupby("patient_id").apply(
                    lambda items: items[
                        items["timeOfSample"] == items["timeOfSample"].min()
                    ].iloc[0]
                )
            ).reset_index(drop=True)
        elif self.split_data == "latest_slide":
            self.clinic_data_train = (
                self.clinic_data_train.groupby("patient_id").apply(
                    lambda items: items[
                        items["timeOfSample"] == items["timeOfSample"].max()
                    ].iloc[0]
                )
            ).reset_index(drop=True)
        elif self.split_data == "first":
            self.clinic_data_train = (
                self.clinic_data_train.groupby("patient_id").first().reset_index()
            )
        else:
            raise NotImplementedError(
                "Only support 'earliest_slide' and 'latest_silde' parameter"
            )

        self.clinic_data_train = self.clinic_data_train.dropna(subset=[self.target_col]).reset_index(drop=True)
        y = self.clinic_data_train[self.target_col].values
        # Target in dataset are strings.
        # We encode them to int with a LabelEncoder
        label_encoder = LabelEncoder()
        self.clinic_data_train[self.target_col] = label_encoder.fit_transform(
            y.ravel()
        ).astype(np.float32)
        self.data_to_stratify = self.clinic_data_train[self.target_col].values

        if self.encode:
            logger.info("Encode hyperparameter set to True")
            auto_encoder = AutoEncoder(
                in_features=2048, hidden=[256], bias=True
            )
            if self.path_to_autoencoder is not None:
                logger.info(
                    f"Loading an existing autoencoder: {self.path_to_autoencoder}"
                )
                auto_encoder.load_state_dict(torch.load(self.path_to_autoencoder))
                transform = Encode(auto_encoder.encoder)
            else:
                # We assume that the auto encoder was not fitted beforehand.
                logger.info("Fitting a new autoencoder.")
                fit_auto_encoder(
                    self.clinic_data_train.features_path.values,
                    auto_encoder=auto_encoder,
                    tiling_tool_format=True,
                    max_tiles_seen=100_000,
                    num_workers=0,
                )
                torch.save(auto_encoder.state_dict(), self.autoencoder_save_file)
                logger.info(f"Autoencoder saved to {str(self.autoencoder_save_file)}")
            self.transform = Encode(auto_encoder.encoder)

    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            metadata_train = None
            if self.metadata_cols is not None:
                metadata_train = self.clinic_data_train[self.metadata_cols].to_dict(
                    orient="records"
                )

            self.dataset_train = SlideFeaturesDataset(
                features=self.clinic_data_train["features_path"].values,
                labels=self.clinic_data_train[self.target_col].values,
                transform=self.transform,
                tiling_tool_format=True,
                metadata=metadata_train,
                n_tiles=self.n_tiles,
            )
        if stage == "test" or stage is None:
            metadata_test = None
            if self.metadata_cols is not None:
                metadata_test = self.clinic_data_test[self.metadata_cols].to_dict(
                    orient="records"
                )

            self.dataset_test = SlideFeaturesDataset(
                features=self.clinic_data_train["features_path"].values,
                labels=self.clinic_data_train[self.target_col].values,
                transform=self.transform,
                tiling_tool_format=True,
                metadata=metadata_test,
                n_tiles=self.n_tiles,
            )
