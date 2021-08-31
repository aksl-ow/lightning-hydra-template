from typing import Optional, List
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
import torch

from classic_algos.utils.data import SlideFeaturesDataset
from torch.utils.data import DataLoader


class BaseLightningDataModule(pl.LightningDataModule):
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
    ):
        super().__init__()
        self.task = task
        self.features_dir = Path(features_dir)
        self.clinic_data_file_train = clinic_data_file_train
        self.clinic_data_file_test = clinic_data_file_test

        self.clinic_data_train = pd.read_csv(self.clinic_data_file_train, sep="\t")
        if self.clinic_data_file_test is not None:
            self.clinic_data_test = pd.read_csv(self.clinic_data_file_test, sep="\t")

        self.target_col = target_col
        self.metadata_cols = metadata_cols
        self.n_tiles = n_tiles
        self.batch_size = batch_size
        self.wsi_dir = Path(wsi_dir)
        self.transform = transform
        self.split_data = split_data
        self.data_to_stratify = None
            
    
    def base_data_engineering(self):
        self.clinic_data_train = self.clinic_data_train.dropna(
            subset=["slideId"]
        ).reset_index(drop=True)

        self.clinic_data_train["slide_name"] = self.clinic_data_train["slideId"].map(
            lambda slide_id: slide_id + ".tiff"
        )
        self.clinic_data_train["wsi_path"] = self.clinic_data_train["slide_name"].map(
            lambda name: str(self.wsi_dir / name)
        )
        self.clinic_data_train["features_path"] = self.clinic_data_train[
            "slide_name"
        ].map(lambda name: str(self.features_dir / name / f"features.npy"))
        features_that_exist = self.clinic_data_train["features_path"].map(
            lambda feature_path: Path(feature_path).exists()
        )
        self.clinic_data_train = self.clinic_data_train.loc[
            features_that_exist, :
        ].reset_index(drop=True)
        self.clinic_data_train = self.clinic_data_train.rename(
            columns={"slideId": "sample_id", "patientId": "patient_id"}
        )

    def prepare_data(self):
        raise NotImplementedError()

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

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        pass
