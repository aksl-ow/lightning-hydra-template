from typing import List, Optional

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from src.utils import utils
from src.utils.dataset import pad_collate_fn, CVSplit

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.prepare_data()
    target = datamodule.data_to_stratify if config.crossval.stratified else None
    
    # Init Cross validation class
    cv = CVSplit(
        n_splits=config.crossval.n_splits,
        n_repeats=config.crossval.n_cv_run,
        stratified=config.crossval.stratified,
    )
    datamodule.setup(stage="fit")

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    for fold_i, (train_valid_dataset, test_dataset) in enumerate(cv(datamodule, target)):
        log.info(f"Training Fold: {fold_i + 1}/{config.crossval.n_splits * config.crossval.n_cv_run}")

        idx_train, idx_valid = train_test_split(
            range(len(train_valid_dataset)), 
            test_size=0.15, 
            stratify=train_valid_dataset.dataset.labels[train_valid_dataset.indices]
        )
        train_dataset = Subset(train_valid_dataset, idx_train)
        valid_dataset = Subset(train_valid_dataset, idx_valid)

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.datamodule.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=pad_collate_fn,
        )
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=config.datamodule.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=pad_collate_fn,
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=config.datamodule.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=pad_collate_fn,
        )
        trainer.fit(
            model=model, 
            train_dataloader=train_dataloader, 
            val_dataloaders=valid_dataloader
        )

        # Evaluate model on test set, using the best model achieved during training
        if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
            log.info("Starting testing!")
            trainer.test(model=model, test_dataloaders=test_dataloader)

        # Make sure everything closed properly
        log.info("Finalizing!")
        utils.finish(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

        # Print path to best checkpoint
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

        # Return metric score for hyperparameter optimization
        optimized_metric = config.get("optimized_metric")
        if optimized_metric:
            return trainer.callback_metrics[optimized_metric]
