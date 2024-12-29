import os
import copy
import torch
import torch.nn as nn
import numpy as np

from base.experiment import GenericExperiment
from base.utils import load_pickle
from trainer import Trainer  # This should be your updated trainer that handles 2-logit classification
from dataset import DataArranger, Dataset
from base.checkpointer import Checkpointer
from models.model import RCMA, CAN
from base.parameter_control import ResnetParamControl


class Experiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # Extra hyperparams
        self.release_count = args.release_count
        self.gradual_release = args.gradual_release
        self.milestone = args.milestone
        self.backbone_mode = "ir"
        self.min_num_epochs = args.min_num_epochs
        self.num_epochs = args.num_epochs
        self.early_stopping = args.early_stopping
        self.load_best_at_each_epoch = args.load_best_at_each_epoch

        self.num_heads = args.num_heads
        self.modal_dim = args.modal_dim
        self.tcn_kernel_size = args.tcn_kernel_size

    def prepare(self):
        """Prepare the data arranger, load or compute mean/std if needed."""
        self.config = self.get_config()

        self.feature_dimension = self.get_feature_dimension(self.config)
        self.multiplier = self.get_multiplier(self.config)
        self.time_delay = self.get_time_delay(self.config)

        # Decide which modalities you want (e.g., ['video','logmel'])
        self.get_modality()

        # For binary classification, we do NOT have continuous_label_dim
        self.dataset_info = load_pickle(os.path.join(self.dataset_path, "dataset_info.pkl"))
        self.data_arranger = self.init_data_arranger()

        if self.calc_mean_std:
            self.calc_mean_std_fn()

        self.mean_std_dict = load_pickle(os.path.join(self.dataset_path, "mean_std_info.pkl"))

    def init_data_arranger(self):
        arranger = DataArranger(self.dataset_info, self.dataset_path, self.debug)
        return arranger

    def run(self):
        """
        The main loop: for each fold, create model/trainer, train, and optionally test.
        """

        # For binary classification with 2 logits, use CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()

        for fold in self.folds_to_run:

            # Example naming scheme (remove references to 'emotion' if you had them):
            save_path = os.path.join(self.save_path,
                                     f"{self.experiment_name}_{self.model_name}_{self.stamp}_fold{fold}_seed{self.seed}")
            os.makedirs(save_path, exist_ok=True)

            checkpoint_filename = os.path.join(save_path, "checkpoint.pkl")

            # Build model
            model = self.init_model()

            # Build dataloaders for train/val/test
            dataloaders = self.init_dataloader(fold)

            # Trainer arguments, referencing your 2-logit classification Trainer
            trainer_kwargs = {
                'device': self.device,
                'model_name': self.model_name,
                'models': model,
                'save_path': save_path,
                'fold': fold,
                'min_epoch': self.min_num_epochs,
                'max_epoch': self.num_epochs,
                'early_stopping': self.early_stopping,
                'scheduler': self.scheduler,
                'learning_rate': self.learning_rate,
                'min_learning_rate': self.min_learning_rate,
                'patience': self.patience,
                'batch_size': self.batch_size,
                'criterion': criterion,                 # CrossEntropyLoss
                'factor': self.factor,
                'verbose': True,
                'milestone': self.milestone,
                'metrics': self.config['metrics'],      # e.g. ['accuracy']
                'load_best_at_each_epoch': self.load_best_at_each_epoch,
                'save_plot': self.config['save_plot']   # or False if not plotting
            }

            trainer = Trainer(**trainer_kwargs)

            # Parameter controller for gradual layer release (if used)
            parameter_controller = ResnetParamControl(
                trainer,
                gradual_release=self.gradual_release,
                release_count=self.release_count,
                backbone_mode=["visual", "audio"]
            )

            checkpoint_controller = Checkpointer(
                checkpoint_filename, trainer, parameter_controller, resume=self.resume
            )

            # If we want to resume training
            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger(self.args, self.config)

            # If not already finished, train
            if not trainer.fit_finished:
                trainer.fit(dataloaders, parameter_controller=parameter_controller,
                            checkpoint_controller=checkpoint_controller)

            # Optionally do a “test” pass (like inference)
            # Example: use partition='extra' or however your dataset organizes test data
            test_kwargs = {'dataloader_dict': dataloaders, 'epoch': None, 'partition': 'extra'}
            trainer.test(checkpoint_controller, predict_only=1, **test_kwargs)

    def init_dataset(self, data, mode, fold):
        """
        Build your dataset for classification.
        No continuous_label_dim is used, so remove that parameter.
        """
        dataset = Dataset(
            data_list=data,
            modality=self.modality,
            multiplier=self.multiplier,
            feature_dimension=self.feature_dimension,
            window_length=self.window_length,
            mode=mode,
            mean_std=self.mean_std_dict[fold][mode],
            time_delay=self.time_delay
        )
        return dataset

    def init_model(self):
        """
        Initialize RCMA or CAN with 2 outputs for binary classification.
        """
        self.init_randomness()
        modality = [m for m in self.modality if "label" not in m]  # e.g. ['video','logmel']

        if self.model_name == "RCMA":
            model = RCMA(
                backbone_settings=self.config['backbone_settings'],
                modality=modality,
                example_length=self.window_length,
                kernel_size=self.tcn_kernel_size,
                tcn_channel=self.config['tcn']['channels'],
                modal_dim=self.modal_dim,
                num_heads=self.num_heads,
                root_dir=self.load_path,
                device=self.device
            )
            # For binary classification, ensure output_dim=2
            model.init()  # Make sure RCMA sets self.output_dim = 2 inside .init()
        elif self.model_name == "CAN":
            model = CAN(
                root_dir=self.load_path,
                modalities=modality,
                tcn_settings=self.config['tcn_settings'],
                backbone_settings=self.config['backbone_settings'],
                output_dim=2,  # 2 logits for binary
                device=self.device
            )
        return model

    def get_modality(self):
        """
        Specify which modalities you want. e.g.:
        self.modality = ['video', 'logmel']
        """
        pass

    def get_config(self):
        from configs import config
        return config

    def get_selected_continuous_label_dim(self):
        """
        For binary classification, we remove or skip continuous label dimension logic.
        Just return an empty list or do nothing.
        """
        return []
