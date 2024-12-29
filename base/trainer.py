import time
import copy
import os
from tqdm import tqdm

import numpy as np
import torch
from torch import nn, optim

from base.scheduler import GradualWarmupScheduler
from base.utils import ensure_dir


class GenericTrainer(object):
    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.model_name = kwargs['model_name']
        self.model = kwargs['models'].to(self.device)
        self.save_path = kwargs['save_path']
        self.fold = kwargs['fold']
        self.min_epoch = kwargs['min_epoch']
        self.max_epoch = kwargs['max_epoch']
        self.start_epoch = 0
        self.early_stopping = kwargs['early_stopping']
        self.early_stopping_counter = self.early_stopping
        self.scheduler = kwargs['scheduler']
        self.learning_rate = kwargs['learning_rate']
        self.min_learning_rate = kwargs['min_learning_rate']
        self.patience = kwargs['patience']
        # IMPORTANT: This should be `nn.CrossEntropyLoss()` for 2-logit outputs
        self.criterion = kwargs['criterion']  
        self.factor = kwargs['factor']
        self.verbose = kwargs['verbose']
        self.milestone = kwargs['milestone']
        self.load_best_at_each_epoch = kwargs['load_best_at_each_epoch']

        self.optimizer, self.scheduler = None, None

    def train(self, **kwargs):
        """Training mode loop."""
        kwargs['train_mode'] = True
        self.model.train()
        loss, result_dict = self.loop(**kwargs)
        return loss, result_dict

    def validate(self, **kwargs):
        """Validation mode loop."""
        kwargs['train_mode'] = False
        with torch.no_grad():
            self.model.eval()
            loss, result_dict = self.loop(**kwargs)
        return loss, result_dict

    def test(self, checkpoint_controller, predict_only=0, **kwargs):
        """Testing mode loop."""
        kwargs['train_mode'] = False
        with torch.no_grad():
            self.model.eval()

            if predict_only:
                self.predict_loop(**kwargs)
            else:
                loss, result_dict = self.loop(**kwargs)
                # Save test results to CSV
                checkpoint_controller.save_log_to_csv(
                    kwargs['epoch'],
                    mean_train_record=None,
                    mean_validate_record=None,
                    test_record=result_dict['overall']
                )
                return loss, result_dict

    def fit(self, **kwargs):
        raise NotImplementedError

    def loop(self, **kwargs):
        raise NotImplementedError

    def predict_loop(self, **kwargs):
        raise NotImplementedError

    def get_parameters(self):
        """Return only parameters that require grad."""
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        return params_to_update


class GenericVideoTrainer(GenericTrainer):
    """
    Example binary classification trainer using a model with 2-logit output.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = kwargs['batch_size']

        # For checkpoint
        self.fit_finished = False
        self.fold_finished = False
        self.resume = False
        self.time_fit_start = None

        # Track training & validation losses
        self.train_losses = []
        self.validate_losses = []
        self.csv_filename = None

        # We'll track 'best_epoch_info' with best accuracy
        self.best_epoch_info = None

    def fit(self, dataloader_dict, checkpoint_controller, parameter_controller):
        if self.verbose:
            print("------")
            print("Starting training, on device:", self.device)

        self.time_fit_start = time.time()
        start_epoch = self.start_epoch

        # Initialize best info if none
        if self.best_epoch_info is None:
            self.best_epoch_info = {
                'model_weights': copy.deepcopy(self.model.state_dict()),
                'loss': 1e10,
                'accuracy': 0.0,
                'epoch': 0
            }

        for epoch in range(start_epoch, self.max_epoch):

            if self.fit_finished:
                if self.verbose:
                    print("\nEarly Stop!\n")
                break

            improvement = False

            # Possibly release more parameters / check LR
            if epoch in self.milestone or parameter_controller.get_current_lr() < self.min_learning_rate:
                parameter_controller.release_param(self.model.spatial, epoch)
                if parameter_controller.early_stop:
                    break

                # Load best so far
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            time_epoch_start = time.time()

            if self.verbose:
                num_layers_updating = len(self.optimizer.param_groups[0]['params'])
                print(f"There are {num_layers_updating} layers to update.")

            # Training step
            train_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            train_loss, train_record_dict = self.train(**train_kwargs)

            # Validation step
            validate_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            validate_loss, validate_record_dict = self.validate(**validate_kwargs)

            # Track losses
            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)

            # Instead of CCC, we track 'accuracy'
            val_acc = validate_record_dict['overall']['accuracy']
            if val_acc > self.best_epoch_info['accuracy']:
                # Save best model
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_path, "model_state_dict.pth"))
                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'accuracy': val_acc,
                    'epoch': epoch
                }

            if self.verbose:
                time_elapsed = time.time() - time_epoch_start
                best_epoch = int(self.best_epoch_info['epoch']) + 1
                print(f"\nFold {self.fold} Epoch {epoch + 1} in {time_elapsed:.0f}s"
                      f" || Train loss={train_loss:.3f} | Val loss={validate_loss:.3f}"
                      f" | LR={self.optimizer.param_groups[0]['lr']:.1e}"
                      f" | Release_count={parameter_controller.release_count}"
                      f" | best={best_epoch}"
                      f" | improvement={improvement}-{self.early_stopping_counter}")
                print("Train overall:", train_record_dict['overall'])
                print("Val overall:", validate_record_dict['overall'])
                print("------")

            # Save log to CSV
            checkpoint_controller.save_log_to_csv(
                epoch, train_record_dict['overall'], validate_record_dict['overall']
            )

            # Early stopping
            if self.early_stopping and epoch > self.min_epoch:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True

            # Scheduler step
            self.scheduler.step(metrics=validate_loss, epoch=epoch)
            self.start_epoch = epoch + 1

            if self.load_best_at_each_epoch:
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            # Save checkpoint
            checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.fit_finished = True
        # Final checkpoint
        checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        # Load best
        self.model.load_state_dict(self.best_epoch_info['model_weights'])

    def loop(self, **kwargs):
        """
        Loop for either train or validate.
        We'll compute epoch_loss & accuracy for 2-logit classification.
        """
        dataloader_dict = kwargs['dataloader_dict']
        epoch = kwargs['epoch']
        train_mode = kwargs['train_mode']

        if train_mode:
            dataloader = dataloader_dict['train']
        elif epoch is None:
            # Could be a test partition or 'extra'
            dataloader = dataloader_dict['extra']
        else:
            dataloader = dataloader_dict['validate']

        running_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        for batch_idx, (X, trials, lengths, indices) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Warm-up if needed
            if train_mode:
                # e.g. self.scheduler.warmup_lr(...)
                pass

            # Prepare inputs
            inputs = {}
            labels = None
            for feature, tensor_data in X.items():
                # We assume 'label' is the classification label
                if feature == "label":
                    labels = tensor_data.to(self.device)
                else:
                    inputs[feature] = tensor_data.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            # CrossEntropyLoss expects (batch_size, 2) outputs and (batch_size,) labels in {0,1}
            loss = self.criterion(outputs, labels)

            # Update running stats
            batch_size_now = labels.size(0)
            running_loss += loss.item() * batch_size_now
            total_samples += batch_size_now

            # Backward pass & update
            if train_mode:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Compute predictions for accuracy
            preds = torch.argmax(outputs, dim=1)  # shape (batch_size,)
            correct_predictions += (preds == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        result_dict = {
            'overall': {
                'accuracy': epoch_accuracy
            }
        }

        return epoch_loss, result_dict

    def predict_loop(self, **kwargs):
        """
        If you want to generate predictions only, e.g. for test partition.
        """
        partition = kwargs['partition']
        dataloader = kwargs['dataloader_dict'][partition]

        all_preds = []
        all_trials = []

        for batch_idx, (X, trials, lengths, indices) in tqdm(enumerate(dataloader), total=len(dataloader)):

            inputs = {}
            for feature, tensor_data in X.items():
                if feature == "label":
                    continue
                inputs[feature] = tensor_data.to(self.device)

            # Inference
            outputs = self.model(inputs)
            # Decide predicted label
            preds = torch.argmax(outputs, dim=1)  # shape (batch_size,)
            all_preds.extend(preds.cpu().numpy())
            all_trials.extend(trials)

        print(f"Prediction loop done for partition={partition}. #Predictions={len(all_preds)}")
        # At this point you can save or log all_preds / all_trials as needed.
