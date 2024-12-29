import time
import copy
import os
import numpy as np
import torch
from torch import optim

from base.trainer import GenericVideoTrainer
from base.scheduler import MyWarmupScheduler  # or GradualWarmupScheduler


class Trainer(GenericVideoTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # We'll track best epoch info with accuracy instead of ccc
        self.best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10,
            'acc': -1.0,
            'epoch': 0,
        }

    def init_optimizer_and_scheduler(self, epoch=0):
        self.optimizer = optim.Adam(self.get_parameters(), lr=self.learning_rate, weight_decay=0.001)

        # If you want your scheduler to track higher = better (accuracy),
        # set "mode='max'". If you want to track lower = better (loss),
        # set "mode='min'", and update "best" below accordingly.
        self.scheduler = MyWarmupScheduler(
            optimizer=self.optimizer,
            lr=self.learning_rate,
            min_lr=self.min_learning_rate,
            best=self.best_epoch_info['acc'],  # If mode='max', pass best accuracy
            mode="max",
            patience=self.patience,
            factor=self.factor,
            num_warmup_epoch=self.min_epoch,
            init_epoch=epoch
        )

    def fit(self, dataloader_dict, checkpoint_controller, parameter_controller):
        if self.verbose:
            print("------")
            print("Starting training, on device:", self.device)

        self.time_fit_start = time.time()
        start_epoch = self.start_epoch

        for epoch in range(start_epoch, self.max_epoch):

            if self.fit_finished:
                if self.verbose:
                    print("\nEarly Stop!\n")
                break

            improvement = False

            # Possibly release parameters at certain milestones or if LR is small
            # (after min_epoch is reached). This logic is unchanged; remove if not needed.
            if (epoch in self.milestone 
                or parameter_controller.get_current_lr() < self.min_learning_rate
                and epoch >= self.min_epoch
                and self.scheduler.relative_epoch > self.min_epoch):
                
                parameter_controller.release_param(self.model.spatial, epoch)
                if parameter_controller.early_stop:
                    break

                # Load best so far
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            time_epoch_start = time.time()

            if self.verbose:
                num_layers_updating = len(self.optimizer.param_groups[0]['params'])
                print(f"There are {num_layers_updating} layers to update.")

            # TRAIN
            train_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            train_loss, train_record_dict = self.train(**train_kwargs)

            # VALIDATE
            validate_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            validate_loss, validate_record_dict = self.validate(**validate_kwargs)

            if validate_loss < 0:
                raise ValueError('Validate loss is negative, which is unexpected.')

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)

            # Suppose your loop returns:
            #   validate_record_dict['overall'] = {'accuracy': some_value}
            val_acc = validate_record_dict['overall']['accuracy']

            # Update scheduler's best if you're tracking 'acc' in mode='max'
            self.scheduler.best = self.best_epoch_info['acc']

            # Check improvement
            if val_acc > self.best_epoch_info['acc']:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_path, f"model_state_dict_{val_acc:.4f}.pth")
                )
                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'acc': val_acc,
                    'epoch': epoch
                }

            if self.verbose:
                elapsed = time.time() - time_epoch_start
                print(
                    f"\n Fold {self.fold:2} Epoch {epoch + 1:2} in {elapsed:.0f}s || "
                    f"Train loss={train_loss:.3f} | Val loss={validate_loss:.3f} | "
                    f"LR={self.optimizer.param_groups[0]['lr']:.1e} | "
                    f"Release_count={parameter_controller.release_count} | "
                    f"best={int(self.best_epoch_info['epoch']) + 1} | "
                    f"improvement={improvement}-{self.early_stopping_counter}"
                )
                print("Train Overall:", train_record_dict['overall'])
                print("Val Overall:", validate_record_dict['overall'])
                print("------")

            # Save logs to CSV
            checkpoint_controller.save_log_to_csv(
                epoch, train_record_dict['overall'], validate_record_dict['overall']
            )

            # Early stopping
            if self.early_stopping and self.scheduler.relative_epoch > self.min_epoch:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True

            # Step the scheduler with your chosen metric
            # If you want to reduce LR on plateau w.r.t. accuracy (higher is better),
            # pass `validate_loss` if you prefer to track loss, or `val_acc` if you prefer accuracy
            # but remember your MyWarmupScheduler is set to mode='max'.
            self.scheduler.step(metrics=val_acc, epoch=epoch)

            self.start_epoch = epoch + 1

            # Optionally re-load best each epoch
            if self.load_best_at_each_epoch:
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            # Save checkpoint
            checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.fit_finished = True
        checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        # Finally load the best weights
        self.model.load_state_dict(self.best_epoch_info['model_weights'])
