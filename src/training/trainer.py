
import os

import torch

from src.training.checkpointing import (
    save_best_model,
    load_best_model,
    load_pretrained_model,
    unfreeze_encoder,
)
from src.training.evaluator import ModelEvaluator


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, criterion, optimizer, scheduler, fold_num,
                 save_path, num_epochs=50, patience=20, initial_lr=3e-4,
                 warmup_scheduler=None, warmup_epochs=0, fine_tune=True,
                 reinit_classifier=True, freeze_encoder=False, unfreeze_epoch=None):
        """
        Enhanced Trainer with encoder freezing and two-stage training support.

        Important Args:
            reinit_classifier: If True, reinitialize classifier when loading pretrained model
            freeze_encoder: If True, freeze encoder weights during training
            unfreeze_epoch: Epoch at which to unfreeze encoder (for two-stage training)
            warmup_scheduler / warmup_epochs: used by HS Cycle 1; unused when warmup_epochs=0
        """

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fold_num = fold_num
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.patience = patience
        self.initial_lr = initial_lr
        self.warmup_scheduler = warmup_scheduler
        self.warmup_epochs = warmup_epochs
        self.fine_tune = fine_tune
        self.reinit_classifier = reinit_classifier
        self.freeze_encoder = freeze_encoder
        self.unfreeze_epoch = unfreeze_epoch

        # Internal state
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.start_epoch = 0

        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }

        # Paths
        self.pretrain_model_path = os.path.join(save_path, f"pretrain_model_fold_{fold_num}.pth")
        self.checkpoint_path = os.path.join(save_path, f"best_model_fold_{fold_num}.pth")
        self.training_history_path = os.path.join(save_path, f"train_history_fold_{fold_num}.pt")
        os.makedirs(save_path, exist_ok=True)

        # Decide how to initialize training
        self._initialize_training()

    def _initialize_training(self):
        """
        Decide whether to:
          - initialize from pretrained weights for fine-tuning
          - resume from a training checkpoint
          - or start fresh
        """
        if self.fine_tune and os.path.exists(self.pretrain_model_path):
            self._init_from_pretrained()
        elif os.path.exists(self.checkpoint_path):
            self._resume_from_checkpoint()
        else:
            self._start_fresh_training()

    def _init_from_pretrained(self):
        """
        Load encoder from a pretrained model and prepare for fine-tuning.
        """
        print("→ Found pretrained model checkpoint...")
        print("→ Fine-tuning model")

        try:
            # Load encoder-only / partial weights
            self.model = load_pretrained_model(
                self.model,
                self.pretrain_model_path,
                self.device,
                reinit_classifier=self.reinit_classifier,
                freeze_encoder=self.freeze_encoder
            )

            # Reset training state for fine-tuning
            self.start_epoch = 0
            self.best_val_acc = 0.0
            self.training_history = {
                'train_loss': [], 'val_loss': [],
                'train_acc': [], 'val_acc': []
            }

            # If encoder is frozen, restrict optimizer params
            if self.freeze_encoder:
                self._update_optimizer_for_frozen_encoder()

            print("→ Ready for fine-tuning")

        except Exception as e:
            print(f"→ Error loading pretrained model: {e}, starting fresh")
            self._start_fresh_training()

    def _resume_from_checkpoint(self):
        """
        Resume training from an existing best-model checkpoint and history.
        """
        print("→ Found training checkpoint...")
        print("→ Resuming training from checkpoint...")

        try:
            self.start_epoch, self.best_val_acc = load_best_model(
                self.model, self.checkpoint_path, self.device, load_optimizer=False,
                optimizer=None, freeze_encoder=self.freeze_encoder
            )
            print(f"→ Loaded best model from epoch {self.start_epoch} for final evaluation")

            # Load training history if available
            if os.path.exists(self.training_history_path):
                history_data = torch.load(self.training_history_path, weights_only=False)
                self.training_history = history_data

                # Check if encoder was already unfrozen
                if 'encoder_unfrozen_at_epoch' in history_data:
                    print("="*60, "\n", f"→ Encoder was already unfrozen at epoch {history_data['encoder_unfrozen_at_epoch']}", "\n", "="*60)

            print(f"→ Resuming from epoch {self.start_epoch}, "
                  f"best acc: {self.best_val_acc:.4f}")

        except Exception as e:
            print(f"→ Error loading checkpoint: {e}, starting fresh")
            self._start_fresh_training()

    def _start_fresh_training(self):
        self.start_epoch = 0
        self.best_val_acc = 0.0
        if hasattr(self.model, '_init_weights'):
            self.model.apply(self.model._init_weights)

        # Filter out unsupported arguments before creating optimizer
        valid_keys = {'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'}
        opt_kwargs = {k: v for k, v in self.optimizer.defaults.items() if k in valid_keys}

        # type(self.optimizer)(...) is equivalent to calling the optimizer’s constructor
        self.optimizer = type(self.optimizer)(
            self.model.parameters(),
            **opt_kwargs
        )

    def _update_optimizer_for_frozen_encoder(self):
        """Update optimizer to only include trainable parameters"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Recreate optimizer with only trainable parameters
        self.optimizer = type(self.optimizer)(
            trainable_params,
            lr=self.initial_lr,
            **{k: v for k, v in self.optimizer.defaults.items() if k != 'lr'}
        )

        print(f"→ Optimizer updated to include only {len(trainable_params)} trainable parameters")

    def _update_optimizer_for_unfreeze_encoder(self):
        """Update optimizer to include all parameters"""

        self.optimizer = type(self.optimizer)(
            self.model.parameters(),
            lr=self.initial_lr * 0.1,  # Use lower LR for fine-tuning
            **{k: v for k, v in self.optimizer.defaults.items() if k != 'lr'}
        )
        print(f"→ Optimizer updated with all parameters, LR reduced to {self.initial_lr * 0.1}")

    def _check_and_unfreeze_encoder(self, epoch):
        """Check if it's time to unfreeze the encoder"""
        if (epoch >= self.unfreeze_epoch and self.freeze_encoder):

            print(f"\n→ Epoch {epoch}: Unfreezing encoder for full fine-tuning")
            self.model = unfreeze_encoder(self.model)
            self.freeze_encoder = False

            # Update optimizer to include all parameters
            self._update_optimizer_for_unfreeze_encoder()

            # Save unfreezing info in history
            self.training_history['encoder_unfrozen_at_epoch'] = epoch

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).view(-1)  # (B,1) -> (B,)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)    # (B, 2)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Only clip gradients for parameters that require grad
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Binary-safe predictions
            if outputs.shape[1] == 1:
                preds = (torch.sigmoid(outputs) >= 0.5).long().view(-1)
            else:
                preds = torch.argmax(outputs, dim=1) # (B,)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader.dataset)
        train_acc = correct / total

        self.training_history['train_loss'].append(epoch_loss)
        self.training_history['train_acc'].append(train_acc)

        return epoch_loss, train_acc

    def validate_one_epoch(self):
        evaluator = ModelEvaluator(self.model, self.device, self.criterion)
        metrics = evaluator.evaluate(self.val_loader)

        val_acc = metrics['accuracy']
        val_loss = metrics.get('val_loss', None)

        self.training_history['val_acc'].append(val_acc)
        self.training_history['val_loss'].append(val_loss)

        return val_acc, val_loss

    def update_scheduler(self, val_acc, epoch):
        # Warmup first (HS Cycle 1 notebook behavior)
        if self.warmup_scheduler is not None and epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
            return

        # After warmup, use the main scheduler
        if self.scheduler is not None:
            self.scheduler.step(val_acc)

    def save_checkpoint_if_best(self, val_acc, epoch):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.patience_counter = 0
            save_best_model(self.model, self.optimizer, val_acc, epoch + 1, self.fold_num, self.checkpoint_path)
            torch.save(self.training_history, self.training_history_path)
            print(f"→ New best model saved with val_acc: {val_acc:.4f}")
        else:
            self.patience_counter += 1

    def _check_early_stopping(self):
        return self.patience_counter >= self.patience

    def run(self):
        print(f"Starting training for fold {self.fold_num}...")

        # Print training configuration
        if self.freeze_encoder:
            print(f"→ Encoder frozen, training classifier only")
            if self.unfreeze_epoch:
                print(f"→ Will unfreeze encoder at epoch {self.unfreeze_epoch}")

        for epoch in range(self.start_epoch, self.num_epochs):
            # Check if we should unfreeze encoder
            if(self.unfreeze_epoch is not None):
                self._check_and_unfreeze_encoder(epoch)

            # ---- Training ----
            train_loss, train_acc = self.train_one_epoch(epoch)

            # ---- Validation ----
            val_acc, val_loss = self.validate_one_epoch()

            # ---- Scheduler ----
            self.update_scheduler(val_acc, epoch)

            # ---- Logging ----
            print(f"Epoch {epoch+1:02d}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}")

            # ---- Checkpoint ----
            self.save_checkpoint_if_best(val_acc, epoch)

            # ---- Early stopping ----
            if self._check_early_stopping():
                print(f"→ Early stopping after {self.patience} epochs without improvement")
                break

        # Load best model
        if os.path.exists(self.checkpoint_path):
            start_epoch, _ = load_best_model(self.model, self.checkpoint_path, self.device)
            print(f"→ Loaded best model from epoch {start_epoch} for final evaluation")

        # Final evaluation
        final_eval = ModelEvaluator(self.model, self.device, criterion=self.criterion)
        final_metrics = final_eval.evaluate(self.val_loader)
        final_eval.summarize_metrics()

        return self.model, final_metrics, self.training_history
