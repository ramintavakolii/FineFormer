import torch
import torch.nn as nn


def save_best_model(model, optimizer, val_accuracy, epoch, fold_num, checkpoint_path):
    """Save model checkpoint"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
        'epoch': epoch,
        'fold_num': fold_num
    }, checkpoint_path)


def load_best_model(model, model_file_path, device, load_optimizer=False, optimizer=None, freeze_encoder=False):
    """Load model checkpoint and optionally optimizer state"""

    checkpoint = torch.load(model_file_path, map_location=device, weights_only=False)
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    do_freeze = freeze_encoder
    if do_freeze:
        model = globals()['freeze_encoder'](model)

    # Load optimizer state if requested
    if load_optimizer and optimizer is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('val_accuracy', 0.0)


def load_pretrained_model(model, pretrain_model_path, device, reinit_classifier=True, freeze_encoder=False):
    """Load pretrained model and reinitialize classifier if needed"""

    checkpoint = torch.load(pretrain_model_path, map_location=device)
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()

    if reinit_classifier:
        # === Case 1: Load only encoder /// Works even if classifier architecture differs===
        encoder_keys_to_load = {k: v for k, v in pretrained_dict.items() if not k.startswith('classifier.')}
        model_dict.update(encoder_keys_to_load)
        model.load_state_dict(model_dict)

        # Reinitialize classifier manually
        for module in model.classifier.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        print(f"✓ Loaded encoder from pretrained model")
        print(f"✓ Reinitialized classifier weights")

    else:
        # === Case 2: Load full model (encoder + classifier) ===
        model.load_state_dict(pretrained_dict)
        print(f"✓ Loaded encoder and classifier weights from pretrained model")

    do_freeze = freeze_encoder
    if do_freeze:
        model = globals()['freeze_encoder'](model)
    return model


def freeze_encoder(model):
    """Freeze all encoder parameters for fine-tuning"""
    frozen_params = 0
    for name, param in model.named_parameters():
        if not name.startswith('classifier'):
            param.requires_grad = False
            frozen_params += 1

    classifier_tensors = sum(1 for name, p in model.named_parameters()
                              if name.startswith('classifier') and p.requires_grad)

    print(f"✓ Frozen {frozen_params} encoder parameter tensors")
    print(f"✓ Classifier remains trainable ({classifier_tensors} tensors)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"→ Total parameters: {total_params:,}")
    print(f"→ Trainable parameters: {trainable_total:,} ({trainable_total/total_params*100:.1f}%)")

    return model


def unfreeze_encoder(model):
    """Unfreeze all encoder parameters for fine-tuning"""
    unfrozen_params = 0
    for name, param in model.named_parameters():
        if not name.startswith('classifier'):
            param.requires_grad = True
            unfrozen_params += 1

    print(f"✓ Unfrozen {unfrozen_params} encoder parameters")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"→ Total parameters: {total_params:,}")
    print(f"→ Trainable parameters: {trainable_total:,} ({trainable_total/total_params*100:.1f}%)")

    return model