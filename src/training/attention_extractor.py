import torch
import os

class AttentionExtractor:
    def __init__(self, model, val_loader, device, fold_num, save_dir, average_heads=True):
        """
        Extracts and saves attention weights from the validation set.
        """
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.fold_num = fold_num
        self.save_dir = save_dir
        self.average_heads = average_heads  # Whether to average attention heads inside the model

        self.attention_data = {
            "fold": fold_num,
            "samples": []
        }

        os.makedirs(save_dir, exist_ok=True)

    def forward_with_attention(self, inputs):
        """
        Forward pass with attention extraction.
        The model internally handles head averaging based on self.average_heads.
        """
        logits, attn_weights = self.model(
            inputs,
            return_attention=True,
            average_attn_weights=self.average_heads
        )
        return logits, attn_weights

    def process_attention_batch(self, inputs, labels, attn_weights, probs, preds, sample_idx_start):

        for i in range(inputs.size(0)):
            sample_data = {
                "sample_idx": sample_idx_start + i,
                "input": inputs[i].cpu(),
                "label": labels[i].cpu().item(),
                "pred": preds[i].cpu().item(),
                "probs": probs[i].cpu(),
                "attention_weights": [attn[i].cpu() for attn in attn_weights]
                # attn[i] shape depends on average_heads:
                #   if average_heads=True → (seq_len, seq_len)
                #   if average_heads=False → (num_heads, seq_len, seq_len)
            }
            self.attention_data["samples"].append(sample_data)

    def save_attention_to_file(self):
        save_path = os.path.join(self.save_dir, f"fold_{self.fold_num}_attention.pt")
        torch.save(self.attention_data, save_path)
        print(f"Saved attention weights for Fold {self.fold_num} → {save_path}")
        return save_path

    def run(self):
        print("Extracting attention weights...")
        self.model.eval()

        sample_idx = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward pass
                logits, attn_weights = self.forward_with_attention(inputs)

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                # store this batch's attention information
                self.process_attention_batch(inputs, labels, attn_weights, probs, preds, sample_idx)

                sample_idx += inputs.size(0)

        return self.save_attention_to_file()