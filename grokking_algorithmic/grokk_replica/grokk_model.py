import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .transformer import Transformer
    from .utils import causal_attn_mask, parameter_norm
except ImportError:  # pragma: no cover - supports direct script execution
    from transformer import Transformer
    from utils import causal_attn_mask, parameter_norm

class GrokkModel(nn.Module):
    def __init__(self, transformer_config, vocab_size, output_size, device):
        super(GrokkModel, self).__init__()
        self.transformer_config = transformer_config
        self.transformer = Transformer(**transformer_config, vocab_size=vocab_size, output_size=output_size)
        self.device = device
        self.last_layer = self.transformer.output

    def _build_attn_mask(self, x):
        return causal_attn_mask(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1, 1).to(x.device)

    def forward(self, x):
        attn_mask = self._build_attn_mask(x)
        predictions, attns, _ = self.transformer(x, attn_mask)
        return predictions, attns

    def feat_nograd_forward(self, x):
        attn_mask = self._build_attn_mask(x)
        with torch.no_grad():
            hidden_states, _, _ = self.transformer.forward_hidden(x, attn_mask)
        features = hidden_states[:, -1, :]
        logits = self.transformer.output(features)
        return logits, features

    def get_loss(self, x, y):
        predictions, attns = self(x)
        # print(torch.argmax(predictions[:, -1, :], dim=-1), x[:, -1])
        loss = F.cross_entropy(predictions[:, -1, :], y)
        accuracy = (torch.argmax(predictions[:, -1, :], dim=-1) == y).float().mean()
        attn_entropies = sum([-(attn * torch.log(attn+1e-7)).sum(dim=-1).mean().item() for attn in attns]) / len(attns)
        param_norm = parameter_norm(self)
        return loss, {'loss': (loss.item(), x.shape[0]), 'accuracy': (accuracy.item(), x.shape[0]), 
                      'attn_entropy': (attn_entropies, len(attns)*x.shape[0]*(x.shape[1]-1)), 'param_norm': (param_norm, 1)}
