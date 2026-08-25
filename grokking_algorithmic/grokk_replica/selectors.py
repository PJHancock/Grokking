import torch
import torch.nn.functional as F

from grokking_mnist.Bayesian_models.BayesNet import KFCALLAWrapper


class BayesianSelector:
    def __init__(
        self,
        model,
        num_effective_data=200,
        prior_precision=10.0,
        n_f_samples=64,
        momentum=0.99,
        last_layer_name="last_layer",
    ):
        self.model = KFCALLAWrapper(
            net=model,
            num_effective_data=num_effective_data,
            prior_precision=prior_precision,
            n_f_samples=n_f_samples,
            momentum=momentum,
            last_layer_name=last_layer_name,
        )
        self.device = next(model.parameters()).device
        self.model.to(self.device)

    def select_points(self, inputs, targets, num_to_select):
        self.model.train()
        with torch.no_grad():
            _ = self.model(inputs, targets=targets)

        self.model.eval()
        with torch.no_grad():
            f_samples, _, _, _ = self.model(inputs, selection_pass=True)
            f_samples_flat = f_samples.flatten(0, 1)
            targets_repeated = targets.repeat_interleave(f_samples.shape[1], dim=0)
            surrogate_loss = F.cross_entropy(f_samples_flat, targets_repeated, reduction="none")
            surrogate_loss = surrogate_loss.view(f_samples.shape[0], f_samples.shape[1]).mean(dim=1)

            mean_probs = f_samples.softmax(-1).mean(dim=1).clamp_min(1e-8)
            predictive_entropy = -(mean_probs * mean_probs.log()).sum(dim=1)
            hardness = F.cross_entropy(mean_probs.log(), targets, reduction="none")
            scores = surrogate_loss + predictive_entropy + hardness

        _, selected = torch.topk(scores, min(num_to_select, inputs.shape[0]))
        return selected
