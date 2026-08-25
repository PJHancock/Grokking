import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import warnings
import numpy as np
import random
import math

mean = {
    "CIFAR10": [0.4914, 0.4822, 0.4465],
    "CIFAR10_LT": [0.4914, 0.4822, 0.4465],
    "CIFAR100": [0.5071, 0.4867, 0.4408],
    "CIFAR100_LT": [0.5071, 0.4867, 0.4408],
    "TinyImageNet": [0.4802, 0.4481, 0.3975],
    "MNIST": [0.1307, 0.1307, 0.1307],
    "fashionMNIST": [0.2860, 0.2860, 0.2860],
    "SVHN": [0.4416, 0.4461, 0.4718],
    "LSUN": [0.5084, 0.4706, 0.4341],
    "WEBVISION": [0.485, 0.456, 0.406],
}

std = {
    "CIFAR10": [0.2470, 0.2435, 0.2616],
    "CIFAR10_LT": [0.2470, 0.2435, 0.2616],
    "CIFAR100": [0.2675, 0.2565, 0.2761],
    "CIFAR100_LT": [0.2675, 0.2565, 0.2761],
    "TinyImageNet": [0.2302, 0.2265, 0.2262],
    "MNIST": [0.3081, 0.3081, 0.3081],
    "fashionMNIST": [0.3530, 0.3530, 0.3530],
    "SVHN": [0.2040, 0.2081, 0.2058],
    "LSUN": [0.2487, 0.2492, 0.2675],
    "WEBVISION": [0.229, 0.224, 0.225],
}

class KFCALLAWrapper(nn.Module):
    def __init__(self, net, num_effective_data, prior_precision, n_f_samples, last_layer_name="last_layer", momentum=0.99):
        super(KFCALLAWrapper, self).__init__()
        self.net = net
        self.num_effective_data = num_effective_data
        self.prior_precision = prior_precision
        self.n_f_samples = n_f_samples
        self.momentum = momentum

        self.input_features_of_last_layer = None
        target_net = self.net.module if isinstance(self.net, torch.nn.DataParallel) else self.net
        last_layer = target_net
        for attr in last_layer_name.split("."):
            last_layer = getattr(last_layer, attr)
        self.fhook = last_layer.register_forward_hook(self.forward_hook())

        self.register_buffer("num_data", torch.zeros(1))
        self.register_buffer("A", torch.empty(0))
        self.register_buffer("G", torch.empty(0))
        self.register_buffer("G2", torch.empty(0))

    def _extract_logits(self, out):
        if isinstance(out, tuple):
            out = out[0]
        if out.dim() > 2:
            out = out[:, -1, :]
        return out

    def _maybe_initialize_statistics(self, out):
        if self.A.numel() != 0:
            return
        if self.input_features_of_last_layer is None:
            raise RuntimeError("Last-layer features were not captured before KFCA initialization.")

        feature_dim = self.input_features_of_last_layer.shape[-1]
        out_dim = out.shape[-1]
        self.A = torch.zeros(feature_dim, feature_dim, device=out.device, dtype=out.dtype)
        self.G = torch.zeros(out_dim, out_dim, device=out.device, dtype=out.dtype)
        self.G2 = torch.zeros(out_dim, out_dim, device=out.device, dtype=out.dtype)

    def forward_hook(self):
        def hook(module, input, output):
            features = input[0]
            if isinstance(features, tuple):
                features = features[0]
            if features.dim() > 2:
                features = features[:, -1, :]
            self.input_features_of_last_layer = features
        return hook

    def forward(self, x, **kwargs):
        selection_pass = kwargs.get('selection_pass', False)
        y = kwargs.get('targets', kwargs.get('y', None))

        bs = x.shape[0]
        if selection_pass:
            self.net.apply(_freeze)
        out = self._extract_logits(self.net(x))
        self._maybe_initialize_statistics(out)

        if selection_pass:
            self.net.apply(_unfreeze)

            if self.num_data.item() == 0:
                return out[:, None, :], out, None, None

            with torch.no_grad():
                V = math.sqrt(self.num_effective_data) * self.A
                V = V.clone()
                V.diagonal().add_(math.sqrt(self.prior_precision))
                L_V = psd_safe_cholesky(V)

                U = math.sqrt(self.num_effective_data) * self.G
                U = U.clone()
                U.diagonal().add_(math.sqrt(self.prior_precision))
                L_U = psd_safe_cholesky(U)

                V_inv = torch.cholesky_inverse(L_V)
                stds = (self.input_features_of_last_layer @ V_inv * self.input_features_of_last_layer).sum(-1).clamp(min=1e-6).sqrt()
                out_dim = out.shape[-1]
                I = torch.eye(out_dim, device=out.device, dtype=out.dtype)
                L_U_T_inv = torch.linalg.solve_triangular(L_U.T, I, upper=True)

                L_f = stds.view(-1, 1, 1) * L_U_T_inv

                eps = torch.randn((bs, self.n_f_samples, out_dim), device=out.device, dtype=out.dtype)
                f_samples = out[:, None, :] + eps @ L_f
                return f_samples, out, None, None
        elif self.training:
            assert y is not None, "Targets must be provided during training"
            with torch.no_grad():
                feature_cov = self.input_features_of_last_layer.T @ self.input_features_of_last_layer / bs
                if self.num_data.item() == 0:
                    self.A.data.copy_(feature_cov)
                else:
                    self.A.mul_(self.momentum).add_(feature_cov, alpha=1 - self.momentum)

                prob = out.softmax(-1)
                grad = prob - F.one_hot(y, out.shape[-1]).to(prob.dtype)
                grad_cov = grad.T @ grad / bs
                if self.num_data.item() == 0:
                    self.G.data.copy_(grad_cov)
                else:
                    self.G.mul_(self.momentum).add_(grad_cov, alpha=1 - self.momentum)

                self.num_data.add_(bs)

        return out

class CLIPZeroShotClassifier(nn.Module):
    def __init__(self, classnames, template, dataset, arch, tau):
        super(CLIPZeroShotClassifier, self).__init__()
        clip = _load_clip()
        clip_model, preprocess = clip.load(arch, jit=False)
        clip_model.eval()
        self.clip_model = clip_model
        clip_weights = clip_classifier(classnames, template, clip_model, clip)
        self.register_buffer('clip_weights', clip_weights)            

        self.register_buffer('old_mean', torch.Tensor(mean[dataset]))
        self.register_buffer('old_std', torch.Tensor(std[dataset]))
        
        self.register_buffer('new_mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]))
        self.register_buffer('new_std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
        self.input_size = preprocess.transforms[0].size
        self.tau = tau
    
    @torch.no_grad()
    def forward(self, inputs):
        inputs = inputs.mul(self.old_std.view(-1, 1, 1)).add(self.old_mean.view(-1, 1, 1))
        if inputs.shape[1] == 1:
            # Convert grayscale to RGB
            inputs = inputs.repeat(1, 3, 1, 1)
        if inputs.shape[2] != self.input_size:
            inputs = F.interpolate(inputs, self.input_size, mode='bicubic')
        inputs = inputs.sub(self.new_mean.view(-1, 1, 1)).div(self.new_std.view(-1, 1, 1))

        input_features = self.clip_model.encode_image(inputs)
        clip_logits = self.tau * input_features @ self.clip_weights
        return clip_logits
    

def _load_clip():
    try:
        import clip
    except ImportError as exc:
        raise ImportError("The CLIP dependency is only required for CLIPZeroShotClassifier.") from exc
    return clip


def clip_classifier(classnames, template, clip_model, clip_module=None):
    clip_module = _load_clip() if clip_module is None else clip_module
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip_module.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def _freeze(m):
    if isinstance(m, (nn.BatchNorm2d)):
        m.track_running_stats = False

def _unfreeze(m):
    if isinstance(m, (nn.BatchNorm2d)):
        m.track_running_stats = True

def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.linalg.cholesky(A, upper=upper, out=out)
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise ValueError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(10):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        # return torch.randn_like(A).tril()
        raise e
