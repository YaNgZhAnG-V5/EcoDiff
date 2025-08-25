# Normalization Layer Hooks for FLUX Models
#
# Architecture Note: This module implements normalization layer pruning hooks.
# Future refactoring should consider creating a unified parent class for all
# hook types (attention, FFN, normalization) to improve code organization
# and reduce redundancy across the hooking system.

import logging
import os
from collections import OrderedDict
from functools import partial

import torch
from torch import nn

import re
from sdib.utils.utils import hard_concrete_distribution


class NormHooker:
    def __init__(
        self,
        pipeline: nn.Module,
        regex: str,
        dtype: torch.dtype,
        masking: str,
        dst: str,
        epsilon: float = 0.0,
        eps: float = 1e-6,
        use_log: bool = False,
        binary: bool = False,
    ):
        self.pipeline = pipeline
        self.net = pipeline.unet if hasattr(pipeline, "unet") else pipeline.transformer
        self.logger = logging.getLogger(__name__)
        self.dtype = dtype
        self.regex = regex
        self.hook_dict = {}
        self.masking = masking
        self.dst = dst
        self.epsilon = epsilon
        self.eps = eps
        self.use_log = use_log
        self.lambs = []
        self.lambs_module_names = []  # store the module names for each lambda block
        self.hook_counter = 0
        self.module_neurons = OrderedDict()
        self.binary = binary  # default, need to discuss if we need to keep this attribute or not

    def add_hooks_to_norm(self, hook_fn: callable):
        """
        Add forward hooks to every feed forward layer matching the regex
        :param hook_fn: a callable to be added to torch nn module as a hook
        :return: dictionary of added hooks
        """
        total_hooks = 0
        for name, module in self.net.named_modules():
            name_last_word = name.split(".")[-1]
            if "norm1" in name_last_word or ("norm" == name_last_word and len(name.split(".")) == 3):
                if re.match(self.regex, name):
                    hook_fn_with_name = partial(hook_fn, name=name)

                    if hasattr(module, "linear"):
                        actual_module = module.linear
                    else:
                        if isinstance(module, nn.Linear):
                            actual_module = module
                        else:
                            continue

                    hook = actual_module.register_forward_hook(hook_fn_with_name, with_kwargs=True)
                    self.hook_dict[name] = hook

                    # AdaLayerNormZero
                    if isinstance(actual_module, torch.nn.Linear):
                        self.module_neurons[name] = actual_module.out_features
                    else:
                        raise NotImplementedError(f"Module {name} is not implemented, please check")
                    self.logger.info(f"Adding hook to {name}, neurons: {self.module_neurons[name]}")
                    total_hooks += 1
        self.logger.info(f"Total hooks added: {total_hooks}")
        return self.hook_dict

    def add_hooks(self, init_value=1.0):
        hook_fn = self.get_norm_masking_hook(init_value)
        self.add_hooks_to_norm(hook_fn)
        # initialize the lambda
        self.lambs = [None] * len(self.hook_dict)
        # initialize the lambda module names
        self.lambs_module_names = [None] * len(self.hook_dict)

    def clear_hooks(self):
        """clear all hooks"""
        for hook in self.hook_dict.values():
            hook.remove()
        self.hook_dict.clear()

    def save(self, name: str = None):
        if name is not None:
            dst = os.path.join(os.path.dirname(self.dst), name)
        else:
            dst = self.dst
        dst_dir = os.path.dirname(dst)
        if not os.path.exists(dst_dir):
            self.logger.info(f"Creating directory {dst_dir}")
            os.makedirs(dst_dir)
        torch.save(self.lambs, dst)

    @property
    def get_lambda_block_names(self):
        return self.lambs_module_names

    def load(self, device, threshold):
        if os.path.exists(self.dst):
            self.logger.info(f"loading lambda from {self.dst}")
            self.lambs = torch.load(self.dst, weights_only=True, map_location=device)
            if self.binary:
                # set binary masking for each lambda by using clamp
                self.lambs = [(torch.relu(lamb - threshold) > 0).float() for lamb in self.lambs]
            else:
                self.lambs = [torch.clamp(lamb, min=0.0) for lamb in self.lambs]
            # self.lambs_module_names = [None for _ in self.lambs]
        else:
            self.logger.info("skipping loading, training from scratch")

    def binarize(self, scope: str, ratio: float):
        """
        Binarize lambda values to 0 or 1 based on scope and sparsity ratio.
        
        Performance Note: This implementation duplicates logic from other hook types.
        Future optimization should extract common binarization utilities to reduce
        code duplication and improve maintainability.
        :param scope: either locally (sparsity within layer) or globally (sparsity within model)
        :param ratio: the ratio of the number of 1s to the total number of elements
        """
        assert scope in ["local", "global"], "scope must be either local or global"
        assert not self.binary, "binarization is not supported when using binary mask already"
        if scope == "local":
            # Local binarization
            for i, lamb in enumerate(self.lambs):
                num_heads = lamb.size(0)
                num_activate_heads = int(num_heads * ratio)
                # Sort the lambda values with stable sorting to maintain order for equal values
                sorted_lamb, sorted_indices = torch.sort(lamb, descending=True, stable=True)
                # Find the threshold value
                threshold = sorted_lamb[num_activate_heads - 1]
                # Create a mask based on the sorted indices
                mask = torch.zeros_like(lamb)
                mask[sorted_indices[:num_activate_heads]] = 1.0
                # Binarize the lambda based on the threshold and the mask
                self.lambs[i] = torch.where(lamb > threshold, torch.ones_like(lamb), mask)
        else:
            # Global binarization
            all_lambs = torch.cat([lamb.flatten() for lamb in self.lambs])
            num_total = all_lambs.numel()
            num_activate = int(num_total * ratio)
            # Sort all lambda values globally with stable sorting
            sorted_lambs, sorted_indices = torch.sort(all_lambs, descending=True, stable=True)
            # Find the global threshold value
            threshold = sorted_lambs[num_activate - 1]
            # Create a global mask based on the sorted indices
            global_mask = torch.zeros_like(all_lambs)
            global_mask[sorted_indices[:num_activate]] = 1.0
            # Binarize all lambdas based on the global threshold and mask
            start_idx = 0
            for i in range(len(self.lambs)):
                end_idx = start_idx + self.lambs[i].numel()
                lamb_mask = global_mask[start_idx:end_idx].reshape(self.lambs[i].shape)
                self.lambs[i] = torch.where(self.lambs[i] > threshold, torch.ones_like(self.lambs[i]), lamb_mask)
                start_idx = end_idx
        self.binary = True

    @staticmethod
    def masking_fn(hidden_states, **kwargs):
        hidden_states_dtype = hidden_states.dtype
        lamb = kwargs["lamb"].view(1, 1, kwargs["lamb"].shape[0])
        if kwargs.get("masking", None) == "sigmoid":
            mask = torch.sigmoid(lamb)
        elif kwargs.get("masking", None) == "hard_discrete":
            use_log = kwargs.get("use_log", True)
            eps = kwargs.get("eps", 1e-8)
            mask = hard_concrete_distribution(lamb, use_log=use_log, eps=eps)
        elif kwargs.get("masking", None) == "binary":
            mask = lamb
        elif kwargs.get("masking", None) == "continues2binary":
            # TODO: this might cause potential issue as it hard threshold at 0
            mask = (lamb > 0).float()
        elif kwargs.get("masking", None) == "no_masking":
            mask = torch.ones_like(lamb)
        else:
            raise NotImplementedError
        epsilon = kwargs.get("epsilon", 0.0)

        if hidden_states.dim() == 2:
            mask = mask.squeeze(1)

        hidden_states = hidden_states * mask + torch.randn_like(hidden_states) * epsilon * (1 - mask)
        return hidden_states.to(hidden_states_dtype)

    def get_norm_masking_hook(self, init_value=1.0):
        """
        Get a hook function to mask feed forward layer
        """

        def hook_fn(module, args, kwargs, output, name):
            # initialize lambda with acual head dim in the first run
            if self.lambs[self.hook_counter] is None:
                self.lambs[self.hook_counter] = (
                    torch.ones(self.module_neurons[name], device=self.pipeline.device, dtype=self.dtype) * init_value
                )
                self.lambs[self.hook_counter].requires_grad = True
                # load norm lambda module name for logging
                self.lambs_module_names[self.hook_counter] = name

            # perform masking
            output = self.masking_fn(
                output,
                masking=self.masking,
                lamb=self.lambs[self.hook_counter],
                epsilon=self.epsilon,
                eps=self.eps,
                use_log=self.use_log,
            )
            self.hook_counter += 1
            self.hook_counter %= len(self.lambs)
            return output

        return hook_fn
