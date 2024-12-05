import logging
import os
from collections import OrderedDict
from functools import partial

import torch

import re

import math
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import deprecate


def scaled_dot_product_attention_atten_weight_only(
    query, key, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight


def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def masking_fn(hidden_states, kwargs):
    lamb = kwargs["lamb"].view(1, kwargs["lamb"].shape[0], 1, 1)
    if kwargs.get("masking", None) == "sigmoid":
        mask = torch.sigmoid(lamb)
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
    hidden_states = hidden_states * mask + torch.randn_like(hidden_states) * epsilon * (1 - mask)
    return hidden_states


class AttnProcessor2_0_Masking:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = (
                "The `scale` argument is deprecated and will be ignored. "
                "Please remove it, as passing it will raise an error "
                "in the future. `scale` should directly be passed while "
                "calling the underlying pipeline component i.e., via "
                "`cross_attention_kwargs`."
            )
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if kwargs.get("return_attention", True):
            # add the attention output from F.scaled_dot_product_attention
            attn_weight = scaled_dot_product_attention_atten_weight_only(
                query, key, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states_aft_attention_ops = hidden_states.clone()
            attn_weight_old = attn_weight.to(hidden_states.device).clone()
        else:
            hidden_states_aft_attention_ops = None
            attn_weight_old = None

        # masking for the hidden_states after the attention ops
        if kwargs.get("lamb", None) is not None:
            hidden_states = masking_fn(hidden_states, kwargs)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states, hidden_states_aft_attention_ops, attn_weight_old

class BaseCrossAttentionHooker:
    def __init__(self, pipeline, regex, dtype, head_num_filter, masking, model_name, attn_name, use_log, eps):
        self.pipeline = pipeline
        # unet for SD2 SDXL, transformer for SD3, FLUX DIT
        self.net = pipeline.unet if hasattr(pipeline, "unet") else pipeline.transformer
        self.model_name = model_name
        self.module_heads = OrderedDict()
        self.masking = masking
        self.hook_dict = {}
        self.regex = regex
        self.dtype = dtype
        self.head_num_filter = head_num_filter
        self.attn_name = attn_name
        self.logger = logging.getLogger(__name__)
        self.use_log = use_log  # use log parameter to control hard_discrete
        self.eps = eps

    def add_hooks_to_cross_attention(self, hook_fn: callable):
        """
        Add forward hooks to every cross attention
        :param hook_fn: a callable to be added to torch nn module as a hook
        :return:
        """
        total_hooks = 0
        for name, module in self.net.named_modules():
            name_last_word = name.split(".")[-1]
            if self.attn_name in name_last_word:
                if re.match(self.regex, name):
                    hook_fn = partial(hook_fn, name=name)
                    hook = module.register_forward_hook(hook_fn, with_kwargs=True)
                    self.hook_dict[name] = hook
                    self.module_heads[name] = module.heads
                    self.logger.info(f"Adding hook to {name}, module.heads: {module.heads}")
                    total_hooks += 1
        self.logger.info(f"Total hooks added: {total_hooks}")

    def clear_hooks(self):
        """clear all hooks"""
        for hook in self.hook_dict.values():
            hook.remove()
        self.hook_dict.clear()


class CrossAttentionExtractionHook(BaseCrossAttentionHooker):
    def __init__(
        self,
        pipeline,
        dtype,
        head_num_filter,
        masking,
        dst,
        regex=None,
        epsilon=0.0,
        binary=False,
        return_attention=False,
        model_name="sdxl",
        attn_name="attn",
        use_log=False,
        eps=1e-6,
    ):
        super().__init__(
            pipeline,
            regex,
            dtype,
            head_num_filter,
            masking=masking,
            model_name=model_name,
            attn_name=attn_name,
            use_log=use_log,
            eps=eps,
        )
        self.attention_processor = AttnProcessor2_0_Masking()
        self.lambs = []
        self.lambs_module_names = []
        self.cross_attn = []
        self.hook_counter = 0
        self.device = self.pipeline.unet.device if hasattr(self.pipeline, "unet") else self.pipeline.transformer.device
        self.dst = dst
        self.epsilon = epsilon
        self.binary = binary
        self.return_attention = return_attention
        self.model_name = model_name

    def clean_cross_attn(self):
        self.cross_attn = []

    def validate_dst(self):
        if os.path.exists(self.dst):
            raise ValueError(f"Destination {self.dst} already exists")

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

    def load(self, device, threshold=2.5):
        if os.path.exists(self.dst):
            self.logger.info(f"loading lambda from {self.dst}")
            self.lambs = torch.load(self.dst, weights_only=True, map_location=device)
            if self.binary:
                # set binary masking for each lambda by using clamp
                self.lambs = [(torch.relu(lamb - threshold) > 0).float() for lamb in self.lambs]
        else:
            self.logger.info("skipping loading, training from scratch")

    def binarize(self, scope: str, ratio: float):
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

    def bizarize_threshold(self, threshold: float):
        """
        Binarize lambda values based on a predefined threshold.
        :param threshold: The threshold value for binarization
        """
        assert not self.binary, "Binarization is not supported when using binary mask already"

        for i in range(len(self.lambs)):
            self.lambs[i] = (self.lambs[i] >= threshold).float()

        self.binary = True

    def get_cross_attn_extraction_hook(self, init_value=1.0):
        """get a hook function to extract cross attention"""

        # the reason to use a function inside a function is to save the extracted cross attention
        def hook_fn(module, args, kwargs, output, name):
            # initialize lambda with acual head dim in the first run
            if self.lambs[self.hook_counter] is None:
                self.lambs[self.hook_counter] = (
                    torch.ones(module.heads, device=self.pipeline.device, dtype=self.dtype) * init_value
                )
                # Only set requires_grad to True when the head number is larger than the filter
                if self.head_num_filter <= module.heads:
                    self.lambs[self.hook_counter].requires_grad = True

                # load attn lambda module name for logging
                self.lambs_module_names[self.hook_counter] = name

            hidden_states, _, attention_output = self.attention_processor(
                module,
                args[0],
                encoder_hidden_states=kwargs["encoder_hidden_states"],
                attention_mask=kwargs["attention_mask"],
                lamb=self.lambs[self.hook_counter],
                masking=self.masking,
                epsilon=self.epsilon,
                return_attention=self.return_attention,
                use_log=self.use_log,
                eps=self.eps,
            )
            if attention_output is not None:
                self.cross_attn.append(attention_output)
            self.hook_counter += 1
            self.hook_counter %= len(self.lambs)
            return hidden_states

        return hook_fn

    def add_hooks(self, init_value=1.0):
        hook_fn = self.get_cross_attn_extraction_hook(init_value)
        self.add_hooks_to_cross_attention(hook_fn)
        # initialize the lambda
        self.lambs = [None] * len(self.module_heads)
        # initialize the lambda module names
        self.lambs_module_names = [None] * len(self.module_heads)

    def get_process_cross_attn_result(self, text_seq_length, timestep: int = -1):
        if isinstance(timestep, str):
            timestep = int(timestep)
        # num_lambda_block contains lambda (head masking)
        num_lambda_block = len(self.lambs)

        # get the start and end position of the timestep
        start_pos = timestep * num_lambda_block
        end_pos = (timestep + 1) * num_lambda_block
        if end_pos > len(self.cross_attn):
            raise ValueError(f"timestep {timestep} is out of range")

        # list[cross_attn_map] num_layer x [batch, num_heads, seq_vis_tokens, seq_text_tokens]
        attn_maps = self.cross_attn[start_pos:end_pos]

        def heatmap(attn_list, attn_idx, head_idx, text_idx):
            # only select second element in the tuple (with text guided attention)
            # layer_idx, 1, head_idx, seq_vis_tokens, seq_text_tokens
            map = attn_list[attn_idx][1][head_idx][:][:, text_idx]
            # get the size of the heatmap
            size = int(map.shape[0] ** 0.5)
            map = map.view(size, size, 1)
            data = map.cpu().float().numpy()
            return data

        output_dict = {}
        for lambda_block_idx, lambda_block_name in zip(range(num_lambda_block), self.lambs_module_names):
            data_list = []
            for head_idx in range(len(self.lambs[lambda_block_idx])):
                for token_idx in range(text_seq_length):
                    # number of heatmap is equal to the number of tokens in the text sequence X number of heads
                    data_list.append(heatmap(attn_maps, lambda_block_idx, head_idx, token_idx))
            output_dict[lambda_block_name] = {"attn_map": data_list, "lambda": self.lambs[lambda_block_idx]}
        return output_dict
