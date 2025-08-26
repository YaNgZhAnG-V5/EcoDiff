import logging
import os
from collections import OrderedDict
from functools import partial

import torch

import re
from sdib.hooks.attention_processor import (
    AttnProcessor2_0_Masking,
    FluxAttnProcessor2_0_Masking,
    JointAttnProcessor2_0_Masking,
)


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
    """
    Note: The naming convention uses "Cross Attention" for historical reasons,
    but this class handles both self-attention and cross-attention mechanisms.
    Future refactoring may simplify the naming to "Attention" for clarity.
    """

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
        if model_name in ["flux", "flux_dev"]:
            self.attention_processor = FluxAttnProcessor2_0_Masking()
        elif model_name == "sd3":
            self.attention_processor = JointAttnProcessor2_0_Masking()
        else:
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
        """
        binarize lambda to be 0 or 1
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

            if self.model_name == "sd3":
                encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
                if encoder_hidden_states is None:
                    hidden_states = self.attention_processor(
                        module,
                        hidden_states=kwargs["hidden_states"],
                        lamb=self.lambs[self.hook_counter],
                        masking=self.masking,
                        epsilon=self.epsilon,
                        use_log=self.use_log,
                        eps=self.eps,
                    )
                else:
                    hidden_states, encoder_hidden_states = self.attention_processor(
                        module,
                        hidden_states=kwargs["hidden_states"],
                        encoder_hidden_states=kwargs["encoder_hidden_states"],
                        attention_mask=kwargs.get("attention_mask", None),
                        lamb=self.lambs[self.hook_counter],
                        masking=self.masking,
                        epsilon=self.epsilon,
                        use_log=self.use_log,
                        eps=self.eps,
                    )
                self.hook_counter += 1
                self.hook_counter %= len(self.lambs)
                if encoder_hidden_states is None:
                    return hidden_states
                else:
                    return hidden_states, encoder_hidden_states
            elif self.model_name in ["flux", "flux_dev"]:
                encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
                # flux has two different attention processors, FluxSingleAttnProcessor and FluxAttnProcessor
                if "single" in name:
                    hidden_states = self.attention_processor(
                        module,
                        hidden_states=kwargs.get("hidden_states", None),
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=kwargs.get("attention_mask", None),
                        image_rotary_emb=kwargs.get("image_rotary_emb", None),
                        lamb=self.lambs[self.hook_counter],
                        masking=self.masking,
                        epsilon=self.epsilon,
                        use_log=self.use_log,
                        eps=self.eps,
                    )
                    self.hook_counter += 1
                    self.hook_counter %= len(self.lambs)
                    return hidden_states
                else:
                    hidden_states, encoder_hidden_states = self.attention_processor(
                        module,
                        hidden_states=kwargs.get("hidden_states", None),
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=kwargs.get("attention_mask", None),
                        image_rotary_emb=kwargs.get("image_rotary_emb", None),
                        lamb=self.lambs[self.hook_counter],
                        masking=self.masking,
                        epsilon=self.epsilon,
                        use_log=self.use_log,
                        eps=self.eps,
                    )
                    self.hook_counter += 1
                    self.hook_counter %= len(self.lambs)
                    return hidden_states, encoder_hidden_states
            else:
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
        """
        this method is used to extract and preprocess the cross attention map

        all attention maps are stored in self.cross_attn as a list of tensors
        total length of self.cross_attn is num_lambda_block * num_timesteps
        each tensor in the list is of shape [batch, num_heads, seq_vis_tokens, seq_text_tokens]
        e.g. for sdxl 'down_blocks.1.attentions.0.transformer_blocks.0.attn2' block contains
        10 heads, the shape of the tensor is [2, 12, 4096, 77]

        e.g. for sdxl model applying masking on all attention blocks with number of intervention steps 50,
        the length of self.cross_attn is 50 * 70 (total number of blocks in UNET from sdxl) 3500

        :param num_tokens: the number of tokens in the text sequence
        :param timestep: t in the paper, the timestep to extract the cross attention map, t in [0, T-1]
                         as number of intervention steps
        return: a dictionary containing the extracted cross attention map for timestep t
                {
                    #  number of heads * number of tokens x (reshape heatmap 64x64x1)
                    block_name1: [heatmap1, heatmap2, ...],
                    block_name2: [heatmap1, heatmap2, ...],
                    num_lambda_block: num_lambda_block
                    lambda_list: lambda_list
                }
        """
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
