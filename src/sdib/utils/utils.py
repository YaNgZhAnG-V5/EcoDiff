# all utiles functions
import math
from typing import List, Optional

import torch
from diffusers.models.activations import GEGLU, GELU


def get_total_params(model, trainable: bool = True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == trainable)


def get_precision(precision: str):
    assert precision in ["fp16", "fp32", "bf16"], "precision must be either fp16, fp32, bf16"
    if precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp32":
        torch_dtype = torch.float32
    elif precision == "fp64":
        torch_dtype = torch.float64
    return torch_dtype


def calculate_mask_sparsity(hooker, threshold: Optional[float] = None):
    total_num_lambs = 0
    num_activate_lambs = 0
    binary = getattr(hooker, "binary", None)  # if binary is not present, it will return None for ff_hooks
    for lamb in hooker.lambs:
        total_num_lambs += lamb.size(0)
        if binary:
            assert threshold is None, "threshold should be None for binary mask"
            num_activate_lambs += lamb.sum().item()
        else:
            assert threshold is not None, "threshold must be provided for non-binary mask"
            num_activate_lambs += (lamb >= threshold).sum().item()
    return total_num_lambs, num_activate_lambs, num_activate_lambs / total_num_lambs


def linear_layer_masking(module, lamb):
    """
    Apply soft masking to attention layer weights (K, Q, V projections).
    
    This function multiplies attention layer weights by mask values without
    removing parameters, allowing for gradual pruning during training.
    
    Args:
        module: Attention module containing to_k, to_q, to_v, and to_out
        lamb: Per-head mask values to apply
        
    Returns:
        module: Modified module with masked weights
    """
    # perform masking on K Q V to see if it still works
    inner_dim = module.to_k.in_features // module.heads
    modules_to_remove = [module.to_k, module.to_q, module.to_v]
    for module_to_remove in modules_to_remove:
        for idx, head_mask in enumerate(lamb):
            module_to_remove.weight.data[idx * inner_dim : (idx + 1) * inner_dim, :] *= head_mask
            if module_to_remove.bias is not None:
                module_to_remove.bias.data[idx * inner_dim : (idx + 1) * inner_dim] *= head_mask

    # perform masking on the output
    for idx, head_mask in enumerate(lamb):
        module.to_out[0].weight.data[:, idx * inner_dim : (idx + 1) * inner_dim] *= head_mask
    return module


# create dummy module for skip connection
class SkipConnection(torch.nn.Module):
    """
    Skip connection module for completely pruned layers.
    
    When a layer is fully pruned, this module replaces it and simply
    returns the input unchanged, maintaining the model's forward pass.
    """
    def __init__(self):
        super(SkipConnection, self).__init__()

    def forward(*args, **kwargs):
        return args[1]


class AttentionSkipConnection(torch.nn.Module):
    """
    Model-specific skip connection for attention layers.
    
    Handles different return patterns based on model architecture:
    - SD3/FLUX models may return multiple values
    - Other models return single hidden states
    
    Args:
        model_type: Type of diffusion model ("sd3", "flux", "flux_dev", etc.)
    """
    def __init__(self, model_type):
        super(AttentionSkipConnection, self).__init__()
        self.model_type = model_type

    def forward(self, hidden_states=None, encoder_hidden_states=None, *args, **kwargs):
        # Return the first non-None input, or hidden_states as default
        if self.model_type not in ["sd3", "flux", "flux_dev"]:
            return hidden_states
        
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        
        return hidden_states


def linear_layer_pruning(module, lamb, model_type):
    """
    Physically prune attention layers by removing parameters for pruned heads.
    
    This function performs structural pruning through the following detailed steps:
    
    1. **Input Processing**: Latent features are fed into linear modules (to_k, to_q, to_v)
       with shape (cross_attn_dim, inner_kv_dim / inner_dim)
       
    2. **Head Division**: Inner features are divided into attention heads, where:
       - Query shape: [B, N, H, D] (batch, sequence, heads, head_dim)
       - New hidden dimension = inner_dim * (unmasked_heads / total_heads)
       - K, Q, V projections have shape [cross_attn_dim, inner_kv_dim / inner_dim]
       - Each head occupies (heads * inner_dim) rows in the weight matrix
       - **Important**: Input channels remain unchanged, only output rows are pruned
       
    3. **Attention Computation**: Updated latent features after scaled dot-product attention
    
    4. **Output Projection**: Final projection layer (to_out) from pruned inner_dim to original latent_dim
       - Pruned dimension changes from input (dim=0) to output (dim=1)
       - **Critical**: Output channels remain unchanged to maintain model compatibility
    
    Args:
        module: Attention module to prune (contains to_k, to_q, to_v, to_out)
        lamb: Learned mask values per attention head (1=keep, 0=prune)
        model_type: Model architecture type for skip connection handling
        
    Returns:
        module: Pruned attention module or AttentionSkipConnection if fully pruned
        
    Note:
        - Supports additional projections (add_k_proj, add_q_proj, add_v_proj) for certain architectures
        - Handles both to_out and to_add_out projection layers
        - Updates all relevant module parameters (inner_dim, query_dim, heads, etc.)
    """

    heads_to_keep = torch.nonzero(lamb).squeeze()
    if len(heads_to_keep.shape) == 0:
        # if only one head is kept, or none
        heads_to_keep = heads_to_keep.unsqueeze(0)

    modules_to_remove = [module.to_k, module.to_q, module.to_v]
    
    if getattr(module, "add_k_proj", None) is not None:
        modules_to_remove.extend([module.add_k_proj, module.add_q_proj, module.add_v_proj])

    new_heads = int(lamb.sum().item())

    if new_heads == 0:
        return AttentionSkipConnection(model_type=model_type)

    for module_to_remove in modules_to_remove:
        # get head dimension
        inner_dim = module_to_remove.out_features // module.heads
        # place holder for the rows to keep
        rows_to_keep = torch.zeros(
            module_to_remove.out_features, dtype=torch.bool, device=module_to_remove.weight.device
        )

        for idx in heads_to_keep:
            rows_to_keep[idx * inner_dim : (idx + 1) * inner_dim] = True

        # overwrite the inner projection with masked projection
        module_to_remove.weight.data = module_to_remove.weight.data[rows_to_keep, :]
        if module_to_remove.bias is not None:
            module_to_remove.bias.data = module_to_remove.bias.data[rows_to_keep]
        module_to_remove.out_features = int(sum(rows_to_keep).item())

    # Also update the output projection layer if available, (for FLUXSingleAttnProcessor2_0)
    # with column masking, dim 1
    if getattr(module, "to_out", None) is not None:
        module.to_out[0].weight.data = module.to_out[0].weight.data[:, rows_to_keep]
        module.to_out[0].in_features = int(sum(rows_to_keep).item())

    if getattr(module, "to_add_out", None) is not None:
        module.to_add_out.weight.data = module.to_add_out.weight.data[:, rows_to_keep]
        module.to_add_out.in_features = int(sum(rows_to_keep).item())

    # update parameters in the attention module
    module.inner_dim = module.inner_dim // module.heads * new_heads
    module.query_dim = module.query_dim // module.heads * new_heads
    module.inner_kv_dim = module.inner_kv_dim // module.heads * new_heads
    module.cross_attention_dim = module.cross_attention_dim // module.heads * new_heads
    module.heads = new_heads
    return module


def update_flux_single_transformer_projection(parent_module, module, lamb, old_inner_dim):
    """
    Updates the proj_out module in a FluxSingleTransformerBlock after attention head pruning.
    
    FLUX models use a proj_out layer that takes concatenated input from both attention output 
    and MLP hidden states: torch.cat([attn_output, mlp_hidden_states], dim=2). When attention 
    heads are pruned, the attention dimension changes but the MLP dimension remains constant,
    requiring careful weight matrix reconstruction.
    
    Args:
        parent_module: FluxSingleTransformerBlock containing the proj_out layer
        module: Pruned attention module (or AttentionSkipConnection)
        lamb: Original mask values used for pruning decisions
        old_inner_dim: Original attention inner dimension before pruning
        
    Returns:
        parent_module: Updated parent module with corrected proj_out dimensions
        
    Note:
        - Handles skip connections when module is completely pruned
        - Preserves MLP weights while updating attention weights
        - Only modifies proj_out if dimensions actually changed
    """
    # Handle Skip Connection case (when module is completely pruned)
    if isinstance(module, AttentionSkipConnection):
        return parent_module

    if hasattr(parent_module, "proj_out"):
        # Calculate how much the attention dimension changed
        attention_dim_change = old_inner_dim - module.inner_dim
        
        if attention_dim_change > 0:  # Only update if dimensions actually changed
            # Get current weight matrix and dimensions
            old_weight = parent_module.proj_out.weight.data
            old_in_features = parent_module.proj_out.in_features
            
            # Calculate new input dimension
            new_in_features = old_in_features - attention_dim_change
            
            # Create new weight matrix
            new_weight = torch.zeros(
                old_weight.shape[0], new_in_features,
                device=old_weight.device, dtype=old_weight.dtype
            )
            
            # Calculate head dimensions
            old_head_dim = old_inner_dim // lamb.shape[0]
            
            # Create mask for attention columns to keep
            heads_to_keep = torch.nonzero(lamb).squeeze()
            if len(heads_to_keep.shape) == 0:
                heads_to_keep = heads_to_keep.unsqueeze(0)
                
            attn_cols_to_keep = torch.zeros(old_inner_dim, dtype=torch.bool, device=old_weight.device)
            for idx in heads_to_keep:
                attn_cols_to_keep[idx * old_head_dim : (idx + 1) * old_head_dim] = True
            
            # Copy weights for kept attention heads
            kept_indices = torch.nonzero(attn_cols_to_keep).squeeze()
            for i, idx in enumerate(kept_indices):
                if i < module.inner_dim:
                    new_weight[:, i] = old_weight[:, idx]
            
            # Copy MLP weights (unchanged part)
            mlp_start = old_inner_dim
            if mlp_start < old_in_features:  # Ensure there's actually an MLP part
                new_weight[:, module.inner_dim:] = old_weight[:, mlp_start:]
            
            # Update the projection layer
            parent_module.proj_out.weight.data = new_weight
            parent_module.proj_out.in_features = new_in_features
    return parent_module


def ffn_linear_layer_pruning(module, lamb):
    """
    Prunes feed-forward network layers based on learned masks.
    
    Note: This function could potentially be merged with linear_layer_pruning 
    for better code organization in future refactoring.
    
    Args:
        module: FFN module to prune
        lamb: Learned mask values for pruning decisions
        
    Returns:
        Pruned module or SkipConnection if fully pruned
    """
    lambda_to_keep = torch.nonzero(lamb).squeeze()
    if len(lambda_to_keep) == 0:
        return SkipConnection()

    num_lambda = len(lambda_to_keep)

    if hasattr(module, "net") and len(module.net) >= 3:
        # Standard FFN blocks
        if isinstance(module.net[0], GELU):
            # linear layer weight remove before activation
            module.net[0].proj.weight.data = module.net[0].proj.weight.data[lambda_to_keep, :]
            module.net[0].proj.out_features = num_lambda
            if module.net[0].proj.bias is not None:
                module.net[0].proj.bias.data = module.net[0].proj.bias.data[lambda_to_keep]

            update_act = GELU(module.net[0].proj.in_features, num_lambda)
            update_act.proj = module.net[0].proj
            module.net[0] = update_act
        elif isinstance(module.net[0], GEGLU):
            output_feature = module.net[0].proj.out_features
            module.net[0].proj.weight.data = torch.cat(
                [
                    module.net[0].proj.weight.data[: output_feature // 2, :][lambda_to_keep, :],
                    module.net[0].proj.weight.data[output_feature // 2 :][lambda_to_keep, :],
                ],
                dim=0,
            )
            module.net[0].proj.out_features = num_lambda * 2
            if module.net[0].proj.bias is not None:
                module.net[0].proj.bias.data = torch.cat(
                    [
                        module.net[0].proj.bias.data[: output_feature // 2][lambda_to_keep],
                        module.net[0].proj.bias.data[output_feature // 2 :][lambda_to_keep],
                    ]
                )

            update_act = GEGLU(module.net[0].proj.in_features, num_lambda * 2)
            update_act.proj = module.net[0].proj
            module.net[0] = update_act

        # proj weight after activation
        module.net[2].weight.data = module.net[2].weight.data[:, lambda_to_keep]
        module.net[2].in_features = num_lambda
    
    elif hasattr(module, "proj_mlp") and hasattr(module, "proj_out"):
        # FFN For FluxSingleTransformerBlock
        module.proj_mlp.weight.data = module.proj_mlp.weight.data[lambda_to_keep, :]
        module.proj_mlp.out_features = num_lambda
        if module.proj_mlp.bias is not None:
            module.proj_mlp.bias.data = module.proj_mlp.bias.data[lambda_to_keep]
        
        # Update mlp_hidden_dim to reflect the new size
        old_mlp_hidden_dim = module.mlp_hidden_dim
        module.mlp_hidden_dim = num_lambda
        
        # The proj_out layer takes concatenated input from both attention output and MLP output
        # We need to keep the attention part unchanged but update the MLP part
        old_dim = module.proj_out.in_features
        attn_dim = old_dim - old_mlp_hidden_dim  # Attention dimension
        new_in_features = attn_dim + num_lambda
        
        new_weight = torch.zeros(
            module.proj_out.weight.shape[0], new_in_features,
            device=module.proj_out.weight.device, dtype=module.proj_out.weight.dtype
        )
        
        # Copy attention part (unchanged)
        new_weight[:, :attn_dim] = module.proj_out.weight.data[:, :attn_dim]
        
        # Copy selected MLP parts
        for i, idx in enumerate(lambda_to_keep):
            new_weight[:, attn_dim + i] = module.proj_out.weight.data[:, attn_dim + idx]
        
        # Update the projection layer
        module.proj_out.weight.data = new_weight
        module.proj_out.in_features = new_in_features
    
    return module


# create SparsityLinear module
class SparsityLinear(torch.nn.Module):
    """
    Sparse linear layer that maintains original output dimensions.
    
    This layer projects to a smaller intermediate dimension then expands
    back to the original size, placing values only at specified indices.
    Used for normalization layer pruning where output dimensions must match.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension (original size)
        lambda_to_keep: Indices of features to keep active
        num_lambda: Number of active features (len(lambda_to_keep))
    """
    def __init__(self, in_features, out_features, lambda_to_keep, num_lambda):
        super(SparsityLinear, self).__init__()
        self.sparse_proj = torch.nn.Linear(in_features, num_lambda)
        self.out_features = out_features
        self.lambda_to_keep = lambda_to_keep

    def forward(self, x):
        x = self.sparse_proj(x)
        output = torch.zeros(x.size(0), self.out_features, device=x.device, dtype=x.dtype)
        output[:, self.lambda_to_keep] = x
        return output


def norm_layer_pruning(module, lamb):
    """
    Pruning the layer normalization layer for FLUX model
    """
    lambda_to_keep = torch.nonzero(lamb).squeeze()
    if len(lambda_to_keep) == 0:
        return SkipConnection()

    num_lambda = len(lambda_to_keep)

    # get num_features
    in_features = module.linear.in_features
    out_features = module.linear.out_features

    sparselinear = SparsityLinear(in_features, out_features, lambda_to_keep, num_lambda)
    sparselinear.sparse_proj.weight.data = module.linear.weight.data[lambda_to_keep]
    sparselinear.sparse_proj.bias.data = module.linear.bias.data[lambda_to_keep]
    module.linear = sparselinear
    return module


def hard_concrete_distribution(
    p, beta: float = 0.83, eps: float = 1e-8, eta: float = 1.1, gamma: float = -0.1, use_log: bool = False
):
    u = torch.rand(p.shape).to(p.device)
    if use_log:
        p = torch.clamp(p, min=eps)
        p = torch.log(p)
    s = torch.sigmoid((torch.log(u + eps) - torch.log(1 - u + eps) + p) / beta)
    s = s * (eta - gamma) + gamma
    s = s.clamp(0, 1)
    return s


def l0_complexity_loss(alpha, beta: float = 0.83, eta: float = 1.1, gamma: float = -0.1, use_log: bool = False):
    offset = beta * math.log(-gamma / eta)
    loss = torch.sigmoid(alpha - offset).sum()
    return loss


def calculate_reg_loss(
    loss_reg,
    lambs: List[torch.Tensor],
    p: int,
    use_log: bool = False,
    mean=True,
    reg=True,  # regularize the lambda with bounded value range
    reg_alpha=0.4,  # alpha for the regularizer, avoid gradient vanishing
    reg_beta=1,  # beta for shifting the lambda toward positive value (avoid gradient vanishing)
):
    if p == 0:
        for lamb in lambs:
            loss_reg += l0_complexity_loss(lamb, use_log=use_log)
        loss_reg /= len(lambs)
    elif p == 1 or p == 2:
        for lamb in lambs:
            if reg:
                lamb = torch.sigmoid(lamb * reg_alpha + reg_beta)
            if mean:
                loss_reg += lamb.norm(p) / len(lamb)
            else:
                loss_reg += lamb.norm(p)
        loss_reg /= len(lambs)
    else:
        raise NotImplementedError
    return loss_reg
