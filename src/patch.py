import torch
import torch.nn as nn
import math
import types
from comfy.model_patcher import ModelPatcher
from comfy import model_sampling
from .rope import get_1d_rotary_pos_embed
try:
    from nunchaku.models.embeddings import pack_rotemb
except: 
    pack_rotemb = lambda x:x


class FluxPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], method: str = 'yarn', dype: bool = True, dype_exponent: float = 2.0): # Add dype_exponent
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.method = method
        self.dype = dype if method != 'base' else False
        self.dype_exponent = dype_exponent
        self.current_timestep = 1.0
        self.base_resolution = 1024
        self.base_patches = (self.base_resolution // 8) // 2

    def set_timestep(self, timestep: float):
        self.current_timestep = timestep

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb_parts = []
        pos = ids.float()
        freqs_dtype = torch.bfloat16

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'repeat_interleave_real': True, 'use_real': True, 'freqs_dtype': freqs_dtype}
            
            # Pass the exponent to the RoPE function
            dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_exponent': self.dype_exponent}

            if i > 0:
                max_pos = axis_pos.max().item()
                current_patches = int(max_pos + 1)

                if self.method == 'yarn' and current_patches > self.base_patches:
                    max_pe_len = torch.tensor(current_patches, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs, yarn=True, max_pe_len=max_pe_len, ori_max_pe_len=self.base_patches, **dype_kwargs)
                elif self.method == 'ntk' and current_patches > self.base_patches:
                    base_ntk_scale = (current_patches / self.base_patches)
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs, ntk_factor=base_ntk_scale, **dype_kwargs)
                else:
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs)
            else:
                cos, sin = get_1d_rotary_pos_embed(**common_kwargs)

            cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
            sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
            row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
            row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
            matrix = torch.stack([row1, row2], dim=-2)
            emb_parts.append(matrix)

        emb = torch.cat(emb_parts, dim=-3)
        return emb.unsqueeze(1).to(ids.device)



class FluxPosEmbedNunchaku(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], method: str = "yarn",
                 dype: bool = True, dype_exponent: float = 2.0):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.method = method
        self.dype = dype if method != "base" else False
        self.dype_exponent = float(dype_exponent)
        self.current_timestep = 1.0
        self.base_resolution = 1024
        self.base_patches = (self.base_resolution // 8) // 2

    def set_timestep(self, timestep: float):
        self.current_timestep = float(timestep)

    def _axis_rope_from_cos_sin(self, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Convert cos/sin with shape (..., M, D) into rope shape (B, M, D//2, 1, 2).
        This mirrors HF/Nunchaku rope contract:
          stacked_out = torch.stack([sin_out, cos_out], dim=-1)
          out = stacked_out.view(B, M, D//2, 1, 2)
        where cos/sin are repeat-interleaved pairs.
        """
        if cos.dim() < 2:
            raise RuntimeError(f"Unexpected cos shape {tuple(cos.shape)}")

        # shape components
        *lead, M, D = list(cos.shape)
        # collapse any leading batch dims into a single batch for the rope contract
        if lead:
            batch = int(torch.prod(torch.tensor(lead, dtype=torch.int64)).item())
            cos_flat = cos.reshape(batch, M, D)
            sin_flat = sin.reshape(batch, M, D)
        else:
            batch = 1
            cos_flat = cos.reshape(1, M, D)
            sin_flat = sin.reshape(1, M, D)

        assert D % 2 == 0, "rotary dimension must be even"
        D_half = D // 2

        # cos/sin were produced with repeat_interleave(2) style: [..., d0, d0, d1, d1, ...]
        # view pairs and take the first of each pair (consistent with rope contract).
        cos_pairs = cos_flat.view(batch, M, D_half, 2)
        sin_pairs = sin_flat.view(batch, M, D_half, 2)
        cos_out = cos_pairs[..., 0]  # (batch, M, D_half)
        sin_out = sin_pairs[..., 0]  # (batch, M, D_half)

        stacked = torch.stack([sin_out, cos_out], dim=-1)  # (batch, M, D_half, 2)
        rope = stacked.view(batch, M, D_half, 1, 2).contiguous()  # (batch, M, D_half, 1, 2)

        # restore original leading dims (if any)
        if lead:
            restore_shape = (*lead, M, D_half, 1, 2)
            rope = rope.reshape(restore_shape)

        return rope.float()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids: (..., n_axes) or (n_axes,) accepted. Returns emb.unsqueeze(1)
        with emb built by concatenating per-axis rope tensors at dim=-3.
        """
        # ensure a batch dim exists like nunchaku/hf expects for rope()
        added_batch = False
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
            added_batch = True

        n_axes = ids.shape[-1]
        pos = ids.float()
        emb_parts = []
        freqs_dtype = torch.float32  # rope contract expects float32

        for i in range(n_axes):
            axis_pos = pos[..., i]           # shape: (..., M_i)
            # axes_dim entries should be numeric (per-axis rotary dim). defensively handle strings/others:
            axis_dim = self.axes_dim[i]
            try:
                axis_dim = int(axis_dim)
            except Exception:
                # fallback: evenly split from model dim if available; otherwise use 16
                total_dim = getattr(self, "dim", None) or getattr(self, "inner_dim", None)
                if total_dim is None:
                    axis_dim = 16
                else:
                    axis_dim = max(2, int(int(total_dim) // (ids.shape[-1])))

            common_kwargs = {
                "dim": axis_dim,
                "pos": axis_pos,
                "theta": self.theta,
                "repeat_interleave_real": True,
                "use_real": True,
                "freqs_dtype": freqs_dtype,
            }
            dype_kwargs = {
                "dype": bool(self.dype),
                "current_timestep": float(self.current_timestep),
                "dype_exponent": float(self.dype_exponent),
            }

            # choose yarn/ntk path analogously to your original code
            if i > 0:
                max_pos = int(axis_pos.max().item())
                current_patches = max_pos + 1
                if self.method == "yarn" and current_patches > self.base_patches:
                    max_pe_len = torch.tensor(current_patches, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs, yarn=True,
                                                       max_pe_len=max_pe_len, ori_max_pe_len=self.base_patches,
                                                       **dype_kwargs)
                elif self.method == "ntk" and current_patches > self.base_patches:
                    base_ntk_scale = (current_patches / self.base_patches)
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs, ntk_factor=base_ntk_scale, **dype_kwargs)
                else:
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs, **dype_kwargs)
            else:
                cos, sin = get_1d_rotary_pos_embed(**common_kwargs, **dype_kwargs)

            # convert cos/sin -> rope shape (B, M, D//2, 1, 2)
            rope_i = self._axis_rope_from_cos_sin(cos, sin)
            emb_parts.append(rope_i)

        # Concatenate along axis dimension like Nunchaku: dim = -3
        emb = torch.cat(emb_parts, dim=-3)  # shape: (B, M, D_total//2, 1, 2) with any leading batch dims preserved

        # match original FluxPosEmbed API: unsqueeze(1)
        out = emb.unsqueeze(1).to(ids.device)

        return out



def apply_dype_to_flux(model: ModelPatcher, is_nunchaku: bool, width: int, height: int, method: str, enable_dype: bool, dype_exponent: float, base_shift: float, max_shift: float) -> ModelPatcher:
    m = model.clone()
    
    if not hasattr(m.model.model_sampling, "_dype_patched"):
        model_sampler = m.model.model_sampling
        if isinstance(model_sampler, model_sampling.ModelSamplingFlux):
            if is_nunchaku:
                patch_size = m.model.diffusion_model.model.config.patch_size
            else:
                patch_size = m.model.diffusion_model.patch_size
            latent_h, latent_w = height // 8, width // 8
            padded_h, padded_w = math.ceil(latent_h / patch_size) * patch_size, math.ceil(latent_w / patch_size) * patch_size
            image_seq_len = (padded_h // patch_size) * (padded_w // patch_size)
            base_seq_len, max_seq_len = 256, 4096
            slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            intercept = base_shift - slope * base_seq_len
            dype_shift = image_seq_len * slope + intercept

            def patched_sigma_func(self, timestep):
                return model_sampling.flux_time_shift(dype_shift, 1.0, timestep)

            model_sampler.sigma = types.MethodType(patched_sigma_func, model_sampler)
            model_sampler._dype_patched = True

    try:
        if is_nunchaku:
            orig_embedder = m.model.diffusion_model.model.pos_embed
        else:
            orig_embedder = m.model.diffusion_model.pe_embedder
        theta, axes_dim = orig_embedder.theta, orig_embedder.axes_dim
        print(is_nunchaku, theta)
    except AttributeError:
        raise ValueError("The provided model is not a compatible FLUX model.")

    if is_nunchaku:
        new_pe_embedder = FluxPosEmbedNunchaku(theta, axes_dim, method, enable_dype, dype_exponent)
        m.add_object_patch("diffusion_model.model.pos_embed", new_pe_embedder)
    else:
        new_pe_embedder = FluxPosEmbed(theta, axes_dim, method, enable_dype, dype_exponent)
        m.add_object_patch("diffusion_model.pe_embedder", new_pe_embedder)
    
    sigma_max = m.model.model_sampling.sigma_max.item()

    def dype_wrapper_function(model_function, args_dict):
        if enable_dype:
            timestep_tensor = args_dict.get("timestep")
            if timestep_tensor is not None and timestep_tensor.numel() > 0:
                current_sigma = timestep_tensor.item()
                if sigma_max > 0:
                    normalized_timestep = min(max(current_sigma / sigma_max, 0.0), 1.0)
                    new_pe_embedder.set_timestep(normalized_timestep)
        
        input_x, c = args_dict.get("input"), args_dict.get("c", {})
        return model_function(input_x, args_dict.get("timestep"), **c)

    m.set_model_unet_function_wrapper(dype_wrapper_function)
    
    return m