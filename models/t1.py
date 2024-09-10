import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d,
        H,
        T,
        chunk_size,  # New parameter for chunk size
        bias=False,
        dropout=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        chunk_size: Size of chunks to divide the sequence into
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        assert d % H == 0
        assert T % chunk_size == 0  # Ensure sequence length is divisible by chunk size

        # Key, query, value projections
        self.c_attn = nn.Linear(d, 3 * d, bias=bias)

        # Projection of concatenated attention head outputs
        self.c_proj = nn.Linear(d, d, bias=bias)

        # Dropout modules
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.H = H
        self.d = d
        self.chunk_size = chunk_size

        # Register buffer for the causal mask
        # This mask ensures attention is only applied to the left
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))

    def forward(self, x):
        B, T, _ = x.size()  # Batch size, sequence length, embedding dimensionality

        # Compute query, key, and value vectors for all heads in batch
        # Split the output into separate query, key, and value tensors
        q, k, v = self.c_attn(x).split(self.d, dim=2)  # [B, T, d]

        # Reshape tensor into sequences of smaller token vectors for each head
        k = k.view(B, T, self.H, self.d // self.H).transpose(1, 2)  # [B, H, T, d // H]
        q = q.view(B, T, self.H, self.d // self.H).transpose(1, 2)
        v = v.view(B, T, self.H, self.d // self.H).transpose(1, 2)

        # Chunk the sequence
        num_chunks = T // self.chunk_size
        k_chunks = k.view(B, self.H, num_chunks, self.chunk_size, self.d // self.H)
        q_chunks = q.view(B, self.H, num_chunks, self.chunk_size, self.d // self.H)
        v_chunks = v.view(B, self.H, num_chunks, self.chunk_size, self.d // self.H)

        # Compute attention for each chunk
        att_chunks = []
        for i in range(num_chunks):
            # Extract the relevant chunk
            k_chunk = k_chunks[:, :, i, :, :]
            q_chunk = q_chunks[:, :, i, :, :]

            # Compute attention within the chunk
            att = (q_chunk @ k_chunk.transpose(-2, -1)) * (
                1.0 / math.sqrt(k_chunk.size(-1))
            )  # [B, H, chunk_size, chunk_size]

            # Apply the causal mask within the chunk
            att = att.masked_fill(self.mask[:, :, : self.chunk_size, : self.chunk_size] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # Store the attention for the current chunk
            att_chunks.append(att)

        # Concatenate the attention matrices from all chunks
        att = torch.cat(att_chunks, dim=2)

        # Compute output vectors for each token
        y = att @ v_chunks.view(B, self.H, num_chunks * self.chunk_size, self.d // self.H)  # [B, H, T, d // H]

        # Concatenate outputs from each attention head and linearly project
        y = y.transpose(1, 2).contiguous().view(B, T, self.d)
        y = self.resid_dropout(self.c_proj(y))
        return y

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        # self.attn = CausalSelfAttention(hidden_size, num_heads,  qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")  # noqa: E731
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Model(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        out_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        condition_channels=2048,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = out_channels * 2 if learn_sigma else out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        # h = w = int(x.shape[1] ** 0.5)
        # assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h // p, w // p, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h, w))
        return imgs

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, pos=None, past_frame=None, past_pos=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # print("========>", t.device, t.dtype, t)
        # print("<==========", x.shape, first_frame.shape, pos.shape)
        # past_frame = rearrange(past_frame, "N C T H W -> N (C T) 1 H W")
        # past_pos = rearrange(past_pos, "N C T H W -> N (C T) 1 H W")

        x = torch.cat([x, past_frame], dim=2)
        pos = torch.cat([pos, past_pos], dim=2)
        T = x.size(2)

        N, _, T, H, W = x.shape

        x = rearrange(x, "N C T H W -> (N T) C H W")
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        x = rearrange(x, "(N T) Z D -> N (T Z) D", N=N)

        with torch.no_grad():
            pos_emb = get_nd_sincos_pos_embed_from_grid(self.hidden_size, pos).detach()
        pos_emb = rearrange(pos_emb, "(N T Z) D -> N (T Z) D", N=N, T=T)

        t = self.t_embedder(t)                   # (N, D)
        c = t.unsqueeze(1).repeat(1, x.shape[1], 1) + pos_emb

        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)       # (N, T, D)
            # x = block(x, c)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)

        x = rearrange(x, "N (T Z) D -> (N T) Z D", T=T)
        x = self.unpatchify(x, H, W)                    # (N, out_channels, H, W)
        x = rearrange(x, "(N T) C H W -> N C T H W", T=T)
        x = torch.mean(x, dim=2, keepdim=True)
        return x

    ''''
    def forward(self, x, t, pos=None, past_frame=None, past_pos=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # print("========>", t.device, t.dtype, t)
        # print("<==========", x.shape, first_frame.shape, pos.shape)
        past_frame = rearrange(past_frame, "N C T H W -> N (C T) 1 H W")
        past_pos = rearrange(past_pos, "N C T H W -> N (C T) 1 H W")

        x = torch.cat([x, past_frame], dim=1)
        pos = torch.cat([pos, past_pos], dim=1)

        N, _, T, H, W = x.shape

        x = rearrange(x, "N C T H W -> (N T) C H W")
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        x = rearrange(x, "(N T) Z D -> N (T Z) D", N=N)

        with torch.no_grad():
            pos_emb = get_nd_sincos_pos_embed_from_grid(self.hidden_size, pos).detach()
        pos_emb = rearrange(pos_emb, "(N T Z) D -> N (T Z) D", N=N, T=T)

        t = self.t_embedder(t)                   # (N, D)
        c = t.unsqueeze(1).repeat(1, x.shape[1], 1) + pos_emb

        for block in self.blocks:
            # x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)       # (N, T, D)
            x = block(x, c)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)

        x = rearrange(x, "N (T Z) D -> (N T) Z D", T=1)
        x = self.unpatchify(x, H, W)                    # (N, out_channels, H, W)
        x = rearrange(x, "(N T) C H W -> N C T H W", T=1)
        return x
    '''

    def forward_with_cfg(self, x, t, y, pos, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, pos)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_nd_sincos_pos_embed_from_grid(embed_dim, pos):
    C = pos.size(1)
    assert embed_dim % C % 2 == 0

    emb = []
    for i in range(C):
        emb_i = get_1d_sincos_pos_embed_from_grid(embed_dim // C, pos[:, i])
        emb.append(emb_i)
    emb = torch.cat(emb, dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    omega = torch.arange(embed_dim // 2, dtype=torch.float64, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)

    emb = emb.to(pos.dtype)
    return emb
