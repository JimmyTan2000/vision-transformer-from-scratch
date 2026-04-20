import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Slices the image into patches and linearly projects them to a hidden dimension.
    This is equivalent to a 2D convolution with kernel_size=patch_size and stride=patch_size.
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, hidden_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, 
            hidden_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, in_channels, height, width)
        x = self.proj(x) # shape: (batch_size, hidden_dim, H/P, W/P)
        # Flatten the spatial dimensions (H/P, W/P) into a sequence of patches
        x = x.flatten(2) # shape: (batch_size, hidden_dim, (H/P) * (W/P))
        # Transpose to get sequence length as the second dimension: (batch_size, num_patches, hidden_dim)
        x = x.transpose(1, 2)
        return x

class Attention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) module.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # We compute queries, keys, and values in a single linear layer for efficiency
        # bias=True: the paper initializes all biases to zero (i.e., they exist)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # Shape of each: (B, num_heads, N, head_dim)

        # Attention mechanism
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Context vector
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (B, N, C)
        
        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block used in the Transformer Encoder.
    Contains two linear layers with a GELU activation in between.
    """
    def __init__(self, hidden_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)  # dropout after fc1 (paper §3)
        x = self.fc2(x)
        x = self.drop(x)  # kept: many implementations include this and it rarely hurts
        return x

class TransformerBlock(nn.Module):
    """
    A single Transformer Encoder block using Pre-LayerNorm architecture.
    """
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = Attention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection 1: x + Attention(LayerNorm(x))
        x = x + self.attn(self.norm1(x))
        # Residual connection 2: x + MLP(LayerNorm(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    The complete Vision Transformer (ViT) architecture as described in:
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    (Dosovitskiy et al., 2020).
    """
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        hidden_dim: int = 192,
        num_layers: int = 12,
        num_heads: int = 3,
        mlp_dim: int = 768,
        dropout: float = 0.1,
        head_type: str = 'pretrain',
    ):
        """
        Args:
            image_size:  Spatial size of input images (must be divisible by patch_size).
            patch_size:  Size of each square patch (P in the paper).
            in_channels: Number of input image channels (3 for RGB).
            num_classes: Number of output classes.
            hidden_dim:  Transformer hidden dimension D.
            num_layers:  Number of Transformer encoder blocks (depth L).
            num_heads:   Number of attention heads.
            mlp_dim:     Hidden dimension of the MLP block inside each Transformer block.
            dropout:     Dropout rate applied throughout the model.
            head_type:   'pretrain' uses an MLP head with one hidden layer (paper §3,
                         used when training from scratch). 'finetune' uses a single
                         linear layer (used when fine-tuning from a pre-trained checkpoint).

        Defaults are set for a ViT-Tiny equivalent suitable for CIFAR-10 (32x32 images).
        Standard ViT-Base: image_size=224, patch_size=16, hidden_dim=768,
                           num_layers=12, num_heads=12, mlp_dim=3072.
        """
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        assert head_type in ('pretrain', 'finetune'), "head_type must be 'pretrain' or 'finetune'."
        num_patches = (image_size // patch_size) ** 2

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(in_channels, patch_size, hidden_dim)

        # 2. Class Token — learnable parameter prepended to the patch sequence (§3, Eq. 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # 3. Positional Embedding — learnable 1D embeddings for all tokens (§3, Eq. 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 4. Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        # 5. Final LayerNorm before the classification head (§3, Eq. 4)
        self.norm = nn.LayerNorm(hidden_dim)

        # 6. Classification Head (§3):
        #    - 'pretrain': MLP with one hidden layer + Tanh, for training from scratch.
        #    - 'finetune': Single linear layer, for fine-tuning a pre-trained model.
        if head_type == 'pretrain':
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:  # 'finetune'
            self.head = nn.Linear(hidden_dim, num_classes)

        # 7. Weight Initialization (Appendix B):
        #    All weight matrices: truncated normal N(0, 0.02). All biases: zero.
        #    LayerNorm: weight=1, bias=0 (PyTorch default, made explicit).
        self.apply(self._init_weights)
        # cls_token and pos_embed are not covered by apply(), initialize separately.
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m: nn.Module):
        """Initialize weights per the paper (Appendix B): trunc_normal(std=0.02) for
        weights, zeros for biases. LayerNorm gamma=1 and beta=0."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Extract patch embeddings: (B, num_patches, hidden_dim)
        x = self.patch_embed(x)

        # Expand class token for the batch and prepend: (B, num_patches + 1, hidden_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings and apply dropout (§3, Eq. 1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through Transformer Encoder blocks (§3, Eq. 2-4)
        for block in self.blocks:
            x = block(x)

        # Apply final LayerNorm
        x = self.norm(x)

        # Extract CLS token representation (first token) and classify
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        return logits
