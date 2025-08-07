import torch
import torch.nn as nn

class TactileTokenizer(nn.Module):
    """

    Python class for a custom tokenizer that turns your 64 tactile taxels into token embeddings, preserving:

    Tokenizer for tactile input:

    - 4 fingers × 16 taxels = 64 tokens
    - Each taxel: [x, y, z, fx, fy, fz]


    Force & position per taxel
    Finger identity (as positional context)
    Learnable positional embeddings (ViT-compatible)
    Output shape: [B, N_tokens, D], ready for ViT-style Sparsh encoders

    """

    def __init__(self,
                 input_dim=6,        # [x, y, z, fx, fy, fz]
                 embed_dim=256,      # token dimension
                 num_fingers=4,
                 taxels_per_finger=16,
                 add_finger_embedding=True,
                 add_positional_encoding=True):
        super().__init__()

        self.num_taxels = num_fingers * taxels_per_finger
        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.add_finger_embedding = add_finger_embedding
        self.add_positional_encoding = add_positional_encoding

        if add_finger_embedding:
            self.finger_embedding = nn.Embedding(num_fingers, embed_dim)
            # Precompute finger IDs for 64 taxels
            finger_ids = torch.arange(num_fingers).repeat_interleave(taxels_per_finger)
            self.register_buffer("finger_ids", finger_ids)

        if add_positional_encoding:
            self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_taxels, embed_dim))
            nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, x):
        """
        Input:
            x ∈ [B, 64, 6]  (64 taxels × [x, y, z, fx, fy, fz])
        Output:
            tokens ∈ [B, 64, D]
        """

        B, N, _ = x.shape
        assert N == self.num_taxels, f"Expected {self.num_taxels} taxels, got {N}"

        tokens = self.input_proj(x)  # [B, 64, D]

        if self.add_finger_embedding:
            finger_embed = self.finger_embedding(self.finger_ids)  # [64, D]
            tokens = tokens + finger_embed.unsqueeze(0)  # [B, 64, D]

        if self.add_positional_encoding:
            tokens = tokens + self.positional_embedding  # [B, 64, D]

        return tokens
