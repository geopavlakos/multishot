import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.trans = nn.TransformerEncoderLayer(d_model=2048, nhead=1, dim_feedforward=1024)
        self.pos_embedding = nn.Embedding(embedding_dim=2048, num_embeddings=13)

    def forward(self, x, src_key_padding_mask):
        b, t, e = x.shape
        y = x.permute(1,0,2)
        positions = self.pos_embedding(torch.arange(t, device='cuda'))[None, :, :].expand(b, t, e).permute(1,0,2)/10.
        y1 = self.trans(y+positions, src_key_padding_mask=src_key_padding_mask)
        y = y+y1/10.
        y = y.permute(1,0,2)
        return y
