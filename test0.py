import torch
import torch.nn as nn

transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
print(src.size())
tgt = torch.rand((20, 32, 512))
print(tgt.size())
out = transformer_model(src, tgt)
print(out)