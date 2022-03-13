from rouge import FilesRouge
import torch

bsz=32
tgt_len=512
mask = torch.full((tgt_len, tgt_len), 0.0)
mask_cond = torch.arange(mask.size(-1))
mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1.0)

msk = mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)