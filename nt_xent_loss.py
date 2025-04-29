import torch
import torch.nn as nn
import torch.nn.functional as F

# https://colab.research.google.com/drive/1UK8BD3xvTpuSj75blz8hThfXksh19YbA?usp=sharing#scrollTo=4ZjswK4oT4qJ
def nt_xent_loss(out_1, out_2, temperature=0.5, eps=1e-6):
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)
    
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)
    
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)
    
    # Positive similarity
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()
    return loss