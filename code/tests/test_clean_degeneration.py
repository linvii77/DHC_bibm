import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from proxy_loss import ProjectionHead, CompositionalSimilarityLoss

B, C_in, D, H, W = 2, 256, 8, 16, 16
emb_dim, num_cls = 256, 14

def rand_targets():
    return torch.randint(0, num_cls, (B, D, H, W))

# (a) use_variation=False → variation_vectors is None
cs_no = CompositionalSimilarityLoss(num_cls, embedding_dim=emb_dim, use_variation=False)
assert cs_no.variation_vectors is None, "FAIL (a): variation_vectors should be None"

# (b) use_variation=False → forward runs, loss backward-able
emb = torch.randn(B, emb_dim, D, H, W, requires_grad=True)
loss_no, stats_no = cs_no(emb, rand_targets())
loss_no.backward()
assert stats_no.p_sub_entropy == 0.0, "FAIL (b): p_sub_entropy should be 0.0 when no variation"

# (c) use_variation=True → variation_vectors.grad is non-None with norm > 0
cs_yes = CompositionalSimilarityLoss(num_cls, embedding_dim=emb_dim, use_variation=True)
emb2 = torch.randn(B, emb_dim, D, H, W, requires_grad=True)
loss_yes, stats_yes = cs_yes(emb2, rand_targets())
loss_yes.backward()
assert cs_yes.variation_vectors.grad is not None, "FAIL (c): variation_vectors.grad is None"
assert cs_yes.variation_vectors.grad.norm() > 0, "FAIL (c): variation_vectors.grad norm is zero"

# (d) Both configs: proxy_dist.grad is non-None with norm > 0
assert cs_no.proxy_dist.grad is not None and cs_no.proxy_dist.grad.norm() > 0, "FAIL (d) no-var: proxy_dist dead"
assert cs_yes.proxy_dist.grad is not None and cs_yes.proxy_dist.grad.norm() > 0, "FAIL (d) var: proxy_dist dead"

# (e) p_sub_entropy is non-zero when use_variation=True
assert stats_yes.p_sub_entropy > 0.0, "FAIL (e): p_sub_entropy should be positive with variation"

print("All assertions passed.")
print(f"  p_sub_entropy (variation=True):  {stats_yes.p_sub_entropy:.4f}")
print(f"  p_sub_entropy (variation=False): {stats_no.p_sub_entropy:.4f}")
print(f"  proxy_dist grad norm (no-var):   {cs_no.proxy_dist.grad.norm().item():.4e}")
print(f"  proxy_dist grad norm (var):      {cs_yes.proxy_dist.grad.norm().item():.4e}")
print(f"  variation_vectors grad norm:     {cs_yes.variation_vectors.grad.norm().item():.4e}")
