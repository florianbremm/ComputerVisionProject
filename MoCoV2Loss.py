# moco_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCoV2Loss(nn.Module):
    def __init__(self, feature_dim=2048, queue_size=32768, temperature=0.07, device="cuda"):
        super(MoCoV2Loss, self).__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.device = device

        # Create the memory queue (K, D)
        self.register_buffer("queue", torch.randn(queue_size, feature_dim))
        self.queue = F.normalize(self.queue, dim=1).to(self.device)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Replace oldest keys in the queue with the new ones.
        """
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Replace entries
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            overflow = ptr + batch_size - self.queue_size
            self.queue[ptr:] = keys[:batch_size - overflow]
            self.queue[:overflow] = keys[batch_size - overflow:]

        # Move pointer
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, query, key):
        """
        query: (B, D) — output from encoder (requires_grad)
        key:   (B, D) — output from momentum encoder (no grad)
        """
        # Normalize
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)

        # Positive logits: (B, 1)
        l_pos = torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)

        # Negative logits: (B, K)
        l_neg = torch.einsum('nc,kc->nk', [query, self.queue.clone().detach()])

        # Combine and apply temperature
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        # Labels — positive key is at index 0
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        # Update the queue
        self._dequeue_and_enqueue(key)

        return loss

def concat_all_gather(tensor):
    """
    Performs all_gather across all processes. For single GPU, it's a no-op.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output