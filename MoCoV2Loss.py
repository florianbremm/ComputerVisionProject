# moco_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCoV2Loss(nn.Module):
    """
    class for the MoCo V2 loss
    """
    def __init__(self, feature_dim=2048, queue_size=32768, temperature=0.07, device="cuda"):
        super(MoCoV2Loss, self).__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.device = device

        # Create the memory queue (K, D) and initialize it with normal distribution
        self.register_buffer("queue", torch.randn(queue_size, feature_dim))
        self.queue = F.normalize(self.queue, dim=1).to(self.device)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
         Replace the oldest keys in the queue with the new ones.
        :param keys: torch.tensor containing the current batch
        :return:
        """

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Replace entries
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        # Extra logic if pointer is close to the end of the queue to put the rest of keys at the begining again
        else:
            overflow = ptr + batch_size - self.queue_size
            self.queue[ptr:] = keys[:batch_size - overflow]
            self.queue[:overflow] = keys[batch_size - overflow:]

        # Move pointer
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, query, key):
        """
        compute the InfoCNE Loss and update the queue
        :param query: (B, D) — output from encoder
        :param key: (B, D) — output from momentum encoder
        :return:
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