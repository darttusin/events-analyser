import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t())  
    sim = sim / temperature

    pos_indices = torch.arange(batch_size, device=z1.device) + batch_size
    labels = torch.cat([pos_indices, torch.arange(0, batch_size, device=z1.device)])

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    sim.masked_fill_(mask, -9e15)

    loss = F.cross_entropy(sim, labels)
    return loss


def classical_contrastive_loss(z1, z2, margin=1.0):
    batch_size = z1.size(0)
    device = z1.device
    z = torch.cat([z1, z2], dim=0)
    dist_matrix = torch.cdist(z, z, p=2)
    positive_mask = torch.zeros(
        (2 * batch_size, 2 * batch_size), dtype=torch.bool, device=device
    )
    for i in range(batch_size):
        positive_mask[i, i + batch_size] = True
        positive_mask[i + batch_size, i] = True

    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    negative_mask = ~(positive_mask | self_mask)

    pos_loss = 0.5 * (dist_matrix[positive_mask] ** 2)
    neg_loss = 0.5 * (torch.clamp(margin - dist_matrix[negative_mask], min=0.0) ** 2)
    loss = (pos_loss.sum() + neg_loss.sum()) / (2 * batch_size)
    return loss


def classical_contrastive_loss(z1, z2, labels, margin=1.0):
    # print(z1.shape, z2.shape, labels.shape)
    distances = torch.norm(z1 - z2, p=2, dim=1)
    positive_loss = labels * 0.5 * distances**2
    negative_loss = (1 - labels) * 0.5 * F.relu(margin - distances) ** 2
    loss = positive_loss + negative_loss
    return loss.mean()