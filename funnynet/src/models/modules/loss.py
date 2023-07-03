import torch
from torch import nn
import torch.nn.functional as F


def cross_entropy(preds, targets, reduction="none"):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class ContrastiveLossELI5(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, feat_2 as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

        def l_ij(i, j):
            # z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            # if self.verbose: print("sim({i}, {j})={sim_i_j}")

            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = (
                torch.ones((2 * self.batch_size,))
                .to(emb_i.device)
                .scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
            )
            # if self.verbose: print("1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            # if self.verbose: print("Denominator", denominator)

            loss_ij = -torch.log(numerator / denominator)
            # if self.verbose: print("loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2 * N) * loss
