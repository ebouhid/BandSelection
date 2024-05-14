import torch
import torch.nn as nn


class GBC(nn.Module):
    def __init__(self, k=10):
        self.k = k
        super(GBC, self).__init__()
    
    def forward(self, pred, truth):
        TP = torch.sum(pred * truth)
        FN = torch.sum(truth) - TP
        FP = torch.sum(pred) - TP
        norm_O = TP + FN
        numerator = (norm_O - FP)*(norm_O - FN)
        denominator = norm_O**2

        gbc = numerator / denominator

        return gbc
        

class GBCLoss(nn.Module):
    def __init__(self, k=10):
        self.k = k
        super(GBCLoss, self).__init__()

    def forward(self, pred, truth):
        TP = torch.sum(pred * truth)
        FN = torch.sum(truth) - TP
        FP = torch.sum(pred) - TP
        norm_O = TP + FN
        numerator = (norm_O - FP)*(norm_O - FN)
        denominator = norm_O**2
        
        gbc = numerator / denominator
        gbc_loss = 1 - gbc
        return gbc_loss

    def backward(self, pred, truth):
        p_i = pred
        p_j = truth
        z = (p_i - pred * truth) / truth
        
        w = p_j + torch.sum((p_i - p_i * p_j) / p_j)
        dw_dpj = 1 - p_i * torch.sum(torch.pow(z.unsqueeze(-1), torch.arange(1, self.k).float().to(pred.device)) * (1 + torch.arange(1, self.k) / p_j))
        
        dGBC_dpj = (p_i * w - p_i * p_j * dw_dpj) / (w ** 2)
        return -dGBC_dpj

def test_GBC():
    loss_fn = GBCLoss()
    pred = torch.tensor([0.7, 0.3, 0.6], requires_grad=True)
    truth = torch.tensor([1, 0, 1], dtype=torch.float32)

    loss = loss_fn(pred, truth)
    print("GBC Loss:", loss.item())

    loss.backward()
    print("Gradient sample_pred:", pred.grad)

    perfect_pred = torch.tensor([1., 0., 1.], requires_grad=True)
    loss = loss_fn(perfect_pred, truth)
    print("GBC Loss:", loss.item())

    loss.backward()
    print("Gradient perfect_pred:", perfect_pred.grad)


if __name__ == "__main__":
    test_GBC()