import torch
import torch.nn.functional as F

class GaussianHMM:
    def __init__(self, n_states, n_features, device='cuda'):
        self.n_states = n_states
        self.n_features = n_features
        self.device = device

        # 初始機率、轉移機率
        self.pi = torch.rand(n_states, device=device)
        self.pi = self.pi / self.pi.sum()

        self.A = torch.rand(n_states, n_states, device=device)
        self.A = self.A / self.A.sum(dim=1, keepdim=True)

        # 每個狀態對應的高斯參數
        self.means = torch.randn(n_states, n_features, device=device)
        self.covars = torch.stack([torch.eye(n_features, device=device) for _ in range(n_states)])

    def multivariate_gaussian(self, x, mean, cov):
        k = mean.shape[0]
        diff = x - mean
        inv_cov = torch.linalg.inv(cov)
        exponent = -0.5 * torch.einsum('...i,ij,...j->...', diff, inv_cov, diff)
        denom = torch.sqrt((2 * torch.pi) ** k * torch.linalg.det(cov))
        return torch.exp(exponent) / (denom + 1e-6)

    def emission_probs(self, X):
        """
        X: (T, n_features)
        return: (T, n_states)
        """
        T = X.shape[0]
        probs = torch.zeros((T, self.n_states), device=self.device)
        for s in range(self.n_states):
            probs[:, s] = self.multivariate_gaussian(X, self.means[s], self.covars[s])
        return probs

    def viterbi(self, X):
        """
        X: (T, n_features)
        return: (T,) 隱藏狀態序列
        """
        T = X.shape[0]
        B = self.emission_probs(X)
        delta = torch.zeros((T, self.n_states), device=self.device)
        psi = torch.zeros((T, self.n_states), dtype=torch.long, device=self.device)

        delta[0] = torch.log(self.pi + 1e-8) + torch.log(B[0] + 1e-8)

        for t in range(1, T):
            for j in range(self.n_states):
                temp = delta[t - 1] + torch.log(self.A[:, j] + 1e-8)
                psi[t, j] = temp.argmax()
                delta[t, j] = temp.max() + torch.log(B[t, j] + 1e-8)

        # 回溯
        states = torch.zeros(T, dtype=torch.long, device=self.device)
        states[-1] = delta[-1].argmax()
        for t in reversed(range(T - 1)):
            states[t] = psi[t + 1, states[t + 1]]

        return states
