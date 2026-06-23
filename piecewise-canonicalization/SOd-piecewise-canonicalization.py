import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def sym(A : torch.Tensor) -> torch.Tensor:
    return A.mT @ A


def psd_matrix_sqrt(A : torch.Tensor) -> torch.Tensor:
    L, Q = torch.linalg.eigh(A) # get eigenvalue decomposition
    L = torch.clamp(L, min=0.0) # clamp eigenvalues to 0 just in case
    L_roots = torch.sqrt(L) # get square roots of eigenvalues
    return Q * L_roots.unsqueeze(-2) @ Q.mT # recompose to get matrix square root


def canonicalize(A : torch.Tensor) -> torch.Tensor:
    assert A.shape[-1] == A.shape[-2], "canonicalize is currently only defined for square matrices"
    # map into the quotient, recording sign
    sign, _ = torch.linalg.slogdet(A)
    symA = sym(A)
    
    # take the section back
    T = torch.ones(A.shape[:-1])
    return T.unsqueeze(-2) * psd_matrix_sqrt(symA)


def generate_dataset(num_samples : int, d : int) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(num_samples, d, d)
    y = torch.linalg.matrix_power(sym(X), 2).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return X, y


class MLP(nn.Module):
    def __init__(self, d, hidden_dims=(128,64)):
        super().__init__()
        input_dim = d * d
        dims = (input_dim,)+hidden_dims
        self.mlp = nn.Sequential()
        for in_dim, out_dim in list(zip(dims[:-1], dims[1:])):
            self.mlp.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(in_features=dims[-1], out_features=1))
    
    def forward(self, x : torch.Tensor):
        x_flat = x.view(x.size(0), -1)
        predictions = self.mlp(x_flat)
        return self.mlp(x_flat).squeeze()


def get_random_rotation(d : int) -> torch.Tensor:
    A = torch.randn(d, d)
    Q, R = torch.linalg.qr(A)
    d_sign = torch.det(Q).sign()
    Q[:, 0] *= d_sign
    return Q


if __name__ == "__main__":
    # torch.manual_seed(42)
    d = 4 
    epochs = 100
    batch_size = 64
    
    # generate data
    X_train, y_train = generate_dataset(num_samples=8000, d=d)
    # X_train = canonicalize(X_train)
    # X_train = sym(X_train)
    X_test, y_test = generate_dataset(num_samples=1000, d=d)
    # X_test = canonicalize(X_test)
    # X_test = sym(X_test)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # initialize model
    model = MLP(d=d, hidden_dims=(32,))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # train
    model.train()
    print("Training...")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        
        print(f"Epoch {epoch+1}/{epochs} | Training MSE Loss: {epoch_loss / len(X_train):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_mse = criterion(test_preds, y_test).item()
        print(f"\nBaseline Test MSE on Unrotated Data: {test_mse:.4f}")
        
        # ------------------------------------------------------------
        # The Vulnerability Test
        # ------------------------------------------------------------
        # Rotate the test set. An invariant target function will yield the exact same y_test.
        # But this network's predictions will wildly change because it is sensitive to coordinate orientation.
        R = get_random_rotation(d)
        X_test_rotated = R.unsqueeze(0) @ X_test
        
        rotated_preds = model(X_test_rotated)
        rotated_mse = criterion(rotated_preds, y_test).item()
        
        print(f"Test MSE on Rotated Data: {rotated_mse:.4f}")
        
        max_diff = torch.max(torch.abs(test_preds - rotated_preds)).item()
        print(f"Max prediction variance caused strictly by rotation: {max_diff:.4f}")