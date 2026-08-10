import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

import pickle
import argparse

from datetime import datetime
from enum import Enum


class Group(Enum):
    ORTHOGONAL = "Od"
    SPECIAL_ORTHOGONAL = "SOd"
    SYMMETRIC = "Sn"
    
    def __str__(self):
        return self.value


class MLP(nn.Module):
    def __init__(self,
                 d : int,
                 n : int,
                 hidden_dims : tuple[int] = (128,64),
                 ):
        super().__init__()
        input_dim = d * n
        dims = (input_dim,)+hidden_dims
        self.mlp = nn.Sequential()
        for in_dim, out_dim in list(zip(dims[:-1], dims[1:])):
            self.mlp.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(in_features=dims[-1], out_features=1))
    
    def forward(self, x : torch.Tensor):
        # x_flat = x.view(x.size(0), -1)
        x_flat = torch.flatten(x, start_dim=1)
        predictions = self.mlp(x_flat)
        return self.mlp(x_flat).squeeze()


def sym(A : torch.Tensor) -> torch.Tensor:
    return A.mT @ A


def psd_matrix_sqrt(A : torch.Tensor) -> torch.Tensor:
    L, Q = torch.linalg.eigh(A) # get eigenvalue decomposition
    L = torch.clamp(L, min=0.0) # clamp eigenvalues to 0 just in case
    L_roots = torch.sqrt(L) # get square roots of eigenvalues
    return Q * L_roots.unsqueeze(-2) @ Q.mT # recompose to get matrix square root


def canonicalize(A : torch.Tensor,
                 group : Group, # which group to use for canonicalization and quotienting: 
                 randomize_columns : bool = False, # whether to randomly select d columns from the n columns of A when d < n or to take the first d columns of A
                 ) -> torch.Tensor:
    # square matrix case
    b = A.shape[:-2]
    d = A.shape[-2]
    n = A.shape[-1]
    if group in [Group.ORTHOGONAL, Group.SPECIAL_ORTHOGONAL]:
        if d == n:
            # map into the quotient, recording sign
            sign, _ = torch.linalg.slogdet(A)
            symA = sym(A)
            
            # take the section back
            F = torch.eye(d, device=A.device, dtype=A.dtype).expand(*b, -1, -1).clone()
            if group == Group.SPECIAL_ORTHOGONAL:
                F[...,-1,-1] = sign
            return F @ psd_matrix_sqrt(symA)
        elif d < n:
            if randomize_columns:
                # sample d unique columns from the n columns of A
                alpha = torch.randperm(n)[:d]
            else:
                # take the first d columns of A
                alpha = torch.arange(d)
            A_alpha = A[..., alpha]
            sign, _ = torch.linalg.slogdet(A_alpha)
            symA_alpha = sym(A_alpha)
            F = torch.eye(d, device=A.device, dtype=A.dtype).expand(*b, -1, -1).clone()
            if group == Group.SPECIAL_ORTHOGONAL:
                F[...,-1,-1] = sign
            return F @ psd_matrix_sqrt(symA_alpha) @ A_alpha.mT @ A
        else:
            raise NotImplementedError("canonicalization is only implemented for square matrices and wide matrices")
    else:
        raise NotImplementedError("canonicalization is currently only implemented for the orthogonal and special orthogonal groups")


def quotient(A : torch.Tensor,
             group : Group,
             ) -> torch.Tensor:
    if group == Group.SPECIAL_ORTHOGONAL:
        # force square matrix
        A_alpha = A[..., :A.shape[-2]]
        sign, _ = torch.linalg.slogdet(A_alpha)
        torch.linalg.svd(A)
        return sign[:, None, None] * sym(A)
    elif group == Group.ORTHOGONAL:
        return sym(A)
    else:
        raise NotImplementedError("quotienting is currently only implemented for the orthogonal and special orthogonal groups")


def martrix_polynomial(A : torch.Tensor, coefficients : tuple[float,...]):
    return sum(c * torch.linalg.matrix_power(A,n) for n, c in enumerate(coefficients))


def generate_dataset(num_samples : int,
                     d : int,
                     n : int,
                     polynomial : tuple[float,...],
                     group : Group,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(num_samples, d, n)
    y = martrix_polynomial(quotient(X, group=group), polynomial).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return X, y


def get_random_actions(batch_size : int,
                         d : int,
                         group : Group,
                         ) -> torch.Tensor:
    if group in [Group.ORTHOGONAL, Group.SPECIAL_ORTHOGONAL]:
        A = torch.randn(batch_size, d, d)
        Q, R = torch.linalg.qr(A)
        if group == Group.SPECIAL_ORTHOGONAL:
            d_sign = torch.det(Q).sign()
            Q[:,:,-1] *= d_sign[:, None] # multiply sign on last column to force determinant to be 1
        return Q
    else:
        raise NotImplementedError("random actions are currently only implemented for the orthogonal and special orthogonal groups")


def train_test(d : int = 3,
               n : int = 5,
               num_epochs : int = 1000,
               batch_size : int = 128,
               mode : str = "none",
               mlp_hidden_dims : tuple[int] = None,
               polynomial : tuple[int] = (0,2),
               group : Group = Group.ORTHOGONAL,
               ) -> tuple[list[float], list[float], list[float]]:
    assert mode in ["none", "canonicalize", "quotient"], f"mode must be one of 'none', 'canonicalize', or 'quotient', but got {mode}"
    if mlp_hidden_dims is None and mode == "quotient":
        mlp_hidden_dims = (n*n, n*n)
    else:
        mlp_hidden_dims = (d*n, d*n)
    # generate data
    X, Y = generate_dataset(num_samples=10000, d=d, n=n, polynomial=polynomial, group=group)
    X_train, y_train = X[:8000], Y[:8000]
    X_test, y_test = X[8000:], Y[8000:]
    R = get_random_actions(len(X_test), d, group=group)
    X_test_rotated = R @ X_test
    if mode == "canonicalize":
        X_train = canonicalize(X_train, group=group)
        X_test = canonicalize(X_test, group=group)
        X_test_rotated = canonicalize(X_test_rotated, group=group)
    elif mode == "quotient":
        X_train = quotient(X_train, group=group)
        X_test = quotient(X_test, group=group)
        X_test_rotated = quotient(X_test_rotated, group=group)

    # X_test = sym(X_test)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # initialize model
    if mode == "quotient":
        model = MLP(d=n, n=n, hidden_dims=mlp_hidden_dims)
    else:
        model = MLP(d=d, n=n, hidden_dims=mlp_hidden_dims)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    train_losses = list()
    test_losses = list()
    rotated_test_losses = list()
    print("Training...")
    for epoch in range(num_epochs):
        # train
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        train_mse = epoch_loss / len(X_train)
        
        # test
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            test_mse = criterion(test_preds, y_test).item()
            rotated_preds = model(X_test_rotated)
            rotated_test_mse = criterion(rotated_preds, y_test).item()
        
        train_losses.append(train_mse)
        test_losses.append(test_mse)
        rotated_test_losses.append(rotated_test_mse)
        print(f"Epoch {epoch+1:03d}/{num_epochs}")
        print("==============================")
        print(f"Train Loss: {train_mse:.4f}")
        print(f"Test Loss: {test_mse:.4f}")
        print(f"Rotated Test Loss: {rotated_test_mse:.4f}")
        print("\033[5A", end="")
    print(end="\n\n\n\n\n")
    print("Done!")
    print()
    
    return train_losses, test_losses, rotated_test_losses


def save_losses(losses_tuple : tuple[list[float],...],
                filename : str
                ) -> None:
    with open(filename, "wb") as f:
        pickle.dump(losses_tuple, f)


def load_losses(filename : str) -> None:
    with open(filename, "rb") as f:
        return pickle.load(f)


def generate_comparison(d : int = 3,
                        n : int = 10,
                        num_epochs : int = 200,
                        batch_size : int = 250,
                        mlp_hidden_dims : tuple[int] = None,
                        polynomial : tuple[float,...] = (0,1,1,),
                        group : Group = Group.ORTHOGONAL,
                        save : bool = True,
                        show : bool = False,
                        ) -> None:
    mlp_hidden_dims = mlp_hidden_dims if mlp_hidden_dims is not None else (d*n, d*n)
    train_losses, test_losses, rotated_test_losses = train_test(
        d=d,
        n=n,
        num_epochs=num_epochs,
        batch_size=batch_size,
        mode="none",
        mlp_hidden_dims=mlp_hidden_dims,
        polynomial=polynomial,
        group=group,
    )

    ctrain_losses, ctest_losses, crotated_test_losses = train_test(
        d=d,
        n=n,
        num_epochs=num_epochs,
        batch_size=batch_size,
        mode="canonicalize",
        mlp_hidden_dims=mlp_hidden_dims,
        polynomial=polynomial,
        group=group,
    )

    qtrain_losses, qtest_losses, qrotated_test_losses = train_test(
        d=d,
        n=n,
        num_epochs=num_epochs,
        batch_size=batch_size,
        mode="quotient",
        mlp_hidden_dims=(n*n, n*n),
        polynomial=polynomial,
        group=group,
    )

    # graph performance
    epochs = range(1, num_epochs + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(18,4))
    ax1.plot(epochs, train_losses, label="Train Loss", color="red")
    ax1.plot(epochs, test_losses, label="Test Loss", color="blue")
    ax1.plot(epochs, rotated_test_losses, label="Rotated Test Loss", color="yellow")
    ax2.plot(epochs, ctrain_losses, label="Train Loss", color="red")
    ax2.plot(epochs, ctest_losses, label="Test Loss", color="blue")
    ax2.plot(epochs, crotated_test_losses, label="Rotated Test Loss", color="yellow")
    ax3.plot(epochs, qtrain_losses, label="Train Loss", color="red")
    ax3.plot(epochs, qtest_losses, label="Test Loss", color="blue")
    ax3.plot(epochs, qrotated_test_losses, label="Rotated Test Loss", color="yellow")
    
    # subplot axis labels
    ax1.set_title("Regular")
    ax1.set_xlabel('Epochs')
    ax2.set_title("Canonicalized")
    ax2.set_xlabel('Epochs')
    ax3.set_title("Quotiented")
    ax3.set_xlabel('Epochs')
    ax1.tick_params(axis='y', labelleft=True)
    ax2.tick_params(axis='y', labelleft=True)
    ax3.tick_params(axis='y', labelleft=True)
    fig.supylabel('MSE Loss', x=0.005)
    # figure title
    fig.suptitle("Train vs Test vs Rotated Test Loss", y=0.98)
    # set subtitle to include training specs in italics
    fig.text(0.5, 0.90, f"d={d}, n={n}, epochs={num_epochs}, batch_size={batch_size}, mlp_hidden_dims={mlp_hidden_dims}, polynomial={polynomial}, group={group.value}", ha='center', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 1])
    # enable legends and grid lines
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.grid(which='major', linestyle='-')
    ax1.grid(which='minor', linestyle=':', alpha=0.5)
    ax2.grid(which='major', linestyle='-')
    ax2.grid(which='minor', linestyle=':', alpha=0.5)
    ax3.grid(which='major', linestyle='-')
    ax3.grid(which='minor', linestyle=':', alpha=0.5)

    # grid formatting, shared since sharey=True
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=20))
    log = True
    if log:
        ax1.set_yscale("log")
    else:
        ax1.set_ylim(bottom=0)
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))

    if save:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        filename = f"piecewise-canonicalization/{timestamp}_{d}-d_{n}-n_{num_epochs}-epochs_{polynomial}-poly_group-{group.value}"
        save_losses(losses_tuple=(train_losses,  test_losses,  rotated_test_losses,
                                  ctrain_losses, ctest_losses, crotated_test_losses,
                                  qtrain_losses, qtest_losses, qrotated_test_losses),
                    filename=filename+".pkl")
        plt.savefig(filename+".png")
    if show:
        plt.show()


if __name__ == "__main__":
    # torch.manual_seed(42)
    #parser = argparse.ArgumentParser(
    #    prog="piecewise-canonicalization",
    #    description="Train and test a neural network on a piecewise polynomial function of the symmetric part of a matrix, with options for canonicalization and quotienting.",
    #    epilog="Developed by Nicholas Kenschaft as part of a Master's thesis project in Mathematical Sciences under the supervision of Professor Florian Frick, 2026"
    #)
    #parser.add_argument("--d", type=int, default=3, help="dimension of the points (default: 3)")
    #parser.add_argument("--n", type=int, default=10, help="number of points (default: 10)")
    #parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train (default: 100)")
    #parser.add_argument("--batch_size", type=int, default=250, help="batch size for training (default: 250)")
    #parser.add_argument("--mlp_hidden_dims", type=int, nargs="+", default=None, help="hidden dimensions for the MLP (default: (d*n, d*n))")
    #parser.add_argument("--polynomial", type=float, nargs="+", default=(0,1,1,), help="coefficients of the matrix polynomial used for generating artificial train/test data (default: (0,1,1))")
    #parser.add_argument("--group", type=str, default="orthogonal", choices=["orthogonal", "special orthogonal", "symmetric"], help="which group to use for canonicalization and quotienting (default: orthogonal)")
    #
    #d = parser.parse_args().d
    #n = parser.parse_args().n
    #num_epochs = parser.parse_args().num_epochs
    #batch_size = parser.parse_args().batch_size
    #mlp_hidden_dims = tuple(parser.parse_args().mlp_hidden_dims) if parser.parse_args().mlp_hidden_dims is not None else (d*n, d*n)
    #polynomial = tuple(parser.parse_args().polynomial)
    #group = parser.parse_args().group
    for d in [2,3,5]:
        for n in [d, d+1, 10]:
            print(f"Running for d={d}, n={n}...")
            generate_comparison(d=d, n=n, num_epochs=250, batch_size=500, mlp_hidden_dims=(128,), polynomial=(0,1,1,), group=Group.SPECIAL_ORTHOGONAL, save=True, show=False)
            print(f"Done for d={d}, n={n}!")
            print("\n\n\n\n\n")