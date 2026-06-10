from sys import argv
from matplotlib import pyplot as plt

def get_Gr_n_3_cuplns(max_s=10):
    cup_lns = [None]*(2**(max_s+1))
    for s in range(1, max_s+1):
        n = 2**(s+1)
        if n == 5:
            print(f"found 5! s={s}")
        cup_lns[n - 1] = 2**(s+2) - 5
        for p in range(1,s+1):
            n = 2**(s+1) - 2**p + 1
            if n == 5:
                print(f"found 5! s={s} p={p}")
            cup_lns[n - 1] = 2**(s+2) - 3*2**(p-1) - 4
            for t in range(0,(2**(p-1)-2)+1):
                n = 2**(s+1) - 2**p + 2 + t
                if n == 5:
                    print(f"found 5! s={s} p={p} t={t}")
                cup_lns[n - 1] = 2**(s+2) - 3*2**(p-1) - 2 + t
    return cup_lns


def get_Gr_n_4_cuplns(max_s=10):
    cup_lns = [None]*(2**(max_s+1))
    for s in range(1,max_s + 1):
        n = 2**s + 1
        cup_lns[n - 1] = 2**(s+1) + 2**s - 7
        for r in range(0,s):
            for t in range(0,(2**r - 1) + 1):
                n = 2**s + 2**r + t + 1
                cup_lns[n - 1] = 2**(s+1) + 2**s + 2**(r+1) + t - 7
    return cup_lns


def plot_cup_lns(k, max_s):
    if k == 3:
        cup_lns = get_Gr_n_3_cuplns(max_s=max_s)
    if k == 4:
        cup_lns = get_Gr_n_4_cuplns(max_s=max_s)
    x = [i+1 for i in range(len(cup_lns))]
    y = cup_lns
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.xlabel("n")
    plt.ylabel(f"Cup Length of Gr(n,{k})")
    title = f"Cup Lengths of Gr(n,{k}) for n <={2**(max_s+1)}"
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(title + ".png")


if __name__ == "__main__":
    # k = int(argv[1])
    max_s = int(argv[2])
    # plot_cup_lns(k=k, max_s=max_s)
    # plot_cup_lns(k=k, max_s=max_s)
    cup_lns3 = get_Gr_n_3_cuplns(max_s=max_s)
    cup_lns4 = get_Gr_n_4_cuplns(max_s=max_s)
    x = [i+1 for i in range(len(cup_lns3))]
    plt.plot(x, cup_lns3, marker='o', linestyle='-', color='b', label="cup(Gr(n,3))")
    plt.plot(x, cup_lns4, marker='o', linestyle='-', color='r', label="cup(Gr(n,4))")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("n")
    plt.ylabel(f"Cup Length of Gr(n,k)")
    title = f"Cup Lengths of Gr(n,k) for n <={2**(max_s+1)}, k=3,4"
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(title + ".png")