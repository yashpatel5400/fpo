import numpy as np
import pickle
import pandas as pd
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

import utils
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set_theme()

mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class SpecOp(nn.Module):
    def __init__(self, k_in, k_out):
        super().__init__()

        hidden_features = 16
        self.linear1 = nn.Linear(k_in, hidden_features)
        self.linear2 = nn.Linear(hidden_features, hidden_features)
        self.linear3 = nn.Linear(hidden_features, k_out)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def get_sobolev_weights(s, gamma, shape):
    coords = cartesian_product(
        np.array(range(shape[0])), 
        np.array(range(shape[1]))
    ).reshape((shape[0], shape[1], 2))
    ks = np.sum(coords, axis=-1)
    d = len(shape)
    return (1 + ks ** (2 * d)) ** (s - gamma)

    
def sobolev_cp_cov(u_hat, u_cal, K, s, gamma, alphas=[0.05], cal_size=150):
    alphas = np.array(alphas)
    sobolev_scaling = get_sobolev_weights(s, gamma, u_hat.shape[1:])
    K_full = 150
    
    full_sobolev_residual = (sobolev_scaling * (u_hat - u_cal) ** 2)[:,:K_full,:K_full]
    full_sobolev_norm = full_sobolev_residual.reshape(-1, np.prod(full_sobolev_residual.shape[1:])).sum(axis=-1)
    np.random.shuffle(full_sobolev_norm)
    
    full_cal_norms, full_test_norms = full_sobolev_norm[:cal_size], full_sobolev_norm[cal_size:]
    q_hat_stars = np.quantile(full_cal_norms, 1-alphas)
    
    truncated_sobolev_residual = full_sobolev_residual[:,:K,:K]
    truncated_sobolev_norm = truncated_sobolev_residual.reshape(-1, K * K).sum(axis=-1)
    truncated_cal_norms, truncated_test_norms = truncated_sobolev_norm[:cal_size], truncated_sobolev_norm[cal_size:]
    q_hats = np.quantile(truncated_cal_norms, 1-alphas)
    
    # NOTE: these coverages are the standard CP guarantee -- we only use for debugging
    tiled_full_test_norms      = einops.repeat(full_test_norms, "n -> n repeat", repeat=len(alphas))
    tiled_truncated_test_norms = einops.repeat(truncated_test_norms, "n -> n repeat", repeat=len(alphas))
    debug_full_coverage        = einops.reduce((tiled_full_test_norms < q_hat_stars) / len(tiled_full_test_norms), "n repeat -> repeat", reduction="sum")
    debug_truncated_coverage   = einops.reduce((tiled_truncated_test_norms < q_hats) / len(truncated_test_norms), "n repeat -> repeat", reduction="sum")
    # print(f"[DEBUG] Truncated Coverage: {debug_truncated_coverage} vs {1 - alphas}  |  Full Coverage: {debug_full_coverage} vs {1 - alphas}")
    
    full_coverages = einops.reduce((tiled_full_test_norms < q_hats) / len(tiled_full_test_norms), "n repeat -> repeat", reduction="sum")
    delta_qs = (q_hat_stars - q_hats) / q_hat_stars
    
    n = np.prod(full_sobolev_residual.shape[1:])
    ellipsoid_vol = n / 2 * np.log(q_hat_stars[0]) - np.sum(np.log(sobolev_scaling)) / 2
    
    return delta_qs, ellipsoid_vol, full_coverages


def calibrate(pde):
    with open(utils.DATA_FN(pde), "rb") as f:
        (fs, us) = pickle.load(f)

    net  = SpecOp(fs.shape[-1], us.shape[-1])
    net.load_state_dict(torch.load(utils.MODEL_FN(pde), weights_only=True))
    net.eval().to("cuda")

    prop_train = 0.75
    N = fs.shape[0]
    N_train = int(N * prop_train)

    f_cal = torch.from_numpy(fs[N_train:]).to(torch.float32).to("cuda")
    u_cal = us[N_train:]

    u_hat = net(f_cal).cpu().detach().numpy()
    u_hat = np.transpose(u_hat.reshape((u_hat.shape[0], 256, 256)), (0, 2, 1))
    u_cal = np.transpose(u_cal.reshape((u_hat.shape[0], 256, 256)), (0, 2, 1))
    s = 2 # working w/ Laplacian in PDE immediately imposes s = 2 smoothness

    alphas = np.arange(0.05, 1, 0.05)
    gamma_eps = 0.1
    gammas = np.arange(1, s + gamma_eps, gamma_eps)
    
    volume_df = pd.DataFrame(columns=["trial", "gamma", "volume"])
    for gamma in gammas:
        coverage_df = pd.DataFrame(columns=["trial", "alpha", "coverage"])
        delta_q_df  = pd.DataFrame(columns=["trial", "K", "delta_q"])

        alphas = np.arange(0.05, 1, 0.05)
        trials = range(1)
        for trial in trials:
            _, volume, coverages = sobolev_cp_cov(u_hat, u_cal, K=125, s=s, gamma=gamma, alphas=alphas)
            for alpha, coverage in zip(alphas, coverages):
                coverage_df.loc[-1] = [trial, 1-alpha, coverage]
                coverage_df.index = coverage_df.index + 1
                coverage_df = coverage_df.sort_index()
            
            volume_df.loc[-1] = [trial, gamma, volume]
            volume_df.index = volume_df.index + 1
            volume_df = volume_df.sort_index()

            # Ks = np.arange(5, 100, 5)
            # for K in Ks:
            #     delta_q, _ = sobolev_cp_cov(u_hat, u_cal, K=125, s=s, gamma=gamma, alphas=[0.05])
            #     delta_q_df.loc[-1] = [trial, K, delta_q[0]]
            #     delta_q_df.index = delta_q_df.index + 1
            #     delta_q_df = delta_q_df.sort_index()
        
        # plt.title(r"$K$ vs $\Delta\widehat{q}$ ($\gamma = " + str(gamma) + "$)")
        # plt.xlabel(r"$K$")
        # plt.ylabel(r"$\widehat{q}^{*}_{\gamma} - \widehat{q}_{\gamma}$")
        # sns.lineplot(delta_q_df, x="K", y="delta_q")
        # plt.savefig(os.path.join(pde, f"delta_q_gamma={gamma}.png"))
        # plt.tight_layout()
        # plt.clf()

        plt.title(r"$\mathrm{Calibration}$")
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\mathrm{Coverage}$")
        sns.lineplot(coverage_df, x="alpha", y="coverage", label=r"$\gamma=" + "{:.1f}".format(gamma) + "$")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(pde, f"coverage.png"))
    plt.clf()

    plt.title(r"$\gamma \mathrm{\ vs\ Volume}$")
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$\log(\mathrm{Vol})$")
    sns.lineplot(volume_df, x="gamma", y="volume")

    plt.tight_layout()
    plt.savefig(os.path.join(pde, f"volume.png"))
    plt.clf()


if __name__ == "__main__":
    pde = "poisson"
    calibrate(pde)