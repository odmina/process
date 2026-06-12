from process.white_gaussian_noise import WhiteGaussianNoise
import numpy as np
from pytest import MonkeyPatch
import builtins
from scipy.stats import kstest, ttest_1samp, norm

# the last test uses statistical tests :)

def green(s):
    return '\033[1;32m%s\033[m' % s

def yellow(s):
    return '\033[1;33m%s\033[m' % s

def red(s):
    return '\033[1;31m%s\033[m' % s

def log(*m):
    print(" ".join(map(str, m)))

# %%
def check_sigma(sigma=1):
    ex_name = "check sigma"
    sigma_process = WhiteGaussianNoise(sigma=sigma)
    test_sigma = sigma_process.get_sigma()
    if sigma==test_sigma:
        log(green("PASS"), ex_name)
    else:
        log(red("FAIL"), ex_name)

# %%
monkeypatch = MonkeyPatch()
def check_sim_params_setter(monkeypatch):
    ex_name= "Simulation parameters"
    monkeypatch.setattr(builtins, "input", lambda _: "y")
    params_process = WhiteGaussianNoise(sigma=1)
    rng = np.random.default_rng()
    deltas = [1, 0.1, 0.01]
    times = [10, 100]
    for delta in deltas:
        for time in times:
            params_process.set_sim_parameters(delta_t=delta, total_time=time)
            data = params_process.sim_data(rng=rng)
            if data.shape == (1, int(time/delta)):
                log(green("PASS"), ex_name, "for", "delta_t", delta, "total_time", time)
            else:
                log(red("FAIL"), ex_name, "for", "delta_t", delta, "total_time", time)

# %%
def check_simulated_data_distribution():
    ex_name = "distribution checks"
    processes = [WhiteGaussianNoise(sigma=0.5+a*0.5) for a in [0,1,2]]
    delta_t = 1
    total_time = 1000
    mu = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
    n = (5, 2)
    rng = np.random.default_rng()
    p_values = []
    n_runs=1000
    for run in range(n_runs):
        for process in processes:
            process.set_sim_parameters(delta_t=delta_t, total_time=total_time)
            sim_data = process.sim_data(n=n, mu=mu, rng=rng)
            scale = process.get_sigma()
            for i in range(n[0]):
                for j in range(n[1]):
                    test_results = kstest(sim_data[i,j], norm.cdf, args=(mu[i,j], scale))
                    p = test_results.pvalue
                    p_values.append(p)

    p_values = np.array(p_values)
    n_sig = np.sum(p_values < 0.05)
    max_different = len(processes)*mu.size*n_runs*0.05
    if n_sig > len(processes)*mu.size*n_runs*0.05:
        log(yellow("CAUTION"), ex_name, "expected different", max_different, "actual different", n_sig)
    else:
        log(green("PASS"), ex_name, "expected different", max_different, "actual different", n_sig)


