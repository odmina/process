from process.OU_process_1v import OU_process_1v
import numpy as np
import matplotlib.pyplot as plt


def test_means_sds(n, mu, process, delta_t, total_time, full_printout=False):
        X = process.sim_data(n=n, mu=mu, delta_t=delta_t, total_time=total_time)
        print("\n### k =", process.get_k(), "n =", n, " ###")
        print("theoretical vs sampled mean")
        means = np.mean(X, axis=-1)
        means_diffs = means - mu
        print("Number of means that differ more than 0.05",
              np.sum(np.abs(means_diffs) > 0.05))
        if full_printout:
            print(means_diffs)
        print("standard deviations")
        std_diffs = np.std(X, axis=-1)
        print("Number of SDs that differ more than 0.05 form theoretical SD",
              np.sum(np.abs(std_diffs - process.get_sigma()) > 0.05))
        if full_printout:
            print(std_diffs)


delta_t = 1
total_time = 1000
rng = np.random.default_rng()

processes = [OU_process_1v(sigma=a, k=b) for a, b in [(0.1, 5), (0.25, 2), (0.5, 1), (2.5, 0.2), (5, 0.1), (0.5, 1)]]
for process in processes:
    process.set_sim_parameters(delta_t=delta_t, total_time=total_time)
    print(process.get_k())
    print(process.get_sigma())
    print(process.get_sim_discrete_k())
    print(process.get_sim_condPDF_variance())
    print(process.get_stationary_autocorrelation())
    print(process.get_stationary_autocov())

# checks for int number of processes
for process in processes:
    print("-------------------------------------")
    print("k =", process.get_k())
    print("-------------------------------------")
    for n in [1, 10, 100]:
        mu = rng.normal(size=n)
        test_means_sds(n, mu, process, delta_t, total_time, full_printout=True)

# check for tuples n
for process in processes:
    print("-------------------------------------")
    print("k =", process.get_k())
    print("-------------------------------------")
    for n in [(1, 10), (10, 10), (100, 10)]:
        mu = rng.normal(size=n)
        test_means_sds(n, mu, process, delta_t, total_time, full_printout=True)

# checks for some int mu
for process in processes:
    print("------------------------------------")
    print("k =", process.get_k())
    print("------------------------------------")
    n = 100
    for mu in [1, 10, 100]:
        test_means_sds(n, mu, process, delta_t, total_time)


print("------------------------------------")
print("###TIMEPOINT-WISE TESTS###")
print("------------------------------------")
processes = [OU_process_1v(sigma=a, k=b) for a, b in [(1, 0.1), (1, 0.2), (1, 0.5), (0.5, 0.1), (0.5, 0.2), (0.5, 0.5)]]
delta_t = 0.1
total_time = 30
n = 1000
timevec = np.arange(0, total_time, delta_t)

for process in processes:
    process.set_sim_parameters(delta_t=delta_t, total_time=total_time)
    print(process.get_k())
    print(process.get_sigma())
    print(process.get_sim_discrete_k())
    print(process.get_sim_condPDF_variance())
    print(process.get_stationary_autocorrelation())
    print(process.get_stationary_autocov())


# draw some processes
for process in processes:
    print("------------------------------------")
    print("k =", process.get_k())
    print("------------------------------------")
    data = process.sim_data(n=n, delta_t=delta_t, total_time=total_time)
    fig = plt.figure(figsize=(12, 12))
    gs = plt.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    for i in range(n):
        ax1.plot(timevec, data[i])
    violin_data = data[:, [int(1/delta_t) * i for i in range(total_time)]]
    ax2.violinplot(violin_data)
    plt.show()
    print("Number of means that differ more than 0.05 from theoretical mean",
          np.sum(np.abs(np.mean(data, axis=0)) > 0.05))
    print("Number of SDs that differ more than 0.05 from theoretical SD",
          np.sum(np.abs(np.std(data, axis=0, ddof=1)-1) > 0.05))

    all_corrs = np.corrcoef(data, rowvar=False)
    corrs_selected = np.ones((all_corrs.shape[0], 100))
    for i in range(all_corrs.shape[0]):
        if i < all_corrs.shape[0]-100:
            corrs_selected[i] = all_corrs[i, i:i + 100]
        else:
            corrs_selected[i] = np.flip(all_corrs[i][i-100+1:i+1])
    corrs_averaged = corrs_selected.mean(axis=0)
    corrs_timevec = np.arange(0, 100*delta_t, delta_t)
    corrs_theoretical = np.exp(-process.get_k()*corrs_timevec)
    fig, ax = plt.subplots()
    ax.plot(corrs_timevec, corrs_theoretical, c="orange")
    ax.plot(corrs_timevec, corrs_averaged, c="grey")
    plt.show()

