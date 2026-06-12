import numpy as np
import warnings


class WhiteGaussianNoise(object):
    """
    This class defines one variable white noise process.
    Provided for compatibility purposes (with OU process).

    Args:
        `sigma`: STANDARD DEVIATION of the process, has to be > 0.

    During data simulation parameter `mu` may be provided.
        It is the mean of the Gaussian, it is 0 by default.

    The simulation uses discrete timesteps of equal length `delta_t`
        and at each timestep samples X(t) from a normal distribution
        with mean mu and variance sigma^2.

    Simulation runs for `total_time` units with 1/'d' samples per unit.

   The initial value of the process X(0) is sampled from a normal
        distribution with mean `mu` and variance sigma^2.
    """

    def __init__(self, sigma):
        assert sigma >= 0, """SD should be >= 0"""
        self.sigma = sigma

        self.delta_t = None
        self.total_time = None

    def __str__(self):
        return """Gaussian white noise process;
        use `get_sigma()` to access parameters."""

    # ------ GETTERS ------
    def get_sigma(self):
        return self.sigma

    # # ------ SETTERS ----------
    # def set_sigma(self, sigma, silent=False):
    #     assert sigma > 0, "sigma must be > 0"
    #     if self.sigma is None:
    #         self.sigma = sigma
    #     else:
    #         if silent:
    #             self.sigma = sigma
    #         else:
    #             print("`sigma` was already set.")
    #             cont = input("Type y if you want to overwrite it: ")
    #             if cont.strip() == "y":
    #                 self.sigma = sigma

    # ------ SIMULATION -------
    def set_sim_parameters(self, delta_t=None, total_time=None):
        """Set simulation delta_t (timestep) and total_time."""
        if delta_t is not None:
            assert delta_t > 0, "delta_t must be > 0"
            if self.delta_t is None:
                self.delta_t = delta_t
            else:
                cont = input("`delta_t` was already set. Type y if you want to overwrite it: ")
                if cont.strip() == "y":
                    self.delta_t = delta_t

        if total_time is not None:
            assert total_time > 0, "total_time must be > 0"
            if self.total_time is None:
                self.total_time = total_time
            else:
                cont = input("`total_time` was already set. Type y if you want to overwrite it: ")
                if cont.strip() == "y":
                    self.total_time = total_time

        if delta_t is None and total_time is None:
            warnings.warn("Nothing to set!")

    def sim_data(self, n=1, mu=0, delta_t=None, total_time=None, rng=None):
        """
        This function simulates a white noise process.

        Args:
            n: Number of processes. If processes need to be grouped, n can be a tuple. For example:
                for n=(2,3), two groups of three processes are going to be generated.
            delta_t: timestep. Has to be 1/i, where i is an integer.
            total_time: total simulation time.
            mu: mean of the simulated process. Default 0.
                Can be a list of or array with n elements.
            rng: the function can use global random number generator. Default None.

        Returns: numpy array containing simulated data with shape
            (n, len(np.arange(0, total_time, delta_t)))
            Returned array will have len(n) + 1 dimensions.

        Simulation runs for `total_time` units
            with 1/'delta_t' samples per unit, each sample being a gaussian rv with
            mean `mu` and variance `sigma`.
        """

        # assert self.sigma is not None, "You have to set standard deviation (sigma) first!"

        if isinstance(n, tuple):
            assert isinstance(n[0], int) and isinstance(n[1], int), "number of processes must be an int"
            assert n[0] > 0 and n[1] > 0, "number of processes must be > 0"
        elif isinstance(n, int):
            assert n > 0, "number of processes must be > 0"
        else:
            assert False, "n must be an int or a tuple"

        if delta_t is None:
            assert self.delta_t is not None, "delta_t must be set"
            delta_t = self.delta_t
            print("Using previously set delta_t", str(delta_t))
        else:
            assert delta_t > 0, "delta_t must be > 0"

        if total_time is None:
            assert self.total_time is not None, "total_time must be set"
            total_time = self.total_time
            print("Using previously set total_time:", str(total_time))
        else:
            assert total_time > 0, "delta_t must be > 0"

        # if rng is None:
        #     rng = np.random.default_rng()

        # sample data
        n_datapoints = len(np.arange(0, total_time, delta_t))
        X = rng.normal(0, self.sigma, size=np.append(n, n_datapoints))

        # mu as an array
        if isinstance(mu, int) or isinstance(mu, float):
            mu = np.full(n, mu)
        else:
            mu = np.array(mu)
            assert mu.shape == n or \
                   "you have to provide mu for each process"

        X = X + np.reshape(mu, np.append(n, 1))

        return X
