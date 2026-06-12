import numpy as np
import warnings


class OU_process_1v(object):
    """
    This class defines one variable, stationary Ornstein–Uhlenbeck process
    and allows data simulation.

    Follows symbols from Gardiner (2009).

    The process can be initialized with any two out of the arguments.

    Args:
        `k`: linear drift term, has to be > 0.
        `D`: diffusion term, has to be > 0.
        `sigma`: stationary VARIANCE of the process, has to be > 0.

    During data simulation parameter `mu` may be provided.
        It is the stationary mean of the process, it is 0 by default.

    The simulation uses discrete timesteps of equal length `delta_t`
        and at each timestep samples X(t) from a conditional
        distribution of X(t) given X(t-d).

    Simulation runs for `total_time` units
        with 1/'d' samples per unit.

    Initial value of the process X(0) is sampled from a normal
        distribution with mean `mu` and covariance `sigma`.
    """

    def __init__(self, **kwargs):
        assert len(kwargs) == 2, """You should provide exactly two arguments."""

        self.k = kwargs.get('k', None)
        self.D = kwargs.get('D', None)
        self.sigma = kwargs.get('sigma', None)

        if self.k is not None and self.D is not None and self.sigma is None:
            assert self.k > 0, 'k and D must be > 0.'
            assert self.D > 0, 'D must be > 0.'
            self.sigma = self.D/(2*self.k)
        elif self.k is not None and self.D is None and self.sigma is not None:
            assert self.k > 0, 'k and sigma must be > 0.'
            assert self.sigma > 0, 'sigma must be > 0.'
            self.D = self.sigma*2*self.k
        elif self.k is None and self.D is not None and self.sigma is not None:
            assert self.sigma > 0, 'sigma and D must be > 0.'
            assert self.D > 0, 'D must be > 0.'
            self.k = self.D/(2*self.sigma)
        else:
            raise ValueError('You should provide exactly two arguments out of sigma, D, k.')

        self.delta_t = None
        self.total_time = None

    def __str__(self):
        return """One variable Ornstein–Uhlenbeck process;
        `use get_k()`, `get_sigma()` and 'get_D()' to access parameters."""

    # ------ GETTERS ------

    def get_k(self):
        return self.k

    def get_sigma(self):
        return self.sigma

    def get_D(self):
        return self.D

    # ------ DISCRETE TIME SIMULATION METHODS ------
    def set_sim_parameters(self, delta_t=None, total_time=None):
        """Set simulation delta_t (timestep) and total_time."""
        if delta_t is not None:
            assert delta_t > 0, "delta_t must be > 0"
            if self.delta_t is None:
                self.delta_t = delta_t
            else:
                print("`delta_t` was already set.")
                cont = input("Type y if you want to overwrite it: ")
                if cont.strip() == "y":
                    self.delta_t = delta_t

        if total_time is not None:
            assert total_time > 0, "total_time must be > 0"
            if self.total_time is None:
                self.total_time = total_time
            else:
                print("`total_time` was already set.")
                cont = input("Type y if you want to overwrite it: ")
                if cont.strip() == "y":
                    self.total_time = total_time

        if delta_t is None and total_time is None:
            warnings.warn("Nothing to set!")

    def get_sim_discrete_k(self, delta_t=None):
        """
        This method calculates the discrete time k.

        Arguments:
            `delta_t`: simulation timestep, if not provided
            one from the process parameters is used.
        """
        if delta_t is None:
            assert self.delta_t is not None, "delta_t was not provided, cannot compute discrete autocorrelation"
            delta_t = self.delta_t
            print("using previously set simulation parameter `delta_t` =", self.delta_t)
        assert delta_t > 0, "delta_t must be > 0"
        return np.exp(-self.k * delta_t)

    def get_sim_condPDF_variance(self, delta_t=None):
        """
        Calculates the variance of the distribution of X(t) given X(t-delta_t).

        Arguments:
            `delta_t`: simulation timestep, if not provided
            one from the process parameters is used.

        The conditional distribution of X(t) given X(t-delta_t) depends
        on k and sigma. For the proof of the formula see for example
        Gillespie, Daniel T. “Exact Numerical Simulation of
        the Ornstein-Uhlenbeck Process and Its Integral.” Physical Review E 54, no. 2 (August 1, 1996): 2084–91.
        https://doi.org/10.1103/PhysRevE.54.2084.
        """
        if delta_t is None:
            assert self.delta_t is not None, "delta_t was not set, cannot compute conditional pdf variance"
            delta_t = self.delta_t
            print("using previously set simulation parameter `delta_t` =", self.delta_t)
        else:
            assert delta_t > 0, "delta_t must be > 0"
        var = self.sigma*(1 - np.exp(-2 * self.k * delta_t))
        return var

    def get_stationary_autocov(self,  delta_t=None):
        """Compute autocovariance for a given time step in a stationary state.
        Based on eq. 3.8.83 from Gardiner, C. (2009). Stochastic Methods.
        Simulation parameters `total_time` and `delta_t` do not have to be set;
        `delta_t` can be provided in method call and will NOT affect simulation
        parameters.

        Arguments:
            `delta_t`: simulation timestep, if not provided
            one from the process parameters is used.

        Returns: autocovariance for the given process and time.
        """
        if delta_t is None:
            assert self.delta_t is not None, "delta_t was not set, cannot compute autocovariance"
            delta_t = self.delta_t
            print("using previously set simulation parameter `delta_t` =", self.delta_t)
        else:
            assert delta_t > 0, "delta_t must be > 0"
        return self.sigma * np.exp(-self.k * delta_t)

    def get_stationary_autocorrelation(self, delta_t=None):
        """Compute autocorrelation for a given time step in a stationary state.
        This is equivalent to getting simulation discrete k.

        Arguments:
            `delta_t`: simulation timestep, if not provided
            one from the process parameters is used.

        Returns: time correlation for the given process and `delta_t`.
        """
        if delta_t is None:
            assert self.delta_t is not None, "delta_t was not set, cannot compute autocorrelation"
            delta_t = self.delta_t
            print("using previously set simulation parameter `delta_t` =", self.delta_t)
        else:
            assert delta_t > 0, "delta_t must be > 0"
        return np.exp(-self.k * delta_t)

    def sim_data(self, n=1, mu=0, delta_t=None, total_time=None, rng=None, initial_value=None):
        """
        This function simulates a one variable Ornstein–Uhlenbeck process.
        Args:
            n: Number of processes. If processes need to be grouped, n can be a tuple. For example:
            for n=(2,3), two groups of three processes are going to be generated.
            mu: stationary mean. Default 0.
                Can be a list or an array with n elements (if n is int)
                or of shape n (if n is a tuple).
            delta_t: timestep.
            total_time: total simulation time.
            rng: the function can use global random number generator. Default None.
            initial_value: initial value of the process. If None (default),
            it is sampled from a normal distribution with mean mu and stationary
            variance of the process.

        Returns: numpy array containing simulated data
            with length equal to len(np.arange(0, total_time, d)).
            Returned array will have len(n) + 1 dimensions.

        Let X(t) be a state of the process at time t.
        Unconditioned, each X(t) is a gaussian with mean `mu` and variance `sigma`.

        The simulation uses discrete timesteps of equal length d
        and at each timestep samples X(t) from a conditional
        distribution of X(t) given X(t-d). Sampling distribution is
        normal with parameters calculated using eq. 3.5a from Gilesspie (1996).

        Simulation runs for `total_time` units
            with 1/'d' samples per unit.

        Important: initial value of the process, unless provided, is sampled from normal
            distribution with mean mu and variance sigma.
        """

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
        else:
            assert delta_t > 0, "delta_t must be > 0"

        if total_time is None:
            assert self.total_time is not None, "total_time must be set"
            total_time = self.total_time
        else:
            assert total_time > 0, "delta_t must be > 0"

        # if rng is None:
        #     rng = np.random.default_rng()

        # compute parameters for the given d
        condPDF_var = self.get_sim_condPDF_variance(delta_t)
        condPDF_sd = np.sqrt(condPDF_var)
        cond_k = self.get_sim_discrete_k(delta_t)

        # initialize data array
        n_datapoints = len(np.arange(0, total_time, delta_t))
        X_size = np.append(n, n_datapoints)
        X = np.zeros(X_size)

        # sample random parts all at once
        X_random = rng.normal(0, condPDF_sd, size=X_size)

        # mu as an array
        if isinstance(mu, int) or isinstance(mu, float):
            mu = np.full(n, mu)
        else:
            mu = np.array(mu)
            assert mu.shape == n, "you have to provide mu for each process"

        # initial value as an array
        if initial_value is None:
            initial_value = rng.normal(mu, np.sqrt(self.sigma), n)
        elif isinstance(initial_value, int) or isinstance(initial_value, float):
            initial_value = np.full(n, initial_value)
        else:
            initial_value = np.array(initial_value)
            assert initial_value.shape == n, "you have to provide initial value for each process"

        # set process start X(t=0)
        # this is a process with mu = 0
        # mu provided in function call will be added later
        X_first_datapoints = np.zeros(np.append(n, 1), dtype=int)
        X_last_axis = X.ndim - 1
        process_start = np.reshape(initial_value - mu, np.append(n, 1))
        np.put_along_axis(X, X_first_datapoints, process_start, axis=X_last_axis)

        # get processes with 0 mean
        for i in np.arange(n_datapoints - 1):
            # get current state of each process
            positions = X_first_datapoints + i
            current_states = np.take_along_axis(X, positions, axis=X_last_axis)
            random_values = np.take_along_axis(X_random, positions+1, axis=X_last_axis)
            next_states = current_states*cond_k + random_values
            np.put_along_axis(X, positions+1, next_states, axis=X_last_axis)

        # add mu
        X = X + np.reshape(mu, np.append(n, 1))

        return X
