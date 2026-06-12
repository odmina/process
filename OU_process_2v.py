import numpy as np
from scipy.linalg import expm


class OU_process_2v(object):
    """
    This class defines two variable, stationary Ornstein–Uhlenbeck process
    and allows data simulation.

    Follows symbols from Gardiner (2009).

    The process can be initialized with A and sigma or A and BB_T
    (for a given A and sigma, BB_T is known and vice versa).
    Initialization with all three parameters will fail.

    Args:

        `A`: centralizing matrix `A` links the variables in
        the process. It is asserted to be a real matrix with
        positive eigenvalues. Voelkle et al. also call it A, but in the exponent in the
        process equation it comes without a minus (and for that
        parametrization eigenvalues have to be negative).
        It is called B by Oravecz et al.

        `sigma`: covariance matrix of the stationary OU process.
        Called sigma by Gardiner (2009). It is a dot product of
        process state at time t, X(t), and its transpose
        (for a process with 0 mean or centered process). It has to be
        a symmetric, positive-semidefinite matrix.

        `BB_T`: error covariance matrix of the process, used in the
        stochastic term of the process equation, also called diffusion
        matrix. Cholesky decomposition of this matrix (B)
        appears in the stochastic term of the equation of X(t). Thus,
        the matrix has to be Hermitian, positive-definite matrix
        (since for the simulation only real valued matrices are used,
        BB_T has to be symmetric and positive-definite and this is the
        condition that is asserted).
        For the simulation, the conditional distribution of X(t)
        given X(t-d) is reparametrized so that it depends on B and sigma.
        Thus, the following condition has to be satisfied:
        BB_T = B*sigma + sigma*B.T is a symmetric, positive-definite matrix.

    During data simulation parameter `mu` may be provided.
        It is the mean of the process, it is 0 by default.

    The simulation uses discrete timesteps of equal length `d`
        and at each timestep samples X(t) from a conditional
        distribution of X(t) given X(t-d). The distribution is
        normal with parameters calculated using eq (9) from
        Oravecz et al. 2011.

    Simulation runs for `total_time` units
        with 1/'d' samples per unit.

    Initial value of the process X(0) is sampled from bivariate normal
        distribution with mean `mu` and covariance matrix `sigma`.
    """

    def __init__(self, A, sigma=None, BB_T=None):
        assert sigma is not None or BB_T is not None, \
            """You should provide either stationary covariance matrix (sigma)
            or diffusion matrix (BB_T)"""

        # if both sigma and BB_T provided
        assert sigma is None or BB_T is None, \
            """Only sigma or BB_T should be provided.
            
            For a known sigma, BB_T can be calculated.
            For a known BB_T, sigma can be calculated.
            The class does not check whether you provided
            a good pair."""

        # set and check A
        # A is assumed to be 2x2, real and have positive eigenvalues
        self.A = np.array(A)
        assert np.all(np.isreal(self.A)), "A is not a real matrix"
        assert self.A.shape == (2, 2), "A is not 2x2 matrix"
        assert np.all(np.linalg.eigvals(self.A) >
                      0), "A has zero or negative eigenvalues"

        # if sigma is provided set and check it and set BB_T
        # sigma is assumed to be 2x2, real, symmetric and positive semi-definite
        # BB_T = B*sigma + sigma*B.T should be positive-definite
        # (symmetric with positive eigenvalues)
        if sigma is not None and BB_T is None:
            self.sigma = np.array(sigma)
            assert np.all(np.isreal(self.sigma)), "sigma is not a real matrix"
            assert self.sigma.shape == (2, 2), "sigma is not 2x2 matrix"
            assert np.all(self.sigma == self.sigma.T), "sigma not symmetric"
            sigma_eigvals = np.linalg.eigvals(self.sigma)
            assert np.all(sigma_eigvals > 0) and \
                   np.all(np.isreal(sigma_eigvals)), \
                   "sigma has negative or complex eigenvalues"
            # set BB_T
            self.BB_T = np.matmul(self.A, self.sigma) + \
                        np.matmul(self.sigma, self.A.T)
            assert np.all(self.BB_T == self.BB_T.T), "BB_T not symmetric"
            assert np.all(np.linalg.eigvals(self.BB_T) > 0), "BB_T has negative eigenvalues"

        # if BB_T is provided, check it and set sigma
        # BB_T is assumed to be 2x2, symmetric and positive definite
        # BB_T could be Hermitian, but for now simulation uses only real values
        if BB_T is not None and sigma is None:
            self.BB_T = np.array(BB_T)
            assert self.BB_T.shape == (2, 2), "BB_T is not 2x2 matrix"
            assert np.all(np.isreal(self.BB_T)), "BB_T is not a real matrix"
            assert np.all(self.BB_T == self.BB_T.T), "BB_T not symmetric"
            assert np.all(np.linalg.eigvals(self.BB_T) > 0), "BB_T has negative eigenvalues"
            # set sigma
            # equation for sigma is eq. 4.4.53 from
            # https://archive.org/details/handbookofstocha0000gard/page/110/mode/2up
            det_A = np.linalg.det(self.A)
            tr_A = np.trace(self.A)
            A_no_trace = self.A - tr_A * np.identity(2)
            self.sigma = (det_A * self.BB_T + np.matmul(np.matmul(A_no_trace, self.BB_T),
                                                        A_no_trace.T)) / \
                         (2 * det_A * tr_A)
            assert np.all(np.isreal(self.sigma)), "sigma is not a real matrix"
            assert np.all(self.sigma == self.sigma.T), "sigma not symmetric"
            sigma_eigvals = np.linalg.eigvals(self.sigma)
            assert np.all(sigma_eigvals > 0) and \
                   np.all(np.isreal(sigma_eigvals)), \
                   "sigma has negative or complex eigenvalues"

    def __str__(self):
        return """Two variable Ornstein–Uhlenbeck process;
        `use get_A()`, `get_sigma()` and 'get_BB_T()' to access parameters."""

    # ------ GETTERS ------

    def get_A(self):
        return self.A

    def get_sigma(self):
        return self.sigma

    def get_BB_T(self):
        return self.BB_T

    # ------ DISCRETE TIME SIMULATION METHODS ------

    def get_sim_discreteA(self, d=0.1):
        """
        This method calculates the discrete time version of A.

        It uses timestep `d`, with default 0.1.
        """
        return expm(-self.A * d)

    def get_sim_condPDF_covariance(self, d=0.1):
        """
        Calculates the variance of the distribution of X(t) given X(t-d).

        Uses timestep `d`, with default 0.1.

        The conditional distribution of X(t) given X(t-d) depends
        on A and sigma. See for example oravecz2011hierarchical eq (5).
        """

        cov = self.sigma - np.matmul(
            np.matmul(expm(-self.A * d), self.sigma),
            expm(-self.A.T * d)
        )
        return cov

    def get_stationary_time_cov(self, d=0.1):
        """Compute time covariance (cross-covariance) matrix for a given time step in a stationary state.

        Based on eq. 4.5.71a from Gardiner, C. (2009). Stochastic Methods.

        Args:
            d: timestep. Default 0.1.

        Returns: time covariance matrix for the given process and t.
        """

        time_covariance = np.matmul(expm(-self.A * d), self.sigma)

        return time_covariance

    def get_stationary_time_correlation(self, d=0.1):
        """Compute time correlation (cross-correlation) matrix for a given time step in a stationary state.

        Based on eq. 4.5.71a from Gardiner, C. (2009). Stochastic Methods.

        Args:
            d: timestep. Default 0.1.

        Returns: time correlation matrix for the given process and t.
        """
        time_covariance = self.get_stationary_time_cov(d=d)
        stationary_sds = np.sqrt(np.diagonal(self.sigma))
        time_correlation = time_covariance / np.outer(stationary_sds, stationary_sds.T)

        return time_correlation

    def sim_data(self, d=0.1, total_time=10, mu=(0, 0)):
        """
        This function simulates a two variable Ornstein–Uhlenbeck process.

        Let vector X(t) be a vector of two variables at time t.
        Covariance matrix of the OU process is `sigma` (and is stationary).
        Mean of the process `mu` is by default 0, but can be changed.
        The simulation uses discrete timesteps of equal length d
        and at each timestep samples X(t) from a conditional
        distribution of X(t) given X(t-d). The distribution is
        normal with parameters calculated using eq (9) from
        Oravecz et al. 2011.
        Variables in X are linked by centralizing (drift) matrix `A`

        Simulation runs for `total_time` units
            with 1/'d' samples per unit.

        Initial value of the process is sampled from bivariate normal
            distribution with mean mu and covariance matrix sigma.
        """

        mu = np.array(mu)
        assert mu.shape == (2,), "mu is not (2,) array"

        rng = np.random.default_rng()

        # initialize data array
        n_datapoints = int(total_time / d)
        X = np.empty((2, n_datapoints))

        # compute parameters for the given d
        condPDF_cov = self.get_sim_condPDF_covariance(d=d)
        cond_A = self.get_sim_discreteA(d=d)

        # set process start X(t=0)
        X[:, 0] = rng.multivariate_normal(mu, self.sigma)

        # simulate data
        for i in np.arange(n_datapoints - 1):
            this_mu = mu + np.matmul(cond_A, (X[:, i] - mu))
            X[:, i + 1] = rng.multivariate_normal(this_mu, condPDF_cov)

        return X
