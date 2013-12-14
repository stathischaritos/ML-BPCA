import copy
import pylab as pl
import numpy as np

class BayesianPCA(object):

    def __init__(self, d, N, a_alpha=10e-3, b_alpha=10e-3, a_tau=10e-3, b_tau=10e-3, beta=10e-3):
        """
        Initializes parameters randomly such that they can be used
        by updating later
        """

        self.d = d # number of dimensions
        self.N = N # number of data points

        # Hyperparameters
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.beta = beta

        # Variational parameters
        self.means_z = np.random.randn(d, N) # called x in bishop99
        self.sigma_z = np.random.randn(d, d)
        self.mean_mu = np.random.randn(d, 1)
        self.sigma_mu = np.random.randn(d, d)
        self.means_w = np.random.randn(d, d)
        self.sigma_w = np.random.randn(d, d)
        self.a_alpha_tilde = np.abs(np.random.randn(1))
        self.b_alpha_tilde = np.abs(np.random.randn(d, 1))
        self.a_tau_tilde = np.abs(np.random.randn(1))
        self.b_tau_tilde = np.abs(np.random.randn(1))

    def __update_z(self, X):
        """
        updates Z
        Z is the prod over datapoints: N(z_n | m_z, Sigma_n)
        X is the data

        means_z^(n) = <tau> * Sigma_z * <W.T> * (X_n - <mu>)
        Sigma_z = (I + <tau> <W.T*W>)^-1
        """
        # variables necessary in calculations
        tau_exp = self.a_tau_tilde / self.b_tau_tilde
        mu_exp = self.mean_mu
        W_exp = self.means_w
        WT_W_exp = np.dot(W_exp.T, W_exp) # TODO fix

        # update sigma, equal for all n's
        self.sigma_z = np.linalg.inv(np.eye(self.d) + tau_exp * WT_W_exp)

        # update mean
        self.means_z = np.dot(np.dot( tau_exp * self.sigma_z, W_exp.T), (X - mu_exp))
        
    def __update_mu(self, X):
        """
        update mean_mu and sigma_mu
        X is data

        sigma_mu = (beta + N * <tau>)^-1 * I
        mean_mu = <tau> * sigma_mu * sum( X_n - <W> * <z_n> )
        """
        # necessary for calculations of both
        tau_exp = self.a_tau_tilde / self.b_tau_tilde

        # update sigma_mu first as it is used in updating mean_mu
        self.sigma_mu = (1.0 / (self.beta + self.N * tau_exp)) * np.eye(self.d)

        # update mean_mu
        sum_x_w_z = np.sum(X - np.dot(self.means_w, self.means_z), axis=1, keepdims=True)
        self.mean_mu = np.dot(tau_exp * self.sigma_mu, sum_x_w_z)

    def __update_w(self, X):
        """
        update mean_w and sigma_w

        sigma_w ( diag<alpha> + <tau> * sum( <z_n * z_n.T  )  )
        mean_w_k = <tau> * sigma_w * sum( <z_n> * ( X_nk - <mu_k>  )  )
        """
        # necessary for calculations
        tau_exp = self.a_tau_tilde / self.b_tau_tilde

        # update sigma_w first as it is used in updating means_w
        exp_alpha = self.a_alpha_tilde / self.b_alpha_tilde
        diag_exp_alpha = np.diag(exp_alpha.T[0])
        tau_sum_zn = tau_exp * (self.N * self.sigma_z + np.dot(self.means_z, self.means_z.T))
        self.sigma_w = np.linalg.inv(diag_exp_alpha + tau_sum_zn)

        # update means_w
        # einsum calculates for all k the summation over <z_n> * ( X_nk - mu_k)
        einsum_result = np.einsum('kj,ij->ik', X - self.mean_mu, self.means_z)
        self.means_w = np.dot(tau_exp * self.sigma_w, einsum_result)

    def __update_alpha(self):
        """
        update b_alpha_tilde, a_alpha_tilde does not change

        b_alpha_tilde[i] = b_alpha + < || w[i] ||^2 > / 2
        """
        # update each element in b_alpha_tilde
        # TODO: not sure, correction: pretty sure it is wrong
        w_norm = np.power(np.linalg.norm(self.means_w, axis=0), 2)
        w_norm = np.reshape(w_norm, (self.d, 1)) # reshape from (d,) to (d,1)

        self.b_alpha_tilde = self.b_alpha + w_norm / 2

    def __update_tau(self, X):
        """
        Update b_tau_tilde, as a_tau_tilde is independent of other update rules

        b_tau_tilde = b_tau + 1/2 sum ( Z  )
        where Z =
        || X_n ||^2 + <|| mu ||^2> + Tr(<W.T * W> <z_n * z_n.T>) +
            2*<mu.T> * <W> * <z_n> - 2 * X_n.T * <W> * <z_n> - 2 * X_n.T * <mu>
        """
        x_norm_sq = np.power(np.linalg.norm(X, axis=0), 2)
        # <|mu|^2> = <mu.T mu> = Tr(Sigma_mu) + mean_mu.T mean_mu
        exp_mu_norm_sq = np.trace(self.sigma_mu) + np.dot(self.mean_mu.T, self.mean_mu)
        exp_mu_norm_sq = exp_mu_norm_sq[0] # reshape from (1,1) to (1,)

        # TODO what is <W.T W>
        exp_w = self.means_w
        exp_wt_w = np.dot(exp_w.T, exp_w) # TODO fix
        exp_z_zt = self.N * self.sigma_z + np.dot(self.means_z, self.means_z.T)

        trace_w_z = np.trace(np.dot(exp_wt_w, exp_z_zt))

        mu_w_z = np.dot(np.dot(self.mean_mu.T, self.means_w), self.means_z)

        x_w_z = np.dot(X.T, self.means_w).T * self.means_z

        x_mu = np.dot(X.T, self.mean_mu)

        big_sum = np.sum(x_norm_sq) + self.N * exp_mu_norm_sq + trace_w_z + \
                  2*np.sum(mu_w_z) - 2*np.sum(x_w_z) - 2*np.sum(x_mu)

        self.b_tau_tilde = self.b_tau + 0.5*big_sum

    def L(self, X):
        L = 0.0
        return L

    def fit(self, X):

        # set constant parameters
        self.a_alpha_tilde = self.a_alpha + self.d / 2
        self.a_tau_tilde = self.a_tau + self.N*self.d / 2

        it = 1000
        converged = False
        while not converged and it > 0:
            print("Updating iteration " + str(it))
            # run each update
            self.__update_z(X)
            self.__update_mu(X)
            self.__update_w(X)
            self.__update_alpha()
            self.__update_tau(X)

            it -= 1
            # TODO: decide on converged


    def test_shapes_after_updates(self, X):
        """
        This test simply tests whether the shapes of the
        important variables change. If they do, a warning is produced
        """
        shapes = self.get_shapes()
        # run tests while updating
        self.__update_z(X)
        self.test_shapes(shapes)

        self.__update_mu(X)
        self.test_shapes(shapes)

        self.__update_w(X)
        self.test_shapes(shapes)

        self.__update_alpha()
        self.test_shapes(shapes)

        self.__update_tau(X)
        self.test_shapes(shapes)


    def get_shapes(self):
        """
        Returns the shape of all variables in vpca
        """
        shapes = [(self.means_z.shape, 'means_z'),
                  (self.sigma_z.shape, 'sigma_z'),
                  (self.mean_mu.shape, 'mean_mu'),
                  (self.sigma_mu.shape, 'sigma_mu'),
                  (self.means_w.shape, 'means_w'),
                  (self.sigma_w.shape, 'sigma_w'),
                  (self.a_alpha_tilde.shape, 'a_alpha_tilde'),
                  (self.b_alpha_tilde.shape, 'b_alpha_tilde'),
                  (self.a_tau_tilde.shape, 'a_tau_tilde'),
                  (self.b_tau_tilde.shape, 'b_tau_tilde')]

        return shapes

    def test_shapes(self, shapes):
        """
        Tests whether the shapes in oldvpca are equal to those in
        newvpca
        """
        for orig_shape, name in shapes:
            var_shape = eval("self.%s.shape" % name)
            assert orig_shape == var_shape, "%s: %s should have been %s" % (name, var_shape, orig_shape)
            
        print 'correct shapes'


def generate_data(N):
    mean = np.zeros(10)
    covariance = np.diag([5,4,3,2,1,1,1,1,1,1])
    return np.random.multivariate_normal(mean, covariance, N).T

def run():
    d = 10
    N = 2
    X = generate_data(N)
    vpca = BayesianPCA(d, N)
    vpca.fit(X)

def run_shapes():
    d = 10
    N = 2
    X = generate_data(N)
    vpca = BayesianPCA(d, N)
    vpca.test_shapes_after_updates(X)
    
def test_default_shapes():
    d = 10
    N = 2
    X = generate_data(N)
    vpca = BayesianPCA(d, N)
    shapes = vpca.get_shapes()
    vpca.test_shapes(shapes)

def show_hinton():
    d = 10
    N = 2
    X = generate_data(N)
    vpca = BayesianPCA(d, N)
    vpca.test_shapes_after_updates(X)
    hinton(vpca.means_w)

def _blob(x,y,area,colour):
    """
    Source: http://wiki.scipy.org/Cookbook/Matplotlib/HintonDiagrams
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    pl.fill(xcorners, ycorners, colour, edgecolor=colour)

def hinton(W, maxWeight=None):
    """
    Source: http://wiki.scipy.org/Cookbook/Matplotlib/HintonDiagrams
    Draws a Hinton diagram for visualizing a weight matrix.
    Temporarily disables matplotlib interactive mode if it is on,
    otherwise this takes forever.
    """
    reenable = False
    if pl.isinteractive():
        pl.ioff()
    pl.clf()
    height, width = W.shape
    if not maxWeight:
        maxWeight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))

    pl.fill(np.array([0,width,width,0]),np.array([0,0,height,height]),'gray')
    pl.axis('off')
    pl.axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y,x]
            if w > 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,w/maxWeight),'white')
            elif w < 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,-w/maxWeight),'black')
    if reenable:
        pl.ion()
    pl.show()

if __name__ == '__main__':
    #run()
    #run_shapes()
    show_hinton()
    #test_default_shapes()
