from numpy.linalg import pinv
from scipy.special import expit
from scipy.optimize import fmin_bfgs
from numpy import np


class BinaryLogisticRegression:
    def __init__(self, eta, penalty=None, iterations=20, C=0.001, l1_ratio=0.5):
        self.eta = eta
        self.penalty = penalty
        self.iters = iterations
        self.C = C
        self.l1_ratio = l1_ratio
        # internally we will store the weights as self.w_ to keep with sklearn conventions

    def __str__(self):
        if (hasattr(self, 'w_')):
            # is we have trained the object
            return 'Binary Logistic Regression Object with coefficients:\n' + str(self.w_)
        else:
            return 'Untrained Binary Logistic Regression Object'

    # convenience, private:
    @staticmethod
    def _add_bias(X):
        return np.hstack((np.ones((X.shape[0], 1)), X))  # add bias term

    @staticmethod
    def _sigmoid(theta):
        # increase stability, redefine sigmoid operation
        return expit(theta)  # 1/(1+np.exp(-theta))

    def _custom_regularization(self, gradient):
        # custom regularization
        if self.penalty is not None:
            if self.penalty == 'l1':
                gradient[1:] += -self.C*np.sign(self.w_[1:])
            elif self.penalty == 'l2':
                gradient[1:] += -2 * self.w_[1:] * self.C
            elif self.penalty == 'elasticnet':
                gradient[1:] += -self.C * \
                    (self.l1_ratio *
                     np.sign(self.w_[1:]) + (1-self.l1_ratio)*2*self.w_[1:])
            else:
                raise ValueError(
                    "choose either 'l1','l2','elasticnet' or leave None for No regularization")
        return gradient

    # vectorized gradient calculation with regularization using L2 Norm
    def _get_gradient(self, X, y):
        # get y difference
        ydiff = y-self.predict_proba(X, add_bias=False).ravel()
        # make ydiff a column vector and multiply through
        gradient = np.mean(X * ydiff[:, np.newaxis], axis=0)

        gradient = gradient.reshape(self.w_.shape)
        return self._custom_regularization(gradient)

    # public:
    def predict_proba(self, X, add_bias=True):
        # add bias term if requested
        Xb = self._add_bias(X) if add_bias else X
        return self._sigmoid(Xb @ self.w_)  # return the probability y=1

    def predict(self, X):
        return (self.predict_proba(X) > 0.5)  # return the actual prediction

    def fit(self, X, y):
        Xb = self._add_bias(X)  # add bias term
        num_samples, num_features = Xb.shape

        self.w_ = np.zeros((num_features, 1))  # init weight vector to zeros

        # for as many as the max iterations
        for _ in range(self.iters):
            gradient = self._get_gradient(Xb, y)
            self.w_ += gradient*self.eta  # multiply by learning rate
            # add bacause maximizing


class SteepestAscentLogisticRegression(BinaryLogisticRegression):
    def fit(self, X, y):
        Xb = self._add_bias(X)  # add bias term
        num_samples, num_features = Xb.shape

        # initialize weight vector to zeros
        self.w_ = np.zeros((num_features, 1))

        for _ in range(self.iters):
            gradient = self._get_gradient(Xb, y)
            # update weights using gradient ascent (positive direction)
            self.w_ += gradient * self.eta


class StochasticLogisticRegression(BinaryLogisticRegression):
    # stochastic gradient calculation
    def _get_gradient(self, X, y):

        # grab a subset of samples in a mini-batch
        # and calculate the gradient according to the small batch only
        mini_batch_size = 16
        idxs = np.random.choice(len(y), mini_batch_size)

        # get y difference (now scalar)
        ydiff = y[idxs]-self.predict_proba(X[idxs], add_bias=False).ravel()
        # make ydiff a column vector and multiply through
        gradient = np.mean(X[idxs] * ydiff[:, np.newaxis], axis=0)

        gradient = gradient.reshape(self.w_.shape)

        return self._custom_regularization(gradient)


class HessianBinaryLogisticRegression(BinaryLogisticRegression):
    # just overwrite gradient function
    def _get_gradient(self, X, y):
        # get sigmoid value for all classes
        g = self.predict_proba(X, add_bias=False).ravel()
        hessian = X.T @ np.diag(g*(1-g)) @ X - 2 * \
            self.C  # calculate the hessian

        ydiff = y-g  # get y difference
        # make ydiff a column vector and multiply through
        gradient = np.sum(X * ydiff[:, np.newaxis], axis=0)
        gradient = gradient.reshape(self.w_.shape)

        return pinv(hessian) @ self._custom_regularization(gradient)


class BFGSBinaryLogisticRegression(BinaryLogisticRegression):

    @staticmethod
    def objective_function(w, X, y, C, penalty, l1_ratio):
        g = expit(X @ w)
        # invert this because scipy minimizes, but we derived all formulas for maximzing
        return -np.sum(np.log(g[y == 1]))-np.sum(np.log(1-g[y == 0])) + C*sum(w**2)
        # -np.sum(y*np.log(g)+(1-y)*np.log(1-g))

    @staticmethod
    def objective_gradient(w, X, y, C, penalty, l1_ratio):
        g = expit(X @ w)
        ydiff = y-g  # get y difference
        gradient = np.mean(X * ydiff[:, np.newaxis], axis=0)
        gradient = gradient.reshape(w.shape)
        # applying custom regularization
        if penalty is not None:
            if penalty == 'l1':
                gradient[1:] += -C * np.sign(w[1:])
            elif penalty == 'l2':
                gradient[1:] += -2 * C * w[1:]
            elif penalty == 'elasticnet':
                gradient[1:] += -C * \
                    (l1_ratio * np.sign(w[1:]) + (1 - l1_ratio) * 2 * w[1:])
            else:
                raise ValueError(
                    "Choose either 'l1', 'l2', 'elasticnet' or leave None for no regularization")
        return -gradient

    # just overwrite fit function
    def fit(self, X, y):
        Xb = self._add_bias(X)  # add bias term
        num_samples, num_features = Xb.shape

        self.w_ = fmin_bfgs(self.objective_function,  # what to optimize
                            np.zeros((num_features, 1)),  # starting point
                            fprime=self.objective_gradient,  # gradient function
                            # extra args for gradient and objective function
                            args=(Xb, y, self.C, self.penalty, self.l1_ratio),
                            gtol=1e-03,  # stopping criteria for gradient, |v_k|
                            maxiter=self.iters,  # stopping criteria iterations
                            disp=False)

        self.w_ = self.w_.reshape((num_features, 1))


class MultiClassLogisticRegression:
    solvers = {
        None: BFGSBinaryLogisticRegression,
        "hessian": HessianBinaryLogisticRegression,
        "stochastic": StochasticLogisticRegression,
        "steepest": SteepestAscentLogisticRegression,
    }

    def __init__(self, eta, iterations=20,
                 C=0.0001,
                 solver=None):
        self.eta = eta
        self.iters = iterations
        self.C = C
        # custom optimization technique with instantiation
        if solver not in self.solvers:
            raise ValueError(
                "choose either 'hessian','stochastic','steepest' or leave None for BFGSBinaryLogisticRegression")
        self.solver = self.solvers[solver]

        self.classifiers_ = []
        # internally we will store the weights as self.w_ to keep with sklearn conventions

    def __str__(self):
        if (hasattr(self, 'w_')):
            # is we have trained the object
            return 'MultiClass Logistic Regression Object with coefficients:\n' + str(self.w_)
        else:
            return 'Untrained MultiClass Logistic Regression Object'

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.unique_ = np.sort(np.unique(y))  # get each unique class value
        num_unique_classes = len(self.unique_)
        self.classifiers_ = []
        for i, yval in enumerate(self.unique_):  # for each unique value
            y_binary = np.array(y == yval).astype(
                int)  # create a binary problem
            # train the binary classifier for this class

            hblr = self.solver(eta=self.eta, iterations=self.iters, C=self.C)
            hblr.fit(X, y_binary)

            # add the trained classifier to the list
            self.classifiers_.append(hblr)

        # save all the weights into one matrix, separate column for each class
        self.w_ = np.hstack([x.w_ for x in self.classifiers_]).T

    def predict_proba(self, X):
        probs = []
        for hblr in self.classifiers_:
            # get probability for each classifier
            probs.append(hblr.predict_proba(X).reshape((len(X), 1)))

        return np.hstack(probs)  # make into single matrix

    def predict(self, X):
        # take argmax along row
        return self.unique_[np.argmax(self.predict_proba(X), axis=1)]
