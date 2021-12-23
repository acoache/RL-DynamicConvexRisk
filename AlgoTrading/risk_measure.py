"""
Risk measures
Implementation of the mean, CVaR, penalized-CVaR, semi-dev and mean-CVaR<

"""
# numpy
import numpy as np
# pytorch
import torch as T
# misc
#import cvxpy as cp
from scipy import optimize
import pdb # use with set_trace() for the debugger

class RiskMeasure():
    # constructor
    def __init__(self, Type, alpha=0.5, kappa=1, r=2):
        self.Type = Type

        # Conditional value-at-risk
        if(self.Type == 'CVaR'):
            assert (alpha > 0) and (alpha < 1), "alpha needs to be in (0,1)"
            self.alpha = alpha

        # Mean semi-deviation
        elif(self.Type == 'semi-dev'):
            assert (kappa >= 0) and (kappa <= 1), "kappa needs to be in [0,1]"
            assert isinstance(r,int) and (r >= 1), "r needs to be an integer greather than 1"
            self.kappa = kappa
            self.r = r

        # Expectation
        elif(self.Type == 'mean'):
            pass

        # Conditional value-at-risk penalized
        elif(self.Type == 'CVaR-penalized'):
            assert (alpha > 0) and (alpha < 1), "alpha needs to be in (0,1)"
            self.alpha = alpha
            self.kappa = kappa

        # Conditional value-at-risk with mean
        elif(self.Type == 'mean-CVaR'):
            assert (alpha > 0) and (alpha < 1), "alpha needs to be in (0,1)"
            self.alpha = alpha
            self.kappa = kappa

        else:
            assert False, "Type of the risk measure is unknown"


    # calculate the risk of a sequence of values
    def compute_risk(self, x):
        # Conditional value-at-risk
        if(self.Type == 'CVaR'):
            quant = T.quantile(x, 1-self.alpha, axis=1).unsqueeze(1).repeat(1,x.shape[1],1)
            cond = x >= quant
            RM = T.sum(x.masked_fill(~cond, 0.0), axis=1) / T.sum(cond, axis=1)

        # Mean semi-deviation
        elif(self.Type == 'semi-dev'):
            mean = T.mean(x, axis=1)
            semi_dev = T.mean( T.maximum(x - mean.unsqueeze(-1), T.zeros(1))**self.r, axis=1 )**(1/self.r)
            RM = mean + self.kappa * semi_dev

        # Expectation
        elif(self.Type == 'mean'):
            RM = T.mean(x, axis=1)

        # Conditional value-at-risk penalized
        elif(self.Type == 'CVaR-penalized'):
            # analytic expression -- root_scalar
            weights = T.zeros(x.shape, dtype=T.float, requires_grad=False)
            for idx1, row in enumerate(x):
                for idx2, col in enumerate(row.transpose(0,1)):
                    col_np = col.detach().numpy()
                    def f(value):
                        return np.mean( np.minimum(np.exp((col_np-value-self.kappa)/self.kappa) - 1/self.alpha, 0) ) - 1 + (1/self.alpha)
                    sol = optimize.root_scalar(f,
                                            bracket=[np.quantile(col_np,1-self.alpha)-5, np.quantile(col_np,1-self.alpha)+5])
                    lambda0 = sol.root
                    weights[idx1,:,idx2] = T.minimum(T.exp((col.detach()-lambda0-self.kappa)/self.kappa), T.tensor(1/self.alpha))
            
            RM = T.mean(weights*x, axis=1) - self.kappa*T.mean(weights*T.log(weights), axis=1)
        
        # Conditional value-at-risk with mean
        elif(self.Type == 'mean-CVaR'):
            quant = T.quantile(x, 1-self.alpha, axis=1).unsqueeze(-1).repeat(1,x.shape[1])
            cond = x >= quant
            RM = (1-self.kappa) * T.sum(x.masked_fill(~cond, 0.0), axis=1) / T.sum(cond, axis=1) \
                    + self.kappa * T.mean(x, axis=1)

        return RM

    # calculate the risk of a sequence of values (numpy version)
    def compute_risk_np(self, x):
        # Conditional value-at-risk
        if(self.Type == 'CVaR'):
            cond = x[x >= np.quantile(x,1-self.alpha)]
            RM = np.mean(cond) if len(cond)>0 else np.quantile(x,1-self.alpha)

        # Mean semi-deviation
        elif(self.Type == 'semi-dev'):
            semi_dev = np.mean( np.maximum(x - np.mean(x), 0)**self.r )**(1/self.r)
            RM = np.mean(x) + self.kappa * semi_dev

        # Expectation
        elif(self.Type == 'mean'):
            RM = np.mean(x)

        # Conditional value-at-risk penalized
        elif(self.Type == 'CVaR-penalized'):
            # # convex optimization -- CVXPY
            # N = x.shape[0]
            # weights = cp.Variable(N)
            # objective = cp.Maximize(cp.sum(cp.multiply(weights,x))/N + self.kappa*cp.sum(cp.entr(weights))/N)
            # constraints = [0 <= weights, \
            #                 weights <= 1/self.alpha,
            #                 cp.sum(weights) == N]
            # prob = cp.Problem(objective, constraints)
            # result = prob.solve()
            # RM = prob.value
            
            # analytic expression -- root_scalar
            def f(value):
                return np.mean( np.minimum(np.exp((x - value - self.kappa)/self.kappa) - 1/self.alpha, 0) ) - 1 + (1/self.alpha)
            
            sol = optimize.root_scalar(f, bracket=[np.quantile(x,1-self.alpha)-5, np.quantile(x,1-self.alpha)+5])
            lambda0 = sol.root
            weights = np.minimum(np.exp((x-lambda0-self.kappa)/self.kappa), 1/self.alpha)
            RM = np.mean(weights*x) - self.kappa*np.mean(weights*np.log(weights))

        # Conditional value-at-risk with mean
        elif(self.Type == 'mean-CVaR'):
            cond = x[x >= np.quantile(x,1-self.alpha)]
            cvar = np.mean(cond) if len(cond)>0 else np.quantile(x,1-self.alpha)
            RM = (1-self.kappa) * cvar + self.kappa * np.mean(x)
        
        return RM

    # calculate the gradient based on transitions and rewards
    def get_V_loss(self, V_tp1, logprob):
        # Conditional value-at-risk
        if(self.Type == 'CVaR'):
            quant = T.quantile(V_tp1, 1-self.alpha, axis=1).unsqueeze(1).repeat(1,V_tp1.shape[1],1)
            cond = V_tp1 > quant
            loss = T.sum( logprob.masked_fill(~cond, 0.0) * (V_tp1.masked_fill(~cond, 0.0) - quant), axis=1) \
                    / T.sum(cond, axis=1)

        # Mean semi-deviation
        elif(self.Type == 'semi-dev'):
            mean_V = T.mean(V_tp1, axis=1).unsqueeze(-1)
            semi_dev = T.mean(T.maximum(V_tp1 - mean_V, T.zeros(1))**self.r, axis=1)**(1/self.r)
            grad_mean_V = T.mean(V_tp1 * logprob, axis=1)
            loss = grad_mean_V + self.kappa/semi_dev * T.mean( T.maximum(V_tp1 - mean_V, T.zeros(1)) * \
                    (logprob*(V_tp1-mean_V) - grad_mean_V.unsqueeze(-1)), axis=1)

        # Expectation
        elif(self.Type == 'mean'):
            loss = T.mean(V_tp1 * logprob, axis=1)

        # Conditional value-at-risk penalized
        elif(self.Type == 'CVaR-penalized'):
            # analytic expression -- root_scalar
            weights = T.zeros(V_tp1.shape, dtype=T.float, requires_grad=False)
            lambda0 = T.zeros(list(V_tp1.shape[i] for i in [0,2]), dtype=T.float, requires_grad=False)
            for idx1, row in enumerate(V_tp1):
                for idx2, col in enumerate(row.transpose(0,1)):
                    col_np = col.numpy()
                    def f(value):
                        return np.mean( np.minimum(np.exp((col_np-value-self.kappa)/self.kappa) - 1/self.alpha, 0) ) - 1 + (1/self.alpha)
                    
                    sol = optimize.root_scalar(f, bracket=[np.quantile(col_np,1-self.alpha)-5, np.quantile(col_np,1-self.alpha)+5])
                    lambda0[idx1,idx2] = sol.root
                    weights[idx1,:,idx2] = T.minimum(T.exp((col-lambda0[idx1,idx2]-self.kappa)/self.kappa), T.tensor(1/self.alpha))
            
            loss = T.mean(weights * logprob * (V_tp1 - self.kappa*T.log(weights+1e-5) - lambda0.unsqueeze(1)), axis=1)

        # Conditional value-at-risk with mean
        elif(self.Type == 'mean-CVaR'):
            quant = T.quantile(V_tp1, 1-self.alpha, axis=1).unsqueeze(-1).repeat(1,V_tp1.shape[1])
            cond = V_tp1 > quant
            loss = (1-self.kappa) * T.sum( logprob.masked_fill(~cond, 0.0)*(V_tp1.masked_fill(~cond, 0.0) - quant), axis=1) \
                    / T.sum(cond, axis=1) \
                    + self.kappa * T.mean(V_tp1 * logprob, axis=1)

        return loss