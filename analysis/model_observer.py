import numpy as np

import cogphysics.lib.rvs as rvs
import cogphysics.lib.stats as stats

import pdb

class ModelObserver(object):

    def __init__(self, ime_samples, kappas, N=1, Cf=10, Cr=6,
                 pseudocount=1, n_F=2, n_R=7, smooth=False):

        self.n_F = n_F
        self.n_R = n_R
        self.OUTCOMES = np.arange(n_F)
        self.RESPONSES = np.arange(n_R)

        self.ime_samples = ime_samples.copy()
        self.ime_output = None

        self.kappas = kappas.copy()
        self.n_kappa = len(self.kappas)
        self.thetas = [np.log(np.ones(self.n_kappa) / float(self.n_kappa))]
        self.joint = self.thetas[0].copy()

        self.time = 0
        self.stimuli = []

        self._N = N
        self._Cf = Cf
        self._Cr = Cr

        self._pseudocount = pseudocount
        self._smooth = smooth
        self._loss = None

    def IME(self, t):
        """P(F_t | S_t, kappa)"""
        if self.ime_output is None:
            samps = self.ime_samples
            shape = list(samps.shape[:-1]) + [self.n_F]
            alpha = np.ones(shape) * self._pseudocount
            prior = rvs.Dirichlet(alpha)
            if self._smooth:
                diff = samps[:, :, None, :] - self.OUTCOMES[None, None, :, None]
                succ = np.exp2(-np.abs(diff))
                post = prior.updateMultinomial(succ, axis=-1)
                mle = np.log(post.mode.copy())
            else:
                succ = samps[:, :, None, :] == self.OUTCOMES[None, None, :, None]
                post = prior.updateMultinomial(succ, axis=-1)
                mle = np.log(post.mean.copy())
            self.ime_output = mle
        return self.ime_output[t]

    def Pt_kappa(self, t):
        """P_t(kappa) = theta_t"""
        return self.thetas[t]

    def P_Ft(self, t):
        """P(F_t | S_t)"""
        p_F_kappa = self.IME(t)
        Pt_kappa = self.Pt_kappa(t)[:, None]
        dist = stats.normalize(p_F_kappa + Pt_kappa, axis=0)[0]
        return dist

    def Loss(self):
        if self._loss is None:
            N, Cf, Cr = self._N, self._Cf, self._Cr
            f = self.OUTCOMES[:, None]
            r = self.RESPONSES[None, :]
            sf = ((f + N).astype('f8') / self.n_F) - 0.5
            sr = ((r - N).astype('f8') / self.n_R) - 0.5
            ssf = 1.0 / (1 + np.exp(-Cf * sf))
            ssr = 1.0 / (1 + np.exp(-Cr * sr))
            self._loss = np.sqrt(np.abs(ssf - ssr))
        return self._loss
    
    def Risk(self, t):
        p_F = np.exp(self.P_Ft(t)[:, None])
        loss = self.Loss()
        risk = np.sum(loss * p_F, axis=0)
        return risk

    ##################################################################

    def viewStimulus(self, S):
        self.stimuli.append(S)
        self.time = len(self.stimuli) - 1

    def generateResponse(self):
        """Compute optimal response to the current stimulus"""
        risk = self.Risk(self.time)
        response = np.argmin(risk)
        return response

    def viewFeedback(self, F):
        # Likelihood of the true outcomes
        P_Ft_kappa = self.IME(self.time)
        lh_F_kappa = P_Ft_kappa[:, F]

        ## Calculate theta_t = P_t(kappa)
        joint = lh_F_kappa + self.joint[self.time]
        Pt_kappa = stats.normalize(joint, axis=-1)[1]

        self.joint.append(joint)
        self.thetas.append(Pt_kappa)

        return lh_F_kappa

    def learningCurve(self, F):
        ime = self.IME(slice(None))
        lh = np.vstack([
            self.thetas[0],
            np.choose(F[:, None], ime.transpose((2, 0, 1)))])
        self.joint = np.cumsum(lh, axis=0)
        self.thetas = stats.normalize(self.joint, axis=1)[1]
        self.time = len(self.thetas)
        return lh, self.joint, self.thetas
