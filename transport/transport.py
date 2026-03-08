import enum
import logging

import numpy as np
import torch as th
from tqdm import tqdm

from . import path
from .integrators import ode, sde
from .utils import EasyDict, log_state, mean_flat, sum_flat


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)


class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
        time_schedule="linear",
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.time_schedule = time_schedule

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        def _fn(x): return -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size ==
                             0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
                and (self.model_type != ModelType.VELOCITY or sde):  # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (
                diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size ==
                             0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """

        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return t, x0, x1

    def training_losses(
        self,
        model,
        x1,
        model_kwargs=None
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs == None:
            model_kwargs = {}

        t, x0, x1 = self.sample(x1)
        t, xt_condot, ut = self.path_sampler.plan(t, x0, x1)

        t_expand = path.expand_t_like_x(t, x0)

        # 统一使用 CondOT 路径
        xt = xt_condot
        model_output = model(xt, t, **model_kwargs)

        C_in = xt.shape[1]

        if self.model_type == ModelType.VELOCITY:
            # Model output layout: [flow_u | gauss_mu | intensity]
            u_theta = model_output[:, :C_in]
            mu_theta = model_output[:, C_in:2*C_in]
            intensity_logits = model_output[:, 2*C_in:3*C_in]

            terms = {}
            terms['pred'] = u_theta

            # Flow loss uses mean per user request
            diff = u_theta - ut
            breg_type = getattr(self, 'bregman_type', 'mse')
            if breg_type == 'cosh':
                L_flow = (th.cosh(diff) - 1).mean()
            elif breg_type == 'exp':
                L_flow = (th.exp(diff) - diff - 1).mean()
            else:
                L_flow = (diff ** 2).mean()

            # CondOT Jump 解析解 (GM 论文 Eq. 2596)
            k_t = xt**2 - (t_expand + 1) * xt * x1 - \
                (1 - t_expand)**2 + t_expand * x1**2
            lambda_target = th.clamp(k_t, min=0.0) / ((1 - t_expand)**3 + 1e-8)
            # Relax the artificial clamp from 500 to a safer bound that allows sharp jumps near t=0.999
            lambda_target = th.clamp(lambda_target, max=100000.0)
            lambda_target_masked = lambda_target

            # Extract lambda intensity: Use EXP instead of SOFTPLUS (Crucial!)
            # Softplus with Poisson NLL has a bounded downward gradient (max 1), causing it to get "stuck" at high values.
            # Exp() is the natural link function for Poisson, providing symmetric restoring force: grad = lambda_theta - lambda_target.
            # max=15.0 ensures lambda maxes around ~3.2M, providing enough dynamic range
            clamped_logits = th.clamp(intensity_logits, min=-8.0, max=15.0)
            lambda_theta = th.exp(clamped_logits)

            # The Jump Ground Truth target is simply the un-noised data x1
            target_y = x1

            # Define Jump Loss
            # 1. Intensity Match: Poisson Bregman divergence (strictly >= 0)
            # According to the CGM paper (Appendix E.3, Eq 2117-2120),
            # The exact Bregman divergence for the Poisson jump kernel is:
            # D(target, pred) = pred - target * log(pred) - (target - target * log(target))
            # The last constant shift is formally part of the divergence definition.
            poisson_min = lambda_target_masked - lambda_target_masked * \
                th.log(lambda_target_masked + 1e-8)
            loss_lambda = (lambda_theta - lambda_target_masked *
                           clamped_logits - poisson_min).mean()

            # 2. Laplace MAE (L1) (GM theory: Bregman divergence on Laplace kernel)
            # We fix the Laplace scale `b` to the physical standard deviation (1-t)
            # and ignore predicting the variance, resolving variance collapse completely.
            b_t = (1.0 - t_expand) + 1e-8
            l1_raw = th.abs(mu_theta - target_y)
            loss_laplace_mu = l1_raw / b_t

            loss_jump_mu = (lambda_target_masked * loss_laplace_mu).mean()
            L_jump_raw = loss_lambda + loss_jump_mu

            terms['loss_flow'] = L_flow
            terms['loss_jump'] = L_jump_raw
            terms['loss_jump_lambda'] = loss_lambda
            terms['loss_jump_mu'] = loss_jump_mu
            terms['lambda_theta'] = lambda_theta.mean().detach()
            terms['lambda_target'] = lambda_target_masked.mean().detach()
            terms['mae'] = l1_raw.mean().detach()
            terms['loss'] = L_flow + L_jump_raw

        else:
            B, *_, C = xt.shape
            assert model_output.size() == (B, *xt.size()[1:-1], C)
            terms = {}
            terms['pred'] = model_output

            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(
                path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                terms['loss'] = (weight * ((model_output - x0)
                                 ** 2)).flatten(1).sum(1).mean()
            else:
                terms['loss'] = (
                    weight * ((model_output * sigma_t + x0) ** 2)).flatten(1).sum(1).mean()

        return terms

    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            if model_output.shape[1] > x.shape[1]:
                model_output = model_output[:, :x.shape[1]]
            # by change of variable
            return (-drift_mean + drift_var * model_output)

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(
                path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            if model_output.shape[1] > x.shape[1]:
                model_output = model_output[:, :x.shape[1]]
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            if model_output.shape[1] > x.shape[1]:
                model_output = model_output[:, :x.shape[1]]  # Extract u_theta
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

    def get_score(
        self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            def _score_fn(x, t, model, **kwargs):
                out = model(x, t, **kwargs)
                if out.shape[1] > x.shape[1]:
                    out = out[:, :x.shape[1]]
                return out / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
            score_fn = _score_fn
        elif self.model_type == ModelType.SCORE:
            def _score_fn(x, t, model, **kwargs):
                out = model(x, t, **kwargs)
                if out.shape[1] > x.shape[1]:
                    out = out[:, :x.shape[1]]
                return out
            score_fn = _score_fn
        elif self.model_type == ModelType.VELOCITY:
            def _score_fn(x, t, model, **kwargs):
                out = model(x, t, **kwargs)
                if out.shape[1] > x.shape[1]:
                    out = out[:, :x.shape[1]]
                return self.path_sampler.get_score_from_velocity(out, x, t)
            score_fn = _score_fn
        else:
            raise NotImplementedError()

        return score_fn


class Sampler:
    """Sampler class for the transport model"""

    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion

        sde_drift = \
            lambda x, t, model, **kwargs: \
            self.drift(x, t, model, **kwargs) + diffusion_fn(x,
                                                             t) * self.score(x, t, model, **kwargs)

        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion

    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""

        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            # simple aliasing; the original name was too long
            alpha = self.transport.path_sampler.compute_alpha_t
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / \
                alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method
        )

        last_step_fn = self.__get_last_step(
            sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(
                xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        return _ode.sample

    def sample_jump_flow(
        self,
        *,
        num_steps=250,
        pure_jump=False,
        stochastic_jump=True,
        jump_alpha=0.5,
    ):
        """returns a sampling function for mixed CTMC/SDE (Algorithm 2) - Euler MS method
        Args:
        - num_steps: the actual number of integration steps performed
        - pure_jump: if True, sets jump_alpha to 1.0
        - stochastic_jump: if True, adds noise to jump landing point
        - jump_alpha: weight of the jump component
        """

        if pure_jump:
            jump_alpha = 1.0

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        dt = (t1 - t0) / num_steps

        def _sample(init, model, **model_kwargs):
            x = init
            xs = [x]

            for i in tqdm(range(num_steps), desc="Jump+Flow MS (Euler)"):
                t = t0 + i * dt

                C = x.shape[1]
                t_vec = th.full((x.shape[0],), t, device=x.device)

                model_output = model(x, t_vec, **model_kwargs)
                out_channels = model_output.shape[1]

                # Laplace Jump-Flow [flow | mu | intensity]
                if out_channels == 3 * C:
                    v_theta_flow = model_output[:, :C]
                    mu_theta = model_output[:, C:2*C]
                    intensity = model_output[:, 2*C:3*C]
                # Legacy Gaussian Jump-Flow [flow | mu | intensity | logvar]
                elif out_channels == 4 * C:
                    v_theta_flow = model_output[:, :C]
                    mu_theta = model_output[:, C:2*C]
                    intensity = model_output[:, 2*C:3*C]
                    # log_var = model_output[:, 3*C:4*C] (unused in pure inference usually)
                elif out_channels == C:  # Pure velocity / ODE model
                    v_theta_flow = model_output
                    x = x + v_theta_flow * dt
                    xs.append(x)
                    continue
                else:
                    raise RuntimeError(
                        f"Unexpected model output channels: {out_channels} for input channels: {C}")

                lambda_t = th.exp(th.clamp(intensity, -8, 15.0))

                # Approximate the integral of lambda_t
                integral = 0.5 * lambda_t * \
                    (1.0 - t) * (1.0 - ((1.0 - t)**2 / ((1.0 - t - dt)**2 + 1e-8)))
                p_jump = 1.0 - th.exp(jump_alpha * integral)
                p_jump = th.clamp(p_jump, 0.0, 1.0)

                is_cfg = "cfg_scale" in model_kwargs
                if is_cfg:
                    half_N = x.shape[0] // 2
                    p_jump_half = p_jump[:half_N]
                    jump_mask_half = th.bernoulli(p_jump_half)
                    jump_mask = th.cat([jump_mask_half, jump_mask_half], dim=0)
                else:
                    jump_mask = th.bernoulli(p_jump)

                x_flow = x + (1.0 - jump_alpha) * v_theta_flow * dt

                if stochastic_jump:
                    # Use the physical, fixed scale (1-t) instead of the dead prediction `log_var`
                    std = max(1.0 - t, 0.0)
                    if is_cfg:
                        noise_half = th.randn_like(x[:half_N])
                        noise = th.cat([noise_half, noise_half], dim=0)
                        x_jump = mu_theta + std * noise
                    else:
                        x_jump = mu_theta + std * th.randn_like(x)
                else:
                    x_jump = mu_theta

                x = jump_mask * x_jump + (1 - jump_mask) * x_flow
                xs.append(x)

            return xs
        return _sample

    def sample_pc(
        self,
        *,
        num_steps=50,
        corrector_steps=1,
        snr=0.1,
    ):
        """Predictor-Corrector sampling (Euler predictor + Langevin corrector)"""
        return self.sample_jump_flow(
            num_steps=num_steps,
            corrector_steps=corrector_steps,
            snr=snr,
        )

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float,
                             device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(
                    th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(
                    grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn
