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

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]):  # avoid endpoint singularities on CondOT-style paths

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
        t = th.rand((x1.shape[0],), device=x1.device) * (t1 - t0) + t0
        t = t.to(x1)

        return t, x0, x1

    def _condot_jump_gaussian_stats(self, z, t_expand):
        """Exact first and second moments of the CondOT landing kernel.

        For p_t(y|z)=N(tz, (1-t)^2), the exact landing kernel is
        J_t(y|z) ∝ [-k_t(y|z)]_+ p_t(y|z).
        In standardized coordinates y = tz + (1-t) eps, this becomes
        q_z(eps) ∝ [1 + z eps - eps^2]_+ phi(eps),
        which admits closed-form first and second moments on the support
        [a(z), b(z)] where a,b are the roots of eps^2 - z eps - 1 = 0.
        """
        z64 = z.to(th.float64)
        t64 = t_expand.to(th.float64)
        sigma = 1.0 - t64
        sqrt_two = np.sqrt(2.0)
        normalizer = 1.0 / np.sqrt(2.0 * np.pi)

        root = th.sqrt(z64.square() + 4.0)
        a = 0.5 * (z64 - root)
        b = 0.5 * (z64 + root)

        phi_a = normalizer * th.exp(-0.5 * a.square())
        phi_b = normalizer * th.exp(-0.5 * b.square())
        cdf_a = 0.5 * (1.0 + th.erf(a / sqrt_two))
        cdf_b = 0.5 * (1.0 + th.erf(b / sqrt_two))

        mass = cdf_b - cdf_a
        diff_pdf = phi_a - phi_b
        Z = b * phi_a - a * phi_b

        mean_eps = (z64 * mass - 2.0 * diff_pdf) / (Z + 1e-12)
        second_eps = (
            -2.0 * mass + (2.0 * b - a) * phi_a + (b - 2.0 * a) * phi_b
        ) / (Z + 1e-12)
        var_eps = second_eps - mean_eps.square()

        mu = t64 * z64 + sigma * mean_eps
        var = sigma.square() * var_eps + 1e-8
        return mu.to(z.dtype), var.to(z.dtype)

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
            # Model output layout: [flow_u | jump_d | jump_rho_raw]
            u_theta = model_output[:, :C_in]
            jump_d_theta = model_output[:, C_in:2*C_in]
            jump_rho_raw_theta = model_output[:, 2*C_in:3*C_in]

            terms = {}
            terms['pred'] = u_theta

            # 1. Flow Loss: True Bregman Divergence (Appendix J)
            # GM theory requires D(a,b) = phi(a) - phi(b) - phi'(b)*(a - b).
            # If we just do cosh(a-b)-1, it's not a Bregman Divergence and Generator Matching fails!
            # Let a = ut (target), b = u_theta (prediction). diff = u_theta - ut = b - a.
            # Then a - b = -diff, and phi'(b) = alpha * sinh(alpha * u_theta).
            diff = u_theta - ut
            breg_type = getattr(self, 'bregman_type', 'mse')
            alpha = getattr(self, 'bregman_alpha', 1.0)

            if breg_type == 'cosh':
                D_cosh = th.cosh(alpha * ut) - th.cosh(alpha *
                                                       u_theta) + alpha * th.sinh(alpha * u_theta) * diff
                L_flow = (0.5 * (diff ** 2) + 0.5 * D_cosh).mean()
            elif breg_type == 'exp':
                D_exp = th.exp(alpha * ut) - th.exp(alpha * u_theta) + \
                    alpha * th.exp(alpha * u_theta) * diff
                L_flow = (0.5 * (diff ** 2) + 0.5 * D_exp).mean()
            else:
                L_flow = (diff ** 2).mean()

            # CondOT Jump 解析解 (GM 论文 Eq. 2596)
            # 彻底摒弃外部除以 (1-t)^3 的隐患，将奇异权重剥离到分布对数散度之外
            k_t = xt**2 - (t_expand + 1) * xt * x1 - \
                (1 - t_expand)**2 + t_expand * x1**2

            # 1. 提取物理真实的核心分子特征量 (rho) + Scaled Log1p (软化保护)
            # 针对真值在 0.08~0.14 的特性，采用带缩放的 Log1p: y = s * log(1 + x/s)
            # s=4.0 意味着在 x < 4.0 的宽广范围内保持近似线性，仅抑制极端的数值爆发
            # 这比纯 log1p (s=1) 更能保留纹理细节，同时防止梯度爆炸
            scale = 4.0
            target_rho = scale * th.log1p(th.relu(k_t) / scale)
            
            # 2. 算子提取外部绝对共轭时间权重 W_t = 1 / (1-t)^3
            # 使用代数平滑 (Algebraic Smooth Clamping) 替代硬截断 clamp
            # 公式: W_safe = 1 / ( (1-t)^3 + 1/M )
            # 优势: 处处可导，前期完美贴合理论权重，后期平滑收敛至 M，消除梯度转折点
            # 设定最大渐近线 M = 1000.0 (兼顾训练稳定与 250 步采样的高保真度)
            max_weight_asymptote = 1000.0
            remaining = th.clamp(1 - t_expand, min=1e-8)
            safe_weight = 1.0 / (remaining**3 + (1.0 / max_weight_asymptote))

            target_mu_jump, target_var_jump = self._condot_jump_gaussian_stats(
                x1, t_expand)
            jump_mu_theta = xt + remaining * jump_d_theta

            clamped_rho_raw = th.clamp(jump_rho_raw_theta, min=-20.0, max=12.0)
            rho_theta = th.exp(clamped_rho_raw)

            jump_diff = jump_mu_theta - target_mu_jump

            # 3. Factorized Poisson Divergence (剥离端点后的纯散度测度)
            # Mathematical equivalence proven:
            # L = lambda_theta - lambda_target + lambda_target * log(lambda_target / lambda_theta) 
            #   = W_t * [ rho_theta - target_rho + target_rho * log(target_rho / rho_theta) ]
            divergence_rho = (
                rho_theta
                - target_rho
                + target_rho * th.log((target_rho + 1e-8) / (rho_theta + 1e-8))
            )
            loss_lambda = (safe_weight * divergence_rho).mean()
            
            # Generator Matching paper formulation & Target Variance Matching:
            # 强制剔除输出 var，网路不再猜测已知的分布方差，阻断 Variance Escape。
            pred_var = target_var_jump
            
            # 由于 pred_var == target_var_jump，KL 散度原本形式如下：
            # 0.5 * (log(target_var) - log(target_var) + (target_var + jump_diff^2) / target_var - 1)
            # 全面化简后等于：0.5 * (jump_diff^2 / target_var) = Gaussian NLL with true var!
            loss_jump_kl = 0.5 * (jump_diff.square() / (target_var_jump + 1e-8))
            
            # 同样地，把乘数 lambda_target 转换为 (safe_weight * target_rho)
            loss_jump_distribution = (safe_weight * target_rho * loss_jump_kl).mean()
            L_jump = loss_lambda + loss_jump_distribution

            # For Logging 
            terms['loss_flow'] = L_flow
            terms['loss_jump'] = L_jump
            terms['loss_jump_lambda'] = loss_lambda
            terms['loss_jump_mu'] = (safe_weight * target_rho * 0.5 * jump_diff.square() / (target_var_jump + 1e-8)).mean()
            terms['loss_jump_var'] = th.zeros_like(terms['loss_jump_mu']) # Deprecated metric
            
            # Log the un-scaled rho values directly to observe internal network convergence
            terms['lambda_theta'] = rho_theta.mean().detach()
            terms['lambda_target'] = target_rho.mean().detach()
            terms['jump_var_theta'] = pred_var.mean().detach()
            terms['jump_var_target'] = target_var_jump.mean().detach()

            # Fix: calculate jump_rmse ONLY over the active support region (where target_rho > 0)
            active_mask = (target_rho > 0).float()
            active_sum = active_mask.sum().clamp(min=1.0)
            weighted_mse = (jump_diff.square() * active_mask).sum() / active_sum
            terms['jump_rmse'] = th.sqrt(weighted_mse.detach() + 1e-12)

            # 严格使用 Markov Superposition 原理下的 1:1 比重 (0.5 Flow + 0.5 Jump)
            terms['loss'] = 0.5 * L_flow + 0.5 * L_jump

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
        self,
        jump_alpha=0.5,
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
        jump_alpha=0.5,
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
        - jump_alpha: the alpha parameter used for continuous flow scaling
        """
        drift = self.transport.get_drift(jump_alpha=jump_alpha)

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
        jump_y_noise_scale=1.0,
        jump_alpha=0.5,
    ):
        """returns a sampling function for Jump+Flow Markov superposition with Euler updates
        Args:
        - num_steps: the actual number of integration steps performed
        - pure_jump: if True, sets jump_alpha to 1.0
        - stochastic_jump: if True, adds noise to jump landing point
        - jump_y_noise_scale: scale multiplier for the jump std implied by Q_theta
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

                if out_channels == 3 * C:
                    v_theta_flow = model_output[:, :C]
                    jump_d_theta = model_output[:, C:2*C]
                    jump_rho_raw_theta = model_output[:, 2*C:3*C]
                elif out_channels == C:
                    v_theta_flow = model_output
                    x = x + v_theta_flow * dt
                    xs.append(x)
                    continue
                else:
                    raise RuntimeError(f"Unexpected model output channels: {out_channels}")

                remaining = th.clamp(th.tensor(1.0 - t, device=x.device, dtype=x.dtype), min=1e-8)
                remaining_next = max(1.0 - t - dt, 0.0)

                jump_mu = x + remaining * jump_d_theta
                
                x1_hat = x + remaining * v_theta_flow
                t_vec_expand = path.expand_t_like_x(t_vec, x)
                _, target_var_jump = self.transport._condot_jump_gaussian_stats(x1_hat, t_vec_expand)
                jump_var = target_var_jump

                jump_rho_raw = th.clamp(jump_rho_raw_theta, min=-20.0, max=12.0)
                jump_rho = th.exp(jump_rho_raw)
                lambda_t = jump_rho / ((remaining ** 3) + 1e-8)

                hazard = 0.5 * jump_alpha * lambda_t * remaining * (
                    (remaining ** 2) / ((remaining_next ** 2) + 1e-8) - 1.0
                )
                p_jump = -th.expm1(-hazard)

                is_cfg = "cfg_scale" in model_kwargs
                if is_cfg:
                    half_N = x.shape[0] // 2
                    p_jump_half = p_jump[:half_N]
                    jump_mask_half = th.bernoulli(p_jump_half)
                    jump_mask = th.cat([jump_mask_half, jump_mask_half], dim=0).bool()
                else:
                    jump_mask = th.bernoulli(p_jump).bool()

                x_flow = x + (1.0 - jump_alpha) * v_theta_flow * dt

                if stochastic_jump:
                    std = th.sqrt(jump_var) * jump_y_noise_scale
                    if is_cfg:
                        noise_half = th.randn_like(x[:half_N])
                        noise = th.cat([noise_half, noise_half], dim=0)
                        x_jump = jump_mu + std * noise
                    else:
                        x_jump = jump_mu + std * th.randn_like(x)
                else:
                    x_jump = jump_mu

                x = th.where(jump_mask, x_jump, x_flow)
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
