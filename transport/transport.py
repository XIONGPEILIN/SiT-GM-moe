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
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
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

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
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
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        model_output = model(xt, t, **model_kwargs)
        
        C_in = xt.shape[1]
        
        if self.model_type == ModelType.VELOCITY:
            # For VELOCITY model, we have incorporated jump head according to gm.md
            u_theta = model_output[:, :C_in]
            jump_head = model_output[:, C_in:]
            
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
            
            # GM jump objective computation
            t_expand = path.expand_t_like_x(t, xt)
            
            if getattr(self, "time_schedule", "linear") == "cubic":
                alpha_t = 1 - (1 - t_expand)**3
                d_alpha_t = 3 * (1 - t_expand)**2
            else:
                alpha_t = t_expand
                d_alpha_t = 1.0

            # k_func uses strictly the theoretical probability time alpha_t
            def k_func(val):
                return val**2 - (alpha_t+1)*val*x1 - (1-alpha_t)**2 + alpha_t*x1**2
            
            # lambda_target (Chain rule: \lambda_t = d_alpha_t * \lambda_{old}(alpha_t) )
            lambda_target = th.relu(k_func(xt)) * d_alpha_t / ((1 - alpha_t)**3 + 1e-8)
            
            # Fetch num_bins and jump_range from model safely
            has_module = hasattr(model, 'module')
            base_model = model.module if has_module else model
            num_bins = getattr(base_model, 'num_bins', 128)
            jump_range = getattr(base_model, 'jump_range', 4.0)
            
            y_bins = th.linspace(-jump_range, jump_range, num_bins, device=xt.device)
            # shape mapping for bins broadcast: (1, 1, 1, 1, num_bins)
            y_expanded = y_bins.view(*([1]*xt.dim()), num_bins)
            
            t_expand_bin = t_expand.unsqueeze(-1)
            x1_bin = x1.unsqueeze(-1)
            
            def k_func_bin(val):
                return val**2 - (t_expand_bin+1)*val*x1_bin - (1-t_expand_bin)**2 + t_expand_bin*x1_bin**2
            
            # J_target for multiple bins (F15)
            k_val_y = k_func_bin(y_expanded)
            relu_neg_k_y = th.relu(-k_val_y)
            
            mu = t_expand_bin * x1_bin
            sigma = th.clamp(1 - t_expand_bin, min=1e-5)
            
            bin_width = 2.0 * jump_range / max(1, num_bins - 1)
            y_left = y_expanded - bin_width / 2.0
            y_right = y_expanded + bin_width / 2.0
            
            normal_dist = th.distributions.Normal(mu, sigma)
            cdf_diff = normal_dist.cdf(y_right) - normal_dist.cdf(y_left)
            cdf_diff = th.clamp(cdf_diff, min=0.0)
            
            J_target_unnorm = relu_neg_k_y * cdf_diff
            J_target_sum = J_target_unnorm.sum(dim=-1, keepdim=True) + 1e-8
            J_target = J_target_unnorm / J_target_sum
            
            Q_target = lambda_target.unsqueeze(-1) * J_target
            
            # Predict
            jump_head_reshaped = jump_head.view(xt.size(0), C_in, num_bins + 1, *xt.shape[2:])
            # Transpose to put bins array at the end: (B, C, H, W, num_bins+1)
            # In pytorch, we can permute. Original: 0=B, 1=C, 2=bins, 3=H, 4=W
            dims = list(range(jump_head_reshaped.dim()))
            dims.append(dims.pop(2))
            jump_head_reshaped = jump_head_reshaped.permute(*dims)
            
            logits_b = jump_head_reshaped[..., :num_bins]
            intensity_logits = jump_head_reshaped[..., num_bins]
            
            J_theta = th.softmax(logits_b, dim=-1)
            lambda_theta = th.nn.functional.softplus(intensity_logits)
            
            Q_theta = lambda_theta.unsqueeze(-1) * J_theta
            
            # D = sum_{y} [ Q_theta(y;x) - Q(y;x) * log Q_theta(y;x) ]
            jump_loss_elementwise = Q_theta - Q_target * th.log(Q_theta + 1e-8)
            
            jump_loss_per_pixel = jump_loss_elementwise.sum(dim=-1)
            
            # Spatial aggregation via mean per user request
            L_jump_raw = jump_loss_per_pixel.mean()
            
            terms['loss_flow'] = L_flow
            terms['loss_jump'] = L_jump_raw  # Keep original value for wandb logging
            terms['loss'] = L_flow + L_jump_raw
            
        else:
            B, *_, C = xt.shape
            assert model_output.size() == (B, *xt.size()[1:-1], C)
            terms = {}
            terms['pred'] = model_output
            
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            
            if self.model_type == ModelType.NOISE:
                terms['loss'] = (weight * ((model_output - x0) ** 2)).flatten(1).sum(1).mean()
            else:
                terms['loss'] = (weight * ((model_output * sigma_t + x0) ** 2)).flatten(1).sum(1).mean()
                
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
            return (-drift_mean + drift_var * model_output) # by change of variable
        
        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            if model_output.shape[1] > x.shape[1]:
                model_output = model_output[:, :x.shape[1]]
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            if model_output.shape[1] > x.shape[1]:
                model_output = model_output[:, :x.shape[1]] # Extract u_theta
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
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
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
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
    
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
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
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

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)
            

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

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
        num_steps=50,
        reverse=False,
        jump_alpha=0.5,
        jump_alpha_schedule="constant",
        flow_sampler="euler",
        corrector_steps=0,
        snr=0.1,
    ):
        """returns a sampling function for mixed CTMC/SDE (Algorithm 2)
        Args:
        - num_steps: the actual number of integration steps performed
        - reverse: whether solving in reverse; default to False
        """
        # Note: Jump logic is currently only valid for forward generation (t=0 to t=1)
        
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        t_list = th.linspace(t0, t1, num_steps)
        dt = t_list[1] - t_list[0]

        def _sample(init, model, **model_kwargs):
            x = init
            xs = [x]
            has_module = hasattr(model, 'module')
            base_model = model.module if has_module else model
            num_bins = getattr(base_model, 'num_bins', 128)
            jump_range = getattr(base_model, 'jump_range', 4.0)
            in_channels = getattr(base_model, 'in_channels', x.shape[1])

            for i, ti in tqdm(enumerate(t_list[:-1]), total=len(t_list)-1, desc="Jump Flow Euler (with PC support)"):
                with th.no_grad():
                    t_vec = th.ones(x.shape[0], device=x.device) * ti
                    
                    for _ in range(corrector_steps):
                        score = self.score(x, t_vec, model, **model_kwargs)
                        noise = th.randn_like(x)
                        eps = snr * dt
                        x = x + 0.5 * eps * score + th.sqrt(eps) * noise

                    model_output = model(x, t_vec, **model_kwargs)
                    cur_t = ti.item()
                    current_alpha = jump_alpha if jump_alpha_schedule != "linear" else cur_t
                    
                    if model_output.shape[1] > in_channels:
                        v_theta = model_output[:, :in_channels]
                        jump_head = model_output[:, in_channels:]
                        
                        jump_head_reshaped = jump_head.view(x.shape[0], in_channels, num_bins + 1, *x.shape[2:])
                        dims = list(range(jump_head_reshaped.dim()))
                        dims.append(dims.pop(2))
                        jump_head_reshaped = jump_head_reshaped.permute(*dims)
                        
                        logits_b = jump_head_reshaped[..., :num_bins]
                        intensity_logits = jump_head_reshaped[..., num_bins]
                        
                        lambda_t = th.nn.functional.softplus(intensity_logits)
                        
                        eps_val = 1e-5
                        h = dt.item()
                        
                        if reverse:
                            raise NotImplementedError("Reverse sampling for Jumps not fully supported.")
                        
                        if jump_alpha_schedule == "linear":
                            current_jump_weight = cur_t
                        else:
                            current_jump_weight = jump_alpha

                        if cur_t + h >= 1.0 - eps_val:
                            # Avoid singularity exactly at t=1
                            R = th.zeros_like(lambda_t)
                        else:
                            lambda_t_weighted = current_jump_weight * lambda_t
                            R_val = 0.5 * lambda_t_weighted * (1 - cur_t) * (1 - ((1 - cur_t)**2) / ((1 - cur_t - h)**2))
                            R = th.exp(R_val)

                        # --- Probability Flow ODE (PF-ODE) for Jump Generator ---
                        # Instead of discrete multinomial leaps (which break the continuous VAE latent space and cause colorful noise),
                        # we construct the equivalent Continuous Velocity Field based on the Kramers-Moyal expansion first-order term.
                        
                        # 1. Expected target position (Center of gravity of the Jump logits)
                        J_theta = th.softmax(logits_b, dim=-1)
                        y_bins = th.linspace(-jump_range, jump_range, num_bins, device=x.device)
                        expected_y = (J_theta * y_bins).sum(dim=-1)
                        
                        # 2. Convert to Velocity Field: v_jump = lambda(x) * (E[y] - x)
                        # We approximate lambda(x) dt with p_jump (the integration of lambda over the timestep dt)
                        # So the step displacement is roughly: dx_jump = p_jump * (E[y] - x)
                        # Which we can write simply as a displacement offset to be applied.
                        p_jump = th.clamp(1 - R, 0.0, 1.0)
                        
                        # We assign this effective continuous displacement back to m and jump_vals
                        # so that x = x + m * (jump_vals - x) exactly yields our PF-ODE integration step.
                        jump_vals = expected_y
                        m = p_jump
                    else:
                        m = 0
                        jump_vals = 0
                        v_theta = model_output

                    if self.transport.model_type == ModelType.NOISE:
                        drift_mean, drift_var = self.transport.path_sampler.compute_drift(x, t_vec)
                        sigma_t, _ = self.transport.path_sampler.compute_sigma_t(path.expand_t_like_x(t_vec, x))
                        score = v_theta / -sigma_t
                        c_drift = (-drift_mean + drift_var * score)
                    elif self.transport.model_type == ModelType.SCORE:
                        drift_mean, drift_var = self.transport.path_sampler.compute_drift(x, t_vec)
                        c_drift = (-drift_mean + drift_var * v_theta)
                    else:
                        c_drift = v_theta

                    c_drift = c_drift * (1.0 - current_alpha)
                    x_continuous = x + c_drift * dt

                    if flow_sampler == "heun" and i < len(t_list) - 2:
                        t_vec_next = th.ones(x.shape[0], device=x.device) * t_list[i+1]
                        model_output_next = model(x_continuous, t_vec_next, **model_kwargs)
                        v_theta_next = model_output_next[:, :in_channels] if model_output_next.shape[1] > in_channels else model_output_next

                        if self.transport.model_type == ModelType.NOISE:
                            drift_mean_next, drift_var_next = self.transport.path_sampler.compute_drift(x_continuous, t_vec_next)
                            sigma_t_next, _ = self.transport.path_sampler.compute_sigma_t(path.expand_t_like_x(t_vec_next, x_continuous))
                            score_next = v_theta_next / -sigma_t_next
                            c_drift_next = (-drift_mean_next + drift_var_next * score_next)
                        elif self.transport.model_type == ModelType.SCORE:
                            drift_mean_next, drift_var_next = self.transport.path_sampler.compute_drift(x_continuous, t_vec_next)
                            c_drift_next = (-drift_mean_next + drift_var_next * v_theta_next)
                        else:
                            c_drift_next = v_theta_next

                        next_alpha = jump_alpha if jump_alpha_schedule != "linear" else t_list[i+1].item()
                        c_drift_next = c_drift_next * (1.0 - next_alpha)
                        x_continuous = x + 0.5 * (c_drift + c_drift_next) * dt
                    
                    if isinstance(m, int) and m == 0:
                        x = x_continuous
                    else:
                        x = m * jump_vals + (1 - m) * x_continuous
                        
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
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
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