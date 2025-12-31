# sd_accel/core/dpm_solver_multistep_scheduler.py
"""
手动实现的 DPM-Solver++ 多步调度器
基于 DPM-Solver++ 算法，用于加速扩散模型的采样过程
"""
import torch
import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class DPMSolverMultistepSchedulerConfig:
    """DPM-Solver++ 调度器配置"""
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    trained_betas: Optional[np.ndarray] = None
    algorithm_type: str = "dpmsolver++"
    solver_type: str = "midpoint"
    lower_order_final: bool = True
    use_karras_sigmas: bool = False
    timestep_spacing: str = "linspace"
    steps_offset: int = 0


class DPMSolverMultistepScheduler:
    """
    手动实现的 DPM-Solver++ 多步调度器
    
    这是一个高阶求解器，用于加速扩散模型的采样过程。
    支持 DPM-Solver++ 算法，可以在更少的步数下生成高质量图像。
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[np.ndarray] = None,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        use_karras_sigmas: bool = False,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
    ):
        self.config = DPMSolverMultistepSchedulerConfig(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            algorithm_type=algorithm_type,
            solver_type=solver_type,
            lower_order_final=lower_order_final,
            use_karras_sigmas=use_karras_sigmas,
            timestep_spacing=timestep_spacing,
            steps_offset=steps_offset,
        )
        
        # 初始化 beta 调度
        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # scaled_linear 是 Stable Diffusion 使用的调度
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=torch.float32) ** 2
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 计算用于 DPM-Solver 的参数
        self.alpha_t = self.alphas_cumprod
        self.sigma_t = torch.sqrt(1.0 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - 0.5 * torch.log(1.0 - self.alphas_cumprod)
        
        # 内部状态
        self.timesteps = None
        self.num_inference_steps = None
        self.model_outputs = []
        self.sample = None
        
    @classmethod
    def from_config(cls, config: Union[dict, DPMSolverMultistepSchedulerConfig]) -> "DPMSolverMultistepScheduler":
        """
        从配置创建调度器实例
        
        Args:
            config: 配置字典或配置对象
            
        Returns:
            DPMSolverMultistepScheduler 实例
        """
        if isinstance(config, dict):
            # 从字典创建
            return cls(**config)
        elif isinstance(config, DPMSolverMultistepSchedulerConfig):
            # 从配置对象创建
            return cls(**config.__dict__)
        else:
            # 假设是 diffusers 的配置对象，尝试获取其属性
            config_dict = {}
            for key in [
                "num_train_timesteps", "beta_start", "beta_end", "beta_schedule",
                "trained_betas", "algorithm_type", "solver_type", "lower_order_final",
                "use_karras_sigmas", "timestep_spacing", "steps_offset"
            ]:
                if hasattr(config, key):
                    config_dict[key] = getattr(config, key)
            return cls(**config_dict)
    
    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        """
        设置推理时间步
        
        Args:
            num_inference_steps: 推理步数
            device: 设备
        """
        self.num_inference_steps = num_inference_steps
        
        # 计算时间步
        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(
                self.config.num_train_timesteps - 1,
                0,
                num_inference_steps,
                dtype=np.float32
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
            timesteps += self.config.steps_offset
        else:
            raise ValueError(f"Unknown timestep_spacing: {self.config.timestep_spacing}")
        
        # 应用 Karras sigmas 如果需要
        if self.config.use_karras_sigmas:
            sigmas = self.sigma_t.numpy()
            timesteps = self._convert_to_karras(in_timesteps=timesteps, num_inference_steps=num_inference_steps)
        
        timesteps = timesteps.astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        
        # 重置内部状态
        self.model_outputs = []
        self.sample = None
    
    def _convert_to_karras(self, in_timesteps: np.ndarray, num_inference_steps: int) -> np.ndarray:
        """转换为 Karras sigmas 调度"""
        sigma_max = self.sigma_t[-1].item()
        sigma_min = self.sigma_t[0].item()
        rho = 7.0  # Karras 参数
        
        step_indices = []
        for timestep in in_timesteps:
            index = int(timestep)
            if index >= len(self.sigma_t):
                index = len(self.sigma_t) - 1
            step_indices.append(index)
        
        # Karras sigmas
        t = np.arange(num_inference_steps)
        t = (sigma_max ** (1 / rho) + t / (num_inference_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        
        # 将 sigmas 映射回 timesteps
        timesteps = []
        for sigma in t:
            # 找到最接近的 timestep
            idx = np.argmin(np.abs(self.sigma_t.numpy() - sigma))
            timesteps.append(idx)
        
        return np.array(timesteps, dtype=np.float32)
    
    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        缩放模型输入（对于 DPM-Solver，通常不需要缩放）
        
        Args:
            sample: 输入样本
            timestep: 当前时间步
            
        Returns:
            缩放后的样本（对于 DPM-Solver，直接返回原样本）
        """
        return sample
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], dict]:
        """
        执行一步 DPM-Solver++ 更新
        
        Args:
            model_output: 模型预测的输出（噪声）
            timestep: 当前时间步
            sample: 当前样本
            return_dict: 是否返回字典格式
            
        Returns:
            更新后的样本
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()
        
        # 获取当前时间步的索引
        step_index = (self.timesteps == timestep).nonzero().item()
        
        # 保存模型输出
        self.model_outputs.append(model_output)
        
        # 获取当前和前序时间步
        prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1].item()
        t = timestep
        t_prev = prev_timestep
        
        # 获取对应的 lambda 值
        lambda_t = self.lambda_t[t]
        lambda_t_prev = self.lambda_t[t_prev] if t_prev > 0 else torch.tensor(0.0)
        
        # DPM-Solver++ 算法
        if self.config.solver_type == "midpoint":
            # Midpoint 方法（二阶）
            if step_index == 0:
                # 第一步：使用 Euler 方法
                h = lambda_t - lambda_t_prev
                D_0 = model_output
                sample_prev = sample - h * D_0
            else:
                # 后续步：使用 midpoint 方法
                h = lambda_t - lambda_t_prev
                D_0 = self.model_outputs[-1]  # 当前预测
                D_1 = self.model_outputs[-2] if len(self.model_outputs) >= 2 else D_0  # 前一步预测
                
                # Midpoint 方法
                if self.config.lower_order_final and step_index == len(self.timesteps) - 1:
                    # 最后一步使用一阶方法
                    sample_prev = sample - h * D_0
                else:
                    # 使用二阶 midpoint 方法
                    sample_prev = sample - h * D_0 - (h ** 2) / 2 * (D_0 - D_1) / (lambda_t_prev - self.lambda_t[self.timesteps[step_index - 1].item()] if step_index > 1 else h)
        else:
            # 默认使用 Euler 方法（一阶）
            h = lambda_t - lambda_t_prev
            D_0 = model_output
            sample_prev = sample - h * D_0
        
        # 从 lambda 空间转换回样本空间
        alpha_t_prev = self.alpha_t[t_prev] if t_prev > 0 else torch.tensor(1.0)
        alpha_t = self.alpha_t[t]
        
        # 转换回原始空间
        pred_original_sample = (sample - (1 - alpha_t) * model_output) / alpha_t
        prev_sample = alpha_t_prev * pred_original_sample + (1 - alpha_t_prev) * model_output
        
        if not return_dict:
            return (prev_sample,)
        
        return {"prev_sample": prev_sample}
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        向原始样本添加噪声
        
        Args:
            original_samples: 原始样本
            noise: 噪声
            timesteps: 时间步
            
        Returns:
            添加噪声后的样本
        """
        # 获取对应时间步的 alpha_cumprod
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

