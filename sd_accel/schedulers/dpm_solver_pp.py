# sd_accel/schedulers/dpm_solver_pp.py
import torch
import numpy as np
from typing import Optional, Union, List
from dataclasses import dataclass


@dataclass
class DPMSolverPPConfig:
    """DPM-Solver++ 配置"""
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    solver_order: int = 2  # 1, 2, or 3
    prediction_type: str = "epsilon"  # "epsilon" or "v_prediction"
    algorithm_type: str = "dpmsolver++"  # "dpmsolver++" or "dpmsolver"
    solver_type: str = "midpoint"  # "midpoint" or "heun"
    lower_order_final: bool = True
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    sample_max_value: float = 1.0


class DPMSolverPP:
    
    def __init__(self, config: DPMSolverPPConfig):
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.solver_order = config.solver_order
        
        # 初始化 beta schedule
        self.betas = self._get_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 计算 lambda (log SNR)
        self.lambda_t = self._compute_lambda()
        
        # 用于存储中间状态
        self.model_outputs = []
        self.timestep_list = []
        self.lower_order_nums = 0
        
    def _get_betas(self) -> torch.Tensor:
        """生成 beta schedule"""
        if self.config.beta_schedule == "linear":
            betas = torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.num_train_timesteps,
                dtype=torch.float32
            )
        elif self.config.beta_schedule == "scaled_linear":
            # 这是 Stable Diffusion 使用的默认 schedule
            betas = torch.linspace(
                self.config.beta_start ** 0.5,
                self.config.beta_end ** 0.5,
                self.num_train_timesteps,
                dtype=torch.float32
            ) ** 2
        else:
            raise ValueError(f"Unknown beta_schedule: {self.config.beta_schedule}")
        
        return betas
    
    def _compute_lambda(self) -> torch.Tensor:
        """
        计算 lambda_t = log(alpha_t / sigma_t)
        这是 DPM-Solver 的核心变换
        """
        alphas_cumprod = self.alphas_cumprod
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        lambda_t = torch.log(alphas_cumprod) - torch.log(1 - alphas_cumprod)
        return lambda_t
    
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = "cpu"):
        """设置推理步数和时间步"""
        self.num_inference_steps = num_inference_steps
        
        # 使用线性间隔的时间步
        timesteps = np.linspace(
            0, 
            self.num_train_timesteps - 1, 
            num_inference_steps,
            dtype=np.float32
        )[::-1].copy()
        
        self.timesteps = torch.from_numpy(timesteps).to(device)
        
        # 重置状态
        self.model_outputs = []
        self.lower_order_nums = 0
        
    def _get_alpha_and_sigma(self, timestep: torch.Tensor) -> tuple:
        """获取给定时间步的 alpha 和 sigma"""
        # 使用线性插值获取连续时间步的值
        timestep = timestep.clamp(0, self.num_train_timesteps - 1)
        
        low_idx = timestep.long()
        high_idx = torch.ceil(timestep).long()
        frac = timestep - low_idx.float()
        
        alpha_cumprod = self.alphas_cumprod[low_idx] * (1 - frac) + \
                        self.alphas_cumprod[high_idx] * frac
        
        alpha = alpha_cumprod ** 0.5
        sigma = (1 - alpha_cumprod) ** 0.5
        
        return alpha, sigma
    
    def _get_lambda(self, timestep: torch.Tensor) -> torch.Tensor:
        """获取给定时间步的 lambda 值"""
        timestep = timestep.clamp(0, self.num_train_timesteps - 1)
        
        low_idx = timestep.long()
        high_idx = torch.ceil(timestep).long()
        frac = timestep - low_idx.float()
        
        lambda_val = self.lambda_t[low_idx] * (1 - frac) + \
                     self.lambda_t[high_idx] * frac
        
        return lambda_val
    
    def _convert_model_output(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor
    ) -> torch.Tensor:
        """
        将模型输出转换为 x_0 预测
        
        DPM-Solver++ 在 data prediction (x_0) 空间工作
        """
        alpha_t, sigma_t = self._get_alpha_and_sigma(timestep)
        
        # 确保维度匹配
        while alpha_t.dim() < sample.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)
        
        if self.config.prediction_type == "epsilon":
            # x_0 = (x_t - sigma_t * epsilon) / alpha_t
            x0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.config.prediction_type == "v_prediction":
            # x_0 = alpha_t * x_t - sigma_t * v
            x0_pred = alpha_t * sample - sigma_t * model_output
        else:
            # 假设直接预测 x_0
            x0_pred = model_output
        
        # 应用阈值处理
        if self.config.thresholding:
            x0_pred = self._threshold_sample(x0_pred)
        
        return x0_pred
    
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """动态阈值处理"""
        if self.config.dynamic_thresholding_ratio < 1.0:
            # 计算动态阈值
            abs_sample = sample.abs()
            s = torch.quantile(
                abs_sample.reshape(abs_sample.shape[0], -1),
                self.config.dynamic_thresholding_ratio,
                dim=1
            )
            s = torch.clamp(s, min=1.0, max=self.config.sample_max_value)
            
            # 应用阈值
            while s.dim() < sample.dim():
                s = s.unsqueeze(-1)
            sample = torch.clamp(sample, -s, s) / s
        else:
            sample = torch.clamp(
                sample,
                -self.config.sample_max_value,
                self.config.sample_max_value
            )
        
        return sample
    
    def _dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        prev_timestep: torch.Tensor,
        sample: torch.Tensor
    ) -> torch.Tensor:
        """一阶 DPM-Solver 更新"""
        lambda_t = self._get_lambda(timestep)
        lambda_s = self._get_lambda(prev_timestep)
        
        alpha_t, sigma_t = self._get_alpha_and_sigma(timestep)
        alpha_s, sigma_s = self._get_alpha_and_sigma(prev_timestep)
        
        h = lambda_s - lambda_t
        
        # 确保维度匹配
        while alpha_t.dim() < sample.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)
            alpha_s = alpha_s.unsqueeze(-1)
            sigma_s = sigma_s.unsqueeze(-1)
        
        if self.config.algorithm_type == "dpmsolver++":
            # DPM-Solver++ 在 data prediction 空间
            x_t = sample
            x0_pred = model_output
            x_s = (alpha_s / alpha_t) * x_t - sigma_s * (torch.exp(h) - 1.0) * x0_pred
        else:
            # 标准 DPM-Solver
            x_t = sample
            x0_pred = model_output
            x_s = (
                (sigma_s / sigma_t) * x_t
                - alpha_s * (torch.exp(-h) - 1.0) * x0_pred
            )
        
        return x_s
    
    def _dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.Tensor],
        timestep_list: List[torch.Tensor],
        prev_timestep: torch.Tensor,
        sample: torch.Tensor
    ) -> torch.Tensor:
        """二阶 DPM-Solver 更新"""
        t = timestep_list[-1]
        s0 = timestep_list[-2]
        s = prev_timestep
        
        m0 = model_output_list[-1]
        m1 = model_output_list[-2]
        
        lambda_t = self._get_lambda(t)
        lambda_s0 = self._get_lambda(s0)
        lambda_s = self._get_lambda(s)
        
        alpha_t, sigma_t = self._get_alpha_and_sigma(t)
        alpha_s, sigma_s = self._get_alpha_and_sigma(s)
        
        h = lambda_s - lambda_t
        h_0 = lambda_t - lambda_s0
        r0 = h_0 / h
        
        # 确保维度匹配
        while alpha_t.dim() < sample.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)
            alpha_s = alpha_s.unsqueeze(-1)
            sigma_s = sigma_s.unsqueeze(-1)
        
        if self.config.algorithm_type == "dpmsolver++":
            if self.config.solver_type == "midpoint":
                x_t = sample
                D0 = m0
                D1 = (1.0 / r0) * (m0 - m1)
                
                x_s = (
                    (alpha_s / alpha_t) * x_t
                    - sigma_s * (torch.exp(h) - 1.0) * D0
                    - 0.5 * sigma_s * (torch.exp(h) - 1.0) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = sample
                x_s = (
                    (alpha_s / alpha_t) * x_t
                    - sigma_s * (torch.exp(h) - 1.0) * m0
                    - sigma_s * ((torch.exp(h) - 1.0) / h + 1.0) * (m0 - m1) / r0
                )
        else:
            # 标准 DPM-Solver 二阶
            x_t = sample
            D0 = m0
            D1 = (1.0 / r0) * (m0 - m1)
            
            x_s = (
                (sigma_s / sigma_t) * x_t
                - alpha_s * (torch.exp(-h) - 1.0) * D0
                - 0.5 * alpha_s * (torch.exp(-h) - 1.0) * D1
            )
        
        return x_s
    
    def _dpm_solver_third_order_update(
        self,
        model_output_list: List[torch.Tensor],
        timestep_list: List[torch.Tensor],
        prev_timestep: torch.Tensor,
        sample: torch.Tensor
    ) -> torch.Tensor:
        """三阶 DPM-Solver 更新"""
        t = timestep_list[-1]
        s0 = timestep_list[-2]
        s1 = timestep_list[-3]
        s = prev_timestep
        
        m0 = model_output_list[-1]
        m1 = model_output_list[-2]
        m2 = model_output_list[-3]
        
        lambda_t = self._get_lambda(t)
        lambda_s0 = self._get_lambda(s0)
        lambda_s1 = self._get_lambda(s1)
        lambda_s = self._get_lambda(s)
        
        alpha_t, sigma_t = self._get_alpha_and_sigma(t)
        alpha_s, sigma_s = self._get_alpha_and_sigma(s)
        
        h = lambda_s - lambda_t
        h_0 = lambda_t - lambda_s0
        h_1 = lambda_s0 - lambda_s1
        
        r0 = h_0 / h
        r1 = h_1 / h
        
        # 确保维度匹配
        while alpha_t.dim() < sample.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)
            alpha_s = alpha_s.unsqueeze(-1)
            sigma_s = sigma_s.unsqueeze(-1)
        
        D0 = m0
        D1 = (1.0 / r0) * (m0 - m1)
        D2 = (1.0 / (r0 * r1)) * (m0 - 2 * m1 + m2)
        
        if self.config.algorithm_type == "dpmsolver++":
            x_t = sample
            x_s = (
                (alpha_s / alpha_t) * x_t
                - sigma_s * (torch.exp(h) - 1.0) * D0
                - sigma_s * ((torch.exp(h) - 1.0) / h - 1.0) * D1
                - sigma_s * ((torch.exp(h) - 1.0 - h) / h ** 2 - 0.5) * D2
            )
        else:
            x_t = sample
            x_s = (
                (sigma_s / sigma_t) * x_t
                - alpha_s * (torch.exp(-h) - 1.0) * D0
                - alpha_s * ((torch.exp(-h) - 1.0) / h + 1.0) * D1
                - alpha_s * ((torch.exp(-h) - 1.0 + h) / h ** 2 - 0.5) * D2
            )
        
        return x_s
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True
    ):
        """
        执行一步去噪
        
        Args:
            model_output: UNet 的输出
            timestep: 当前时间步
            sample: 当前的噪声样本
            return_dict: 是否返回字典
            
        Returns:
            去噪后的样本
        """
        step_index = (self.timesteps == timestep).nonzero().item()
        
        # 转换模型输出为 x_0 预测
        x0_pred = self._convert_model_output(model_output, timestep, sample)
        
        # 保存模型输出
        self.model_outputs.append(x0_pred)
        self.timestep_list.append(timestep)
        
        # 获取前一个时间步
        if step_index == len(self.timesteps) - 1:
            prev_timestep = torch.tensor(0.0, device=timestep.device)
        else:
            prev_timestep = self.timesteps[step_index + 1]
        
        # 根据阶数选择更新方法
        lower_order_final = (
            self.config.lower_order_final and 
            step_index == len(self.timesteps) - 1
        )
        lower_order_second = (
            step_index == len(self.timesteps) - 2 and 
            self.config.lower_order_final
        )
        
        if len(self.model_outputs) == 1 or lower_order_final:
            # 一阶更新
            prev_sample = self._dpm_solver_first_order_update(
                x0_pred, timestep, prev_timestep, sample
            )
        elif len(self.model_outputs) == 2 or lower_order_second or self.solver_order == 2:
            # 二阶更新
            prev_sample = self._dpm_solver_second_order_update(
                self.model_outputs, self.timestep_list, prev_timestep, sample
            )
        else:
            # 三阶更新
            prev_sample = self._dpm_solver_third_order_update(
                self.model_outputs, self.timestep_list, prev_timestep, sample
            )
        
        # 只保留最近的 solver_order 个输出
        if len(self.model_outputs) > self.solver_order:
            self.model_outputs.pop(0)
            self.timestep_list.pop(0)
        
        if return_dict:
            return {"prev_sample": prev_sample, "pred_original_sample": x0_pred}
        else:
            return prev_sample
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """添加噪声（用于训练）"""
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        
        while sqrt_alpha_prod.dim() < original_samples.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples


