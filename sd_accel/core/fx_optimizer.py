# sd_accel/core/fx_optimizer.py
import torch
import torch.fx as fx
from torch.fx import GraphModule, Node
from typing import Dict, Any, Tuple, Optional, Callable, List
import hashlib
import pickle
from dataclasses import dataclass
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@dataclass
class ShapeCacheKey:
    """形状缓存的键"""
    input_shapes: Tuple[Tuple[int, ...], ...]
    dtypes: Tuple[torch.dtype, ...]
    device: str
    
    def __hash__(self):
        return hash((self.input_shapes, self.dtypes, self.device))
    
    def __eq__(self, other):
        if not isinstance(other, ShapeCacheKey):
            return False
        return (
            self.input_shapes == other.input_shapes and
            self.dtypes == other.dtypes and
            self.device == other.device
        )
    
    @classmethod
    def from_tensors(cls, *tensors: torch.Tensor) -> "ShapeCacheKey":
        """从张量创建缓存键"""
        input_shapes = tuple(tuple(t.shape) for t in tensors)
        dtypes = tuple(t.dtype for t in tensors)
        device = str(tensors[0].device) if tensors else "cpu"
        return cls(input_shapes, dtypes, device)


class FXGraphCache:
    """FX 图的形状感知缓存"""
    
    def __init__(self, max_cache_size: int = 16):
        self.cache: Dict[ShapeCacheKey, GraphModule] = {}
        self.max_cache_size = max_cache_size
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: ShapeCacheKey) -> Optional[GraphModule]:
        """获取缓存的图"""
        if key in self.cache:
            self.hit_count += 1
            logger.info(f"FX graph cache hit! (hits: {self.hit_count}, misses: {self.miss_count})")
            return self.cache[key]
        self.miss_count += 1
        logger.info(f"FX graph cache miss. (hits: {self.hit_count}, misses: {self.miss_count})")
        return None
    
    def put(self, key: ShapeCacheKey, graph_module: GraphModule):
        """存储优化后的图"""
        if len(self.cache) >= self.max_cache_size:
            # 简单的 FIFO 策略
            first_key = next(iter(self.cache))
            del self.cache[first_key]
            logger.info(f"Cache full, evicting oldest entry")
        
        self.cache[key] = graph_module
        logger.info(f"Cached optimized graph for key: {key}")
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0.0
        return {
            "cache_size": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
        }


class FXGraphOptimizer:
    """FX 图优化器，实现多种优化 passes"""
    
    def __init__(self, enable_fusion: bool = True, 
                 enable_constant_folding: bool = True,
                 enable_dce: bool = True):
        self.enable_fusion = enable_fusion
        self.enable_constant_folding = enable_constant_folding
        self.enable_dce = enable_dce
    
    def optimize(self, graph_module: GraphModule) -> GraphModule:
        """
        应用所有优化 passes
        
        Args:
            graph_module: 输入的 FX GraphModule
            
        Returns:
            优化后的 GraphModule
        """
        logger.info("Starting FX graph optimization...")
        
        # Pass 1: Pattern-based fusion
        if self.enable_fusion:
            logger.info("Applying pattern-based fusion...")
            graph_module = self.apply_fusion_patterns(graph_module)
        
        # Pass 2: Constant folding
        if self.enable_constant_folding:
            logger.info("Applying constant folding...")
            graph_module = self.constant_folding(graph_module)
        
        # Pass 3: Dead code elimination
        if self.enable_dce:
            logger.info("Applying dead code elimination...")
            graph_module = self.dead_code_elimination(graph_module)
        
        # 重新编译图
        graph_module.recompile()
        logger.info("FX graph optimization completed.")
        
        return graph_module
    
    def apply_fusion_patterns(self, graph_module: GraphModule) -> GraphModule:
        """
        应用模式匹配的算子融合
        
        常见模式:
        - Conv -> BiasAdd -> SiLU
        - Linear -> BiasAdd -> GELU
        - LayerNorm -> Dropout
        """
        graph = graph_module.graph
        
        # Pattern 1: Conv/Linear -> Add -> SiLU
        self._fuse_conv_bias_silu(graph)
        
        # Pattern 2: Linear -> Add -> GELU
        self._fuse_linear_bias_gelu(graph)
        
        # Pattern 3: Matmul -> Add (bias add)
        self._fuse_matmul_add(graph)
        
        graph_module.recompile()
        return graph_module
    
    def _fuse_conv_bias_silu(self, graph: fx.Graph):
        """融合 Conv -> Add -> SiLU 模式"""
        nodes = list(graph.nodes)
        
        for i in range(len(nodes) - 2):
            conv_node = nodes[i]
            add_node = nodes[i + 1]
            silu_node = nodes[i + 2]
            
            # 检查模式
            if (self._is_conv_node(conv_node) and
                self._is_add_node(add_node) and
                self._is_silu_node(silu_node) and
                add_node.args[0] == conv_node and
                silu_node.args[0] == add_node):
                
                logger.debug(f"Fusing Conv->Add->SiLU: {conv_node.name}, {add_node.name}, {silu_node.name}")
                
                # 创建融合节点
                with graph.inserting_after(silu_node):
                    fused_node = graph.call_function(
                        self._fused_conv_bias_silu,
                        args=(conv_node.args[0], conv_node.args[1], add_node.args[1]),
                        kwargs=conv_node.kwargs
                    )
                    fused_node.meta = silu_node.meta.copy()
                
                # 替换使用
                silu_node.replace_all_uses_with(fused_node)
                
                # 标记为待删除（在 DCE pass 中删除）
                add_node.users.clear()
                silu_node.users.clear()
    
    def _fuse_linear_bias_gelu(self, graph: fx.Graph):
        """融合 Linear -> Add -> GELU 模式"""
        nodes = list(graph.nodes)
        
        for i in range(len(nodes) - 2):
            linear_node = nodes[i]
            add_node = nodes[i + 1]
            gelu_node = nodes[i + 2]
            
            if (self._is_linear_node(linear_node) and
                self._is_add_node(add_node) and
                self._is_gelu_node(gelu_node) and
                add_node.args[0] == linear_node and
                gelu_node.args[0] == add_node):
                
                logger.debug(f"Fusing Linear->Add->GELU: {linear_node.name}, {add_node.name}, {gelu_node.name}")
                
                with graph.inserting_after(gelu_node):
                    fused_node = graph.call_function(
                        self._fused_linear_bias_gelu,
                        args=(linear_node.args[0], linear_node.args[1], add_node.args[1]),
                        kwargs=linear_node.kwargs
                    )
                    fused_node.meta = gelu_node.meta.copy()
                
                gelu_node.replace_all_uses_with(fused_node)
                add_node.users.clear()
                gelu_node.users.clear()
    
    def _fuse_matmul_add(self, graph: fx.Graph):
        """融合 Matmul -> Add (bias add) 模式"""
        nodes = list(graph.nodes)
        
        for i in range(len(nodes) - 1):
            matmul_node = nodes[i]
            add_node = nodes[i + 1]
            
            if (self._is_matmul_node(matmul_node) and
                self._is_add_node(add_node) and
                add_node.args[0] == matmul_node):
                
                logger.debug(f"Fusing Matmul->Add: {matmul_node.name}, {add_node.name}")
                
                with graph.inserting_after(add_node):
                    fused_node = graph.call_function(
                        self._fused_matmul_bias,
                        args=(matmul_node.args[0], matmul_node.args[1], add_node.args[1]),
                    )
                    fused_node.meta = add_node.meta.copy()
                
                add_node.replace_all_uses_with(fused_node)
                add_node.users.clear()
    
    # 辅助函数：节点类型检查
    def _is_conv_node(self, node: Node) -> bool:
        return (node.op == 'call_function' and 
                node.target in [torch.nn.functional.conv2d, torch.conv2d]) or \
               (node.op == 'call_module' and 
                isinstance(node.target, (torch.nn.Conv2d,)))
    
    def _is_linear_node(self, node: Node) -> bool:
        return (node.op == 'call_function' and 
                node.target in [torch.nn.functional.linear, torch.matmul]) or \
               (node.op == 'call_module' and 
                isinstance(node.target, (torch.nn.Linear,)))
    
    def _is_matmul_node(self, node: Node) -> bool:
        return node.op == 'call_function' and \
               node.target in [torch.matmul, torch.bmm, torch.mm]
    
    def _is_add_node(self, node: Node) -> bool:
        return (node.op == 'call_function' and 
                node.target in [torch.add, torch.Tensor.add]) or \
               (node.op == 'call_method' and node.target == 'add')
    
    def _is_silu_node(self, node: Node) -> bool:
        return node.op == 'call_function' and \
               node.target in [torch.nn.functional.silu, torch.silu]
    
    def _is_gelu_node(self, node: Node) -> bool:
        return node.op == 'call_function' and \
               node.target in [torch.nn.functional.gelu, torch.gelu]
    
    # 融合操作的实现
    @staticmethod
    def _fused_conv_bias_silu(input, weight, bias, **kwargs):
        """融合的 Conv + Bias + SiLU"""
        x = torch.nn.functional.conv2d(input, weight, bias=None, **kwargs)
        x = x + bias.view(1, -1, 1, 1)
        x = torch.nn.functional.silu(x)
        return x
    
    @staticmethod
    def _fused_linear_bias_gelu(input, weight, bias, **kwargs):
        """融合的 Linear + Bias + GELU"""
        x = torch.nn.functional.linear(input, weight, bias=bias)
        x = torch.nn.functional.gelu(x)
        return x
    
    @staticmethod
    def _fused_matmul_bias(input, weight, bias):
        """融合的 Matmul + Bias"""
        x = torch.matmul(input, weight)
        x = x + bias
        return x
    
    def constant_folding(self, graph_module: GraphModule) -> GraphModule:
        """
        常量折叠：预计算只依赖常量的子图
        """
        graph = graph_module.graph
        folded_count = 0
        
        for node in graph.nodes:
            if node.op == 'call_function' and self._can_fold_node(node, graph_module):
                try:
                    # 计算常量值
                    const_value = self._evaluate_node(node, graph_module)
                    
                    # 替换为常量
                    with graph.inserting_after(node):
                        const_node = graph.get_attr(f"_folded_const_{folded_count}")
                        # 将常量存储为模块属性
                        setattr(graph_module, f"_folded_const_{folded_count}", const_value)
                    
                    node.replace_all_uses_with(const_node)
                    folded_count += 1
                    logger.debug(f"Folded constant node: {node.name}")
                    
                except Exception as e:
                    logger.debug(f"Could not fold node {node.name}: {e}")
                    continue
        
        if folded_count > 0:
            logger.info(f"Folded {folded_count} constant nodes")
        
        graph_module.recompile()
        return graph_module
    
    def _can_fold_node(self, node: Node, graph_module: GraphModule) -> bool:
        """检查节点是否可以被折叠"""
        # 只折叠纯函数调用
        if node.op != 'call_function':
            return False
        
        # 检查所有输入是否都是常量
        for arg in node.args:
            if isinstance(arg, Node):
                if not self._is_constant_node(arg, graph_module):
                    return False
        
        for kwarg in node.kwargs.values():
            if isinstance(kwarg, Node):
                if not self._is_constant_node(kwarg, graph_module):
                    return False
        
        return True
    
    def _is_constant_node(self, node: Node, graph_module: GraphModule) -> bool:
        """检查节点是否是常量"""
        return node.op in ['get_attr', 'placeholder'] or \
               (hasattr(node, '_is_constant') and node._is_constant)
    
    def _evaluate_node(self, node: Node, graph_module: GraphModule) -> Any:
        """评估节点的常量值"""
        # 这里需要实际执行节点来获取常量值
        # 简化实现，实际中需要更复杂的求值逻辑
        args = []
        for arg in node.args:
            if isinstance(arg, Node):
                args.append(self._get_node_value(arg, graph_module))
            else:
                args.append(arg)
        
        kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, Node):
                kwargs[k] = self._get_node_value(v, graph_module)
            else:
                kwargs[k] = v
        
        return node.target(*args, **kwargs)
    
    def _get_node_value(self, node: Node, graph_module: GraphModule) -> Any:
        """获取节点的值"""
        if node.op == 'get_attr':
            return getattr(graph_module, node.target)
        elif node.op == 'placeholder':
            raise ValueError("Cannot evaluate placeholder node")
        else:
            return self._evaluate_node(node, graph_module)
    
    def dead_code_elimination(self, graph_module: GraphModule) -> GraphModule:
        """
        死代码消除：删除不影响输出的节点
        """
        graph = graph_module.graph
        eliminated_count = 0
        
        # 标记所有可达节点
        reachable = set()
        
        def mark_reachable(node: Node):
            if node in reachable:
                return
            reachable.add(node)
            for inp in node.all_input_nodes:
                mark_reachable(inp)
        
        # 从输出节点开始标记
        for node in graph.nodes:
            if node.op == 'output':
                for inp in node.all_input_nodes:
                    mark_reachable(inp)
        
        # 删除不可达节点
        nodes_to_remove = []
        for node in graph.nodes:
            if node not in reachable and node.op not in ['placeholder', 'output']:
                nodes_to_remove.append(node)
                eliminated_count += 1
        
        for node in nodes_to_remove:
            graph.erase_node(node)
            logger.debug(f"Eliminated dead node: {node.name}")
        
        if eliminated_count > 0:
            logger.info(f"Eliminated {eliminated_count} dead nodes")
        
        graph_module.recompile()
        return graph_module


class FXUNetOptimizer:
    """
    UNet 的 FX 优化器，集成图捕获、优化和缓存
    """
    
    def __init__(self, 
                 enable_fusion: bool = True,
                 enable_constant_folding: bool = True,
                 enable_dce: bool = True,
                 cache_size: int = 16):
        self.graph_optimizer = FXGraphOptimizer(
            enable_fusion=enable_fusion,
            enable_constant_folding=enable_constant_folding,
            enable_dce=enable_dce
        )
        self.graph_cache = FXGraphCache(max_cache_size=cache_size)
        self.original_forward = None
    
    def optimize_unet(self, unet: torch.nn.Module) -> torch.nn.Module:
        """
        优化 UNet 模块
        
        Args:
            unet: 原始 UNet 模块
            
        Returns:
            优化后的 UNet 模块
        """
        # 保存原始 forward
        self.original_forward = unet.forward
        
        # 创建优化的 forward 方法
        def optimized_forward(sample, timestep, encoder_hidden_states, **kwargs):
            # 生成缓存键
            cache_key = ShapeCacheKey.from_tensors(
                sample, timestep, encoder_hidden_states
            )
            
            # 检查缓存
            cached_graph = self.graph_cache.get(cache_key)
            
            if cached_graph is not None:
                # 使用缓存的图
                return cached_graph(sample, timestep, encoder_hidden_states, **kwargs)
            
            # 缓存未命中，需要捕获和优化图
            logger.info("Capturing and optimizing UNet graph...")
            
            try:
                # 使用 FX 捕获图
                traced_graph = fx.symbolic_trace(
                    lambda s, t, e: self.original_forward(s, t, e, **kwargs)
                )
                
                # 应用优化 passes
                optimized_graph = self.graph_optimizer.optimize(traced_graph)
                
                # 缓存优化后的图
                self.graph_cache.put(cache_key, optimized_graph)
                
                # 执行优化后的图
                return optimized_graph(sample, timestep, encoder_hidden_states)
                
            except Exception as e:
                logger.warning(f"FX optimization failed: {e}, falling back to original forward")
                return self.original_forward(sample, timestep, encoder_hidden_states, **kwargs)
        
        # 替换 forward 方法
        unet.forward = optimized_forward
        
        return unet
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self.graph_cache.stats()
    
    def clear_cache(self):
        """清空缓存"""
        self.graph_cache.clear()





