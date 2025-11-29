"""
本地轻量级大模型封装
支持Qwen和ChatGLM系列模型
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalLLM:
    """本地轻量级大模型封装类"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen-7B-Chat",
        device: str = "cuda",
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        quantization_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化本地大模型

        Args:
            model_name: 模型名称或路径
            device: 运行设备 (cuda/cpu)
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: nucleus采样参数
            quantization_config: 量化配置
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {self.device}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # 配置量化
        load_kwargs = {"trust_remote_code": True}
        if quantization_config and quantization_config.get("enabled", False):
            logger.info(f"Using {quantization_config.get('bits', 4)}-bit quantization")
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = torch.float16

            # 4-bit量化配置
            if quantization_config.get("bits") == 4:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
        else:
            load_kwargs["device_map"] = self.device

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        self.model.eval()
        logger.info("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        生成文本

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            **kwargs: 其他生成参数

        Returns:
            生成的文本
        """
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 生成参数
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_length,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # 解码
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # 移除输入部分
        if prompt in response:
            response = response[len(prompt):].strip()

        return response

    def chat(
        self,
        query: str,
        history: Optional[list] = None,
        system: Optional[str] = None
    ) -> tuple:
        """
        对话接口（适配不同模型的对话格式）

        Args:
            query: 用户查询
            history: 对话历史
            system: 系统提示

        Returns:
            (response, updated_history)
        """
        history = history or []

        # 根据模型类型构建提示
        if "Qwen" in self.model_name:
            response, history = self._chat_qwen(query, history, system)
        elif "chatglm" in self.model_name.lower():
            response, history = self._chat_chatglm(query, history)
        else:
            # 通用格式
            response, history = self._chat_generic(query, history, system)

        return response, history

    def _chat_qwen(self, query: str, history: list, system: Optional[str]) -> tuple:
        """Qwen模型对话"""
        # 使用Qwen的chat方法
        response, history = self.model.chat(
            self.tokenizer,
            query,
            history=history,
            system=system
        )
        return response, history

    def _chat_chatglm(self, query: str, history: list) -> tuple:
        """ChatGLM模型对话"""
        # 使用ChatGLM的chat方法
        response, history = self.model.chat(
            self.tokenizer,
            query,
            history=history
        )
        return response, history

    def _chat_generic(self, query: str, history: list, system: Optional[str]) -> tuple:
        """通用对话格式"""
        # 构建对话提示
        prompt = ""
        if system:
            prompt += f"System: {system}\n\n"

        for user_msg, assistant_msg in history:
            prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"

        prompt += f"User: {query}\nAssistant:"

        response = self.generate(prompt)
        history.append((query, response))

        return response, history

    def __call__(self, prompt: str, **kwargs) -> str:
        """便捷调用接口"""
        return self.generate(prompt, **kwargs)
