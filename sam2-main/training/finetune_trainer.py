# In training/finetune_trainer.py

import logging
import torch

from .trainer import Trainer
from training.optimizer import construct_optimizer


class SAMFinetuneTrainer(Trainer):
    """
    一个专门用于 SAM2 微调的 Trainer。
    它继承自通用的 Trainer 类，并重写了优化器的构建方法，
    以便冻结模型的一部分。
    """

    def _construct_optimizers(self):
        """
        重写此方法以冻结特定层。
        PyTorch的优化器会自动忽略 requires_grad=False 的参数。
        """
        logging.info(">>>>> Running SAMFinetuneTrainer: Freezing model parts for fine-tuning. <<<<<")

        # 使用一个健壮的方式来获取未被DDP包装的模型
        model_to_modify = self.model.module if hasattr(self.model, 'module') else self.model

        # 冻结和解冻逻辑
        logging.info("Freezing Image Encoder...")
        for param in model_to_modify.image_encoder.parameters():
            param.requires_grad = False

        logging.info("Enabling training for Prompt Encoder...")
        for param in model_to_modify.sam_prompt_encoder.parameters():
            param.requires_grad = True

        logging.info("Enabling training for Mask Decoder...")
        for param in model_to_modify.sam_mask_decoder.parameters():
            param.requires_grad = True

        # 打印验证
        num_trainable = sum(p.numel() for p in model_to_modify.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model_to_modify.parameters())
        logging.info(
            f"Fine-tuning with {num_trainable:,} trainable parameters out of {num_total:,} total parameters."
        )

        # =================================================================
        # <<< 最终修正 >>>
        # 将完整的模型对象 self.model 传递给优化器构造函数，而不是一个参数列表。
        # 这是因为 construct_optimizer 需要检查参数名。
        self.optim = construct_optimizer(
            self.model,  # <<< 修正这里，传递整个模型
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
        )
        # =================================================================

        logging.info(">>>>> SAMFinetuneTrainer: Optimizer constructed for fine-tuning. <<<<<")