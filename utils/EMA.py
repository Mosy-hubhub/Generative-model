class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}  # 存放平滑后的 EMA 权重
        self.backup = {}  # 临时存放当前权重的备份
        self.register()

    def register(self):
        """初始化时，将当前模型的参数克隆到 shadow 中"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """在每次 optimizer.step() 之后调用此函数，更新 EMA 权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        """测试/保存模型前调用：把模型的当前权重替换为 EMA 权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """测试/保存完成后调用：把真正的训练权重还给模型，继续训练"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}