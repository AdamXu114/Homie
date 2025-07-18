import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GlobalSwitch():
    def __init__(self) -> None:
        self.switch_flag = False
        self.count = 0
        self.hybrid_reward_scales = None
        self.pretrained_reward_scales = None
        
        self.pretrained_to_hybrid_start = 20
        self.pretrained_to_hybrid_end = self.pretrained_to_hybrid_start + 20


    # 初始化sigmoid学习率
    def init_sigmoid_lr(self):
        # 计算预训练到混合训练的范围长度
        range_len = self.pretrained_to_hybrid_end - self.pretrained_to_hybrid_start
        # 生成从-7到7的线性空间，用于sigmoid函数的输入
        divide = np.linspace(-7, 7, range_len)
        # 计算1减去sigmoid函数的值，得到学习率下降的曲线
        self.lr_down = 1-sigmoid(divide)


    # 初始化线性学习率
    def init_linear_lr(self):
        # 计算预训练到混合阶段的范围长度
        range_len = self.pretrained_to_hybrid_end - self.pretrained_to_hybrid_start
        # 生成从1到0的线性递减学习率数组
        self.lr_down = np.linspace(1, 0, range_len)


    def set_reward_scales(self, hybrid_reward_scales, pretrained_reward_scales):
        self.hybrid_reward_scales = hybrid_reward_scales
        self.pretrained_reward_scales = pretrained_reward_scales
        print("Hybrid reward scales:", self.hybrid_reward_scales)
        print("Pretrained reward scales:", self.pretrained_reward_scales)

    # 获取奖励缩放因子的方法
    def get_reward_scales(self):
        # 如果当前计数小于预训练到混合模式的开始计数
        if self.count < self.pretrained_to_hybrid_start:
            # 返回预训练模式的奖励缩放因子
            return self.pretrained_reward_scales

        # 如果当前计数在预训练到混合模式的开始和结束计数之间
        elif self.count < self.pretrained_to_hybrid_end:
            # 初始化奖励缩放因子字典
            reward_scales = {}
            # 获取当前的学习率
            lr = self.lr_down[self.count - self.pretrained_to_hybrid_start]
            # 遍历混合模式的奖励缩放因子
            for key, end in self.hybrid_reward_scales.items():
                # 获取预训练模式的奖励缩放因子
                start = self.pretrained_reward_scales[key]
                # reward_scales[key] = start + (end - start) * (self.count - self.pretrained_to_hybrid_start) / (self.pretrained_to_hybrid_end - self.pretrained_to_hybrid_start)
                reward_scales[key] = start * lr + end * (1 - lr)

            # 返回当前的奖励缩放因子
            return reward_scales

        # 如果当前计数大于或等于预训练到混合模式的结束计数
        else:
            # 返回混合模式的奖励缩放因子
            return self.hybrid_reward_scales


    # def get_reward_scales(self):
    #     if self.count < self.pretrained_to_hybrid_start:
    #         return self.pretrained_reward_scales
        
    #     elif self.count < self.pretrained_to_hybrid_end:
    #         reward_scales = {}
    #         for key, end in self.hybrid_reward_scales.items():
    #             start = self.pretrained_reward_scales[key]
    #             reward_scales[key] = start + (end - start) * (self.count - self.pretrained_to_hybrid_start) / (self.pretrained_to_hybrid_end - self.pretrained_to_hybrid_start)
    #         return reward_scales
        
    #     else:
    #         return self.hybrid_reward_scales

    # 获取beta值的方法
    def get_beta(self):
        # 如果当前计数小于等于预训练到混合模式的开始计数
        if self.count <= self.pretrained_to_hybrid_start:
            # 返回0.0，表示完全使用预训练模式
            return 0.0

        # 如果当前计数在预训练到混合模式的开始和结束计数之间
        elif self.count < self.pretrained_to_hybrid_end:
            # 计算并返回一个线性插值的beta值
            return 0.5 * (self.count - self.pretrained_to_hybrid_start) / (self.pretrained_to_hybrid_end - self.pretrained_to_hybrid_start)

        # 如果当前计数大于等于预训练到混合模式的结束计数
        else:
            # 返回0.5，表示完全使用混合模式
            return 0.5


    def open_switch(self):
        self.switch_flag = True
    
    @property    
    def switch_open(self):
        return self.switch_flag

    def close_switch(self):
        self.switch_flag = False
    
global_switch = GlobalSwitch()