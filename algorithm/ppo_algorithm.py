import copy
import os
import random
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
from torch.distributions import Categorical
import torch.nn.functional as F

from instance.generate_allsize_mofjsp_dataset import Logger, Result_Logger
from model.actor_critic import Operation_Actor_JointAction_selfCritic, Machine_Actor_JointAction_selfGAT_selfCritic,Global_Critic_JointAction_GAT, esa_Operation_Actor_Critic, esa_Machine_Actor, ablation_Machine_Actor_JointAction_selfGAT_selfCritic
from trainer.fig_kpi import get_GPU_usage
from model.gcn_mlp import Encoder, aggr_obs, g_pool_cal

from trainer.train_device import device

"""
当调用这整个py文件时：
1、除非用了if __name__ == '__main__': 不会被执行代码
2、不然，所有的代码都会被执行的，包括变量（在其他py中只有import了该变量，才能直接使用）会被用在class的变量中，赋值会改变
`torch.device` 类本身并不直接指定在哪个 GPU 上运行，它只用于指定设备类型（GPU 或 CPU）
"""
model_path = ('./model/ppo_actor.pth', './model/ppo_critic.pth')


class PPOAlgorithm(object):
    def __init__(self, args, load_pretrained):
        # 基础调度参数提取
        self.n_job = args['n_job']
        self.n_machine = args['n_machine']
        self.n_total_task = self.n_job * self.n_machine
        self.batch_size = args['env_batch']
        # PPO 的核心参数
        self.GAMMA = args['GAMMA'] # reward折扣率
        self.LAMDA = args['LAMDA'] # GAE参数
        self.epsilon = args['epsilon']  # 重要性采样的裁剪
        self.ENTROPY_BETA = args['ENTROPY_BETA']  # 策略熵权重

        #核心网络架构初始化：实例化 actor_critic.py 中的三个核心模型，并为它们分别配备了优化器和学习率衰减。
        # Todo 1. Job Actor: 负责选工序
        self.job_actor = Operation_Actor_JointAction_selfCritic(configs=args).to(device)
        self.job_actor_optimizer = torch.optim.Adam(self.job_actor.parameters(),
                                                    lr=args['LR'],
                                                    eps=args['lr_eps'])  # 使用 Adam 优化器
        # 学习率的调度器，逐渐衰减！（每过step_size就乘以一个系数衰减）
        self.job_actor_lr_decay = torch.optim.lr_scheduler.StepLR(self.job_actor_optimizer,
                                                                  step_size=args['decay_step_size'],
                                                                  gamma=args['decay_ratio'])
        # Todo 2. Machine Actor: 负责选机器 (用了 GCN/GAT)
        self.machine_actor_gcn = Machine_Actor_JointAction_selfGAT_selfCritic(configs=args).to(device)
        self.machine_actor_optimizer_gcn = torch.optim.Adam(self.machine_actor_gcn.parameters(),
                                                            lr=args['LR'],
                                                            eps=args['lr_eps'])  # 修改adam优化器默认的eps，提高训练性能
        # 学习率的调度器，逐渐衰减！（每过step_size就乘以一个系数衰减）
        self.machine_actor_lr_decay_gcn = torch.optim.lr_scheduler.StepLR(self.machine_actor_optimizer_gcn, 
                                                                  step_size=args['decay_step_size'],
                                                                  gamma=args['decay_ratio'])

        # Todo 3. Global Critic: 全局评价 (给 Actor 打分)
        self.global_critic = Global_Critic_JointAction_GAT(configs=args).to(device)
        self.global_critic_optimizer = torch.optim.Adam(self.global_critic.parameters(),
                                                        lr=args['LR'],
                                                        eps=args['lr_eps'])  # 修改adam优化器默认的eps，提高训练性能
        # 学习率的调度器，逐渐衰减！（每过step_size就乘以一个系数衰减）
        self.global_critic_lr_decay = torch.optim.lr_scheduler.StepLR(self.global_critic_optimizer,
                                                                      step_size=args['decay_step_size'],
                                                                      gamma=args['decay_ratio'])
        
        Logger.log("Training/ppo/network_architecture", f"Operation_Actor: self.job_actor={self.job_actor}, Machine_Actor: self.machine_actor_gcn={self.machine_actor_gcn}, Global_Critic: self.global_critic={self.global_critic}", print_true=1)
        
        #初始化了 esa 版本的 Actor，用于后续做性能对比实验。
        self.esa_job_actor = esa_Operation_Actor_Critic(configs=args).to(device)
        self.esa_job_actor_optimizer = torch.optim.Adam(self.esa_job_actor.parameters(),
                                                    lr=args['LR'],
                                                    eps=args['lr_eps'])  # 修改adam优化器默认的eps，提高训练性能
        # 学习率的调度器，逐渐衰减！（每过step_size就乘以一个系数衰减）
        self.esa_job_actor_lr_decay = torch.optim.lr_scheduler.StepLR(self.esa_job_actor_optimizer, 
                                                                      step_size=args['decay_step_size'],
                                                                      gamma=args['decay_ratio'])

        self.esa_machine_actor_gcn = esa_Machine_Actor(configs=args).to(device)
        self.esa_machine_actor_optimizer_gcn = torch.optim.Adam(self.esa_machine_actor_gcn.parameters(),
                                                            lr=args['LR'],
                                                            eps=args['lr_eps'])  # 修改adam优化器默认的eps，提高训练性能
        # 学习率的调度器，逐渐衰减！（每过step_size就乘以一个系数衰减）
        self.esa_machine_actor_lr_decay_gcn = torch.optim.lr_scheduler.StepLR(self.esa_machine_actor_optimizer_gcn,
                                                                              step_size=args['decay_step_size'],
                                                                              gamma=args['decay_ratio'])
       
        Logger.log("Training/ppo/esa/network_architecture", f"Operation_Actor: self.esa_job_actor={self.esa_job_actor}, Machine_Actor: self.esa_machine_actor_gcn={self.esa_machine_actor_gcn}", print_true=1)

        self.global_critic_loss_func = torch.nn.MSELoss()  # 均方误差，公式写的是这个
        #初始化工序（Job）轨迹数据容器
        self.state_values, self.rewards, self.log_a, self.softmax_a, self.log_actions = [], [], [], [], []
        #初始化机器（Machine）轨迹数据容器
        self.m_state_values, self.m_rewards, self.m_log_a, self.m_softmax_a, self.m_log_actions = [], [], [], [], []
        # 初始化 Mask 字典：记录每个 Job 还能做几个工序
        self.remaining_m = {}

        # 调度记账本初始化
        #A. 剩余工序计数
        for i in range(self.n_job):
            self.remaining_m[i] = self.n_machine
        self.remaining_m_batch = []
        for _ in range(self.batch_size):
            remain_m = {}
            for j in range(self.n_job):
                remain_m[j] = self.n_machine
            self.remaining_m_batch.append(remain_m)

        # 记录每个 Job 当前待做的 Task ID
        self.pool_task_list = [1 + self.n_machine * i for i in range(self.n_job)]
        self.pool_task_dict = {}
        for i in range(self.n_job):
            self.pool_task_dict[i] = self.pool_task_list[i]
        self.pool_task_dict_batch = []
        for _ in range(self.batch_size):
            task_dict = {}
            for i in range(self.n_job):
                task_dict[i] = self.pool_task_list[i]
            self.pool_task_dict_batch.append(task_dict)
        # 动作掩码与结果收集
        lst = [0.0] * self.n_job
        self.mask_new = torch.tensor(lst).cuda()
        self.mask_new_batch = []
        for j in range(self.batch_size):
            self.mask_new_batch.append(lst)
        self.mask_new_batch = torch.tensor(self.mask_new_batch).cuda()
        self.chosen_taskID_list = []  # 最终选择的task的id
        self.chosen_taskID_list_batch = [[] for _ in range(self.batch_size)]

    def esa_update_chosenTaskID_CandidateTaskIDx_JobMask(self, paralenv, action_batch, mask_value):
        """
    Todo
        作用： “选定了一个工序” 之后，更新智能体内部的记账本（哪些工序做完了、下一个该做谁），并计算下一轮选择的掩码，防止智能体做出非法或不合理的动作(还有个加快收敛的强规则约束）。
        输入：paralenv: 并行环境对象（用于获取底层图的状态）。
        action_batch: 刚才神经网络选出的动作（Job Index，即选了哪一行的工件）。
        mask_value: 掩码值（通常是一个极小的负数，用于在 Logit 层屏蔽不可选的动作）。
        输出：
        candidate_batch: 更新后的候选工序列表(喂给 GNN 提取特征用）。
        mask_operation_batch: 下一步选择的 Mask（喂给 Actor 网络用于 Softmax 屏蔽）。
        """
        #第一部分：基础状态更新（记账）
        for i_batch in range(self.batch_size):
            # 1. 获取当前 Batch 选了哪个 Job (index_a)
            index_a = action_batch[i_batch].data.item()
            # 2. 减少剩余工序计数
            # self.remaining_m_batch 记录了每个 Job 还有几个工序没做
            if self.remaining_m_batch[i_batch][index_a] != 0:
                self.remaining_m_batch[i_batch][index_a] -= 1
            # 3. 记录被选中的 Task ID
            # pool_task_dict_batch 记录了当前 Job 待做的 Task ID (例如 Job 1 目前待做 Task 5)
            self.chosen_taskID_list_batch[i_batch].append(self.pool_task_dict_batch[i_batch][index_a])
            # 4. 更新候选池
            # 如果该 Job 还有工序没做，就把它的候选 Task ID + 1（指向下一个工序）
            # 例如：Job 1 刚做完 Task 5，那它的候选就变成 Task 6
            if self.remaining_m_batch[i_batch][index_a] != 0:  # 都减到0了，说明选完了没有剩余m，可选task的id就固定在同job的最后一个task，不再更新
                self.pool_task_dict_batch[i_batch][index_a] += 1
            # 5. 基础 Mask 更新
            # 如果某个 Job 的剩余工序数为 0，则在 mask_new_batch 中将其标记为不可选
            for key, value in self.remaining_m_batch[i_batch].items():
                if value == 0:
                    self.mask_new_batch[i_batch][key] = mask_value

        #为了让调度更紧凑，限制智能体：优先选择那些前置工序完工时间最早的 Job。
        #启发式策略，防止智能体在早期随机探索时选到那些由于前置工序还没做完、导致空窗期很长的工序。
        # 1. 初始化
        mask_operation_batch = self.mask_new_batch.bool()
        for i_bs in range(self.batch_size):
            # ... 初始化 mask 和 ft 列表 ...
            eachbs_mask = [0] * self.n_total_task
            eachbs_ft = [0] * self.n_total_task
            # 2. 从底层图环境获取完工时间
            for i_task in range(self.n_total_task): # 遍历所有的task
                # 检查图节点是否有 finish_time 属性 (说明该节点已被调度)
                if paralenv.paral_env_DG[i_bs].G.nodes[i_task+1]['finish_time'] is not None:
                    eachbs_mask[i_task] = 1  # 标记为已调度
                    eachbs_ft[i_task] = paralenv.paral_env_DG[i_bs].G.nodes[i_task+1]['finish_time']
            eachbs_mask_np = np.array(eachbs_mask).reshape(self.n_job, self.n_machine)
            # eachbs_ft_np: [Job, Machine] 矩阵，存的是各工序的完工时间
            eachbs_ft_np = np.array(eachbs_ft).reshape(self.n_job, self.n_machine)
            eachcol_mask_sum = np.sum(eachbs_mask_np,axis=0)

            # 3. 计算每一行(Job)的最大完工时间 -> 即该 Job "当前待做工序" 的 "前置工序完工时间"
            # 逻辑：每一行里，数值最大的那个肯定是刚做完的那个前置工序。
            max_values_in_each_row = [max(row) for row in eachbs_ft_np]
            # 4. 遍历每一列 (Machine维度，这里其实代表工序阶段)
            for i_col in range(self.n_machine): # 遍历每一列！
                if i_col != 0:  # 非第一道工序
                    # 判断逻辑：如果前一列都做完了，但当前列还没做完 -> 说明处于中间阶段
                    if eachcol_mask_sum[i_col-1] == self.n_job and eachcol_mask_sum[i_col] != self.n_job:
                        # a. 把已经做完的 Job 的 FT 设为无穷大 (inf)，排除干扰
                        for i in range(len(self.mask_new_batch[i_bs])):  # 遍历当前bs的所有job
                            if self.mask_new_batch[i_bs][i] == 1:  # 说明该job选完了
                                max_values_in_each_row[i] = float("inf")
                        # b. 找到最小的 FT (min_value)
                        min_value = min(max_values_in_each_row)
                        # c. 只有 FT 等于最小值的那些 Job 才是可选的！
                        min_indexes = [index for index, value in enumerate(max_values_in_each_row) if value == min_value]
                        # d. 生成 Mask：只有 min_indexes 里的 Job 是 False (可选)，其他全 True (屏蔽)
                        mask_temp = [1] * self.n_job
                        for iii in range(len(min_indexes)):
                            mask_temp[min_indexes[iii]] = 0
                        mask_operation_batch[i_bs] = torch.tensor(mask_temp).bool()
                elif i_col == 0: # 第一列的特殊情况
                    if eachcol_mask_sum[i_col] != self.n_job:  # 不等于job说明没选完
                        mask_operation_batch[i_bs] = torch.tensor(eachbs_mask_np[:, i_col]).bool()
                    else: # =job 说明第一列选完了，啥也不做，下一个判断
                        pass

        #第三部分：生成最终输出
        # candidate_batch: 告诉 Job Actor 每个 Job 对应的工序节点特征索引
        # mask_operation_batch: 告诉 Job Actor 哪些 Job 能选，哪些不能选
        value_list = []
        for dict in self.pool_task_dict_batch:
            value_list.append(list(dict.values()))
        candidate_batch = np.array(value_list) - 1
        return candidate_batch, mask_operation_batch

    def Eval_esa_update_chosenTaskID_CandidateTaskIDx_JobMask(self, env, action_batch, mask_value):
        """
    Todo
        之前讲的那个函数是训练（Training）时用的，负责同时维护Batch Size并行环境的状态。
        而这个函数是评估（Evaluation/Testing）时用的，通常一次只跑一个环境（Batch Size = 1），用于测试模型在单个特定算例上的表现。
        """
        index_a = action_batch.data.item()
        if self.remaining_m[index_a] != 0:
            self.remaining_m[index_a] -= 1
        self.chosen_taskID_list.append(self.pool_task_dict[index_a])
        if self.remaining_m[index_a] != 0:
            self.pool_task_dict[index_a] += 1
        for key, value in self.remaining_m.items():
            if value == 0:
                self.mask_new[key] = mask_value
        mask = self.mask_new.bool()
        eachbs_mask = [0] * self.n_total_task
        eachbs_ft = [0] * self.n_total_task
        for i_task in range(self.n_total_task):
            if env.G.nodes[i_task + 1]['finish_time'] is not None:
                eachbs_mask[i_task] = 1
                eachbs_ft[i_task] = env.G.nodes[i_task + 1]['finish_time']
        eachbs_mask_np = np.array(eachbs_mask).reshape(self.n_job,self.n_machine)
        eachbs_ft_np = np.array(eachbs_ft).reshape(self.n_job,self.n_machine)
        eachcol_mask_sum = np.sum(eachbs_mask_np, axis=0)
        max_values_in_each_row = [max(row) for row in eachbs_ft_np]
        for i_col in range(self.n_machine):  # 遍历每一列！
            if i_col != 0:  # 第一列，就是随机选！
                if eachcol_mask_sum[i_col - 1] == self.n_job and eachcol_mask_sum[i_col] != self.n_job:
                    for i in range(len(self.mask_new)):
                        if self.mask_new[i] == 1:  # 说明该job选完了
                            max_values_in_each_row[i] = float("inf")
                    # 找到所有最小值的索引
                    min_value = min(max_values_in_each_row)
                    min_indexes = [index for index, value in enumerate(max_values_in_each_row) if value == min_value]
                    mask_temp = [1] * self.n_job  # ESA：后续都是每一列只有min的才可以选择，所以初始1
                    for iii in range(len(min_indexes)):
                        mask_temp[min_indexes[iii]] = 0  # 0表示是可以选择的！！！！
                    mask = torch.tensor(mask_temp).bool()  # 转成tensor，然后直接01转bool shape = j, 只有一个env！
            elif i_col == 0:  # 第一列的特殊情况，不等于3说明没选完
                if eachcol_mask_sum[i_col] != self.n_job:
                    mask = torch.tensor(eachbs_mask_np[:, i_col]).bool()  # 第一列的01转成bool，被选过不能再选，从该列其他的开始选
                else:  # =job 说明第一列选完了，啥也不做，下一个判断
                    pass
        candidate = np.array(list(self.pool_task_dict.values())) - 1
        return candidate, mask


    def step_for_net_out_Critic_GAT(self, net_model, task_fea, graph_pool_avg, adj, candidate, machine_fea1, machine_fea2):
        """
    TODO
        解决“数据维度”和“网络输入”不匹配的问题，帮助 Global Critic 网络一口气把 ReplayBuffer 里存的所有历史数据的价值（Value）都算出来。
        """
        out_temp_lst = []
        for i in range(task_fea.shape[0]):
            v = net_model(task_fea[i],
                          graph_pool_avg,
                          adj[i],
                          candidate[i],
                          machine_fea1[i],
                          machine_fea2[i]
                          )
            out_temp_lst.append(v)
        out = torch.stack(out_temp_lst, dim=0)  # 沿着第一个维度堆叠，list[tensor_1, tensor_2, ..., tensor_32]变成一个大tensor
        return out
    

    def cal_local_job_machine_reward_GAE(self, mk_r, pt_r, tt_r, it_r, jv, jv_, mv, mv_, done_operation):
        """
    TODO
        作用：计算 “局部优势函数”。 它负责将 4 个独立的奖励指标（完工时间、能耗、运输、空闲）分配给两个不同的 Actor 网络
        并分别计算它们在各自负责领域的表现好坏（GAE）。
        输入：奖励分量，价值估计，结束标志
        输出：adv_lst: 一个包含 4 个 Tensor 的列表 [adv_mk, adv_pt, adv_tt, adv_it]。代表了每个指标在每一步的“优势值”（归一化后的）。
        """
        adv_lst = []
        differ_r_lst = [mk_r, pt_r, tt_r, it_r] # 真实的奖励列表 [完工时间, 能耗, 运输, 空闲]
        differ_v_lst = [jv[:, :, 0], mv[:, :, 0], mv[:, :, 1], jv[:, :, 1]]
        differ_v__lst = [jv_[:, :, 0], mv_[:, :, 0], mv_[:, :, 1], jv_[:, :, 1]]  # 下一时刻的value值
        #循环计算 GAE
        for i in range(4):  # 4个指标
            global_r = differ_r_lst[i]  # 选择具体指标  buffer_step * env_batch
            v = differ_v_lst[i]  # 选择具体指标    step*bs*1
            v_ = differ_v__lst[i]  # 选择具体指标  step*bs*1
            gae = 0
            adv = []
            # 计算 TD-Error
            deltas = global_r + self.GAMMA * v_.squeeze(-1) - v.squeeze(-1)  # TD-error的基本形式，加了batch
            # 倒序计算 GAE
            for delta, d in zip(reversed(deltas), reversed(done_operation)):
                gae = delta + self.GAMMA * self.LAMDA * gae * (1.0 - d)
                adv.insert(0, gae)
            # 归一化
            adv = torch.stack(adv)
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
            adv_lst.append(copy.deepcopy(adv))
        return adv_lst
    
    def separate_cal_4_reward_GAE(self, mk_r, pt_r, tt_r, it_r, v, v_, done_operation):
        """
    TODO
        作用：计算 “全局分离优势函数”。 它利用 Global Critic 输出的 4 维价值预测，分别针对 4 个不同的目标（完工时间、能耗、运输、空闲）计算 GAE。
        输入：奖励分量，价值估计，结束标志
        输出：adv_lst: 一个包含 4 个 Tensor 的列表 [adv_mk, adv_pt, adv_tt, adv_it]。
        """
        adv_lst = []
        differ_r_lst = [mk_r, pt_r, tt_r, it_r]
        for i in range(4):  # 4个指标
            global_r = differ_r_lst[i]  # 选择具体指标
            gae = 0
            adv = []
            deltas = global_r + self.GAMMA * v_[:, :, i:(i+1)].squeeze(-1) - v[:, :, i:(i+1)].squeeze(-1)
            for delta, d in zip(reversed(deltas), reversed(done_operation)):
                gae = delta + self.GAMMA * self.LAMDA * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.stack(adv)
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
            adv_lst.append(copy.deepcopy(adv))
        return adv_lst

    
    def global_update_JointActions_GAT_selfCritic(self, replay_buffer, all_steps, graph_pool_avg, args, mini_bs):
        """
        GCN网络
        """
        self.job_actor.train()
        self.machine_actor_gcn.train()
        self.global_critic.train()
        """
        阶段一：数据准备与“复盘”
        """
        #Todo 1. 从经验池提货
        '''
        1.当前状态数据
        这是神经网络（Actor 和 Critic）的输入原料。
        adj: 邻接矩阵。描述了 FJSP 中工序之间的先后约束关系（析取图）。GCN/GIN 网络用它来理解工序结构。
        tasks_fea: 工序特征。包含工序的原始属性（如完工时间、状态等）。
        candidate: 候选工序列表。告诉 Job Actor 当前哪些工序是可以被调度的（处于待加工状态）。
        mask_operation: 工序掩码。标记哪些工序不能选（例如已完成的工序），防止模型输出非法动作。
        machine_fea1: 候选机器特征。这是你在 parallel_env.py 里精心计算的 (6维) 特征，包含候选机器加工当前任务的能耗、时间等预测值。
        machine_fea2: 机器状态特征。机器当前的实际状态（如累计负荷、位置等），用于 Critic 评估全局局势。
        2. 下一时刻状态数据
        带下划线 _ 的变量。主要用于计算 TD-Error (时序差分误差) 和 GAE。
        adj_, tasks_fea_, candidate_, mask_operation_, machine_fea2_:这些是执行了动作之后，环境变成的新样子。
        3. 历史动作与策略
        这是 PPO 算法计算 Ratio (重要性采样比率) 的分母。
        a_operation: 旧动作（工序）。当时采集数据时，Agent 到底选了哪个工序。
        a_machine: 旧动作（机器）。当时给这个工序选了哪台机器。
        a_logprob_operation: 旧策略的概率（工序）。当时 Job Actor 选这个动作的 Log 概率。
        a_machine_logprob: 旧策略的概率（机器）。当时 Machine Actor 选这个机器的 Log 概率。
        4. “绿色调度”核心资产：分离的奖励
        reward: 总奖励。可能是加权后的总和，通常用于计算一个笼统的 Global Adv。
        mk_r (Makespan Reward): 这一步操作对完工时间的贡献（通常是负值或惩罚）。
        pt_r (Power Reward): 这一步操作产生的能耗（加工能耗）。
        tt_r (Transport Reward): 这一步产生的运输时间/能耗。
        it_r (Idle Reward): 机器的空闲时间惩罚。
        5. 偏好权重
        random_weight: 随机权重向量。含义：在采集这条数据的那一刻，系统给（时间、能耗、运输）分配的权重是多少？比如 [0.8, 0.1, 0.1] 代表当时极度重视时间。
        6. 旧价值估计
        用于辅助计算 Loss 或者作为 Critic 更新的参考。
        job_v, machine_v: 旧的状态价值。当时 Critic 认为当前状态值多少分。
        job_v_, machine_v_: 旧的下一状态价值。
        '''
        adj, tasks_fea, candidate, mask_operation, a_operation, a_logprob_operation, \
            adj_, tasks_fea_, candidate_, mask_operation_, reward, done, \
                machine_fea2, a_machine, a_machine_logprob, machine_fea2_, mask_machine_batch_, \
                    mk_r, pt_r, tt_r, it_r, machine_fea1, random_weight, \
                        job_v, machine_v, job_v_, machine_v_ = replay_buffer.numpy_to_tensor_operation()


        global_r = reward

       #Todo 2. 全局评论家算全局状态价值
        gae = 0
        adv = []
        rewards = []
        with torch.no_grad():
            # 1. 重新计算当前状态的全局价值
            multi_v = self.step_for_net_out_Critic_GAT(
                        net_model=self.global_critic,
                        task_fea=tasks_fea,
                        graph_pool_avg=graph_pool_avg,  # step * env_batch * （env_batch * tasks）
                        adj=adj,  # step * env_batch * tasks * tasks
                        candidate=candidate, #
                        machine_fea1=machine_fea1, # step * env_batch * m* 6
                        machine_fea2=machine_fea2  # step * env_batch * m* 8
                        )    # 按照step重新输入，shape = buffer_step * env_batch * 1
            # 2. 构造下一时刻的“候选机器特征”
            # 因为 ReplayBuffer 里存的 machine_fea1 是当前的，我们需要手动构造下一时刻的
            machine_fea1_ = copy.deepcopy(machine_fea1)
            for i in range(machine_fea1.shape[0]): # step的个数
                if i == machine_fea1.shape[0]-1: # 表明是最后一个step
                    # machine_fea1_[i] = torch.rand((configs.env_batch, configs.n_machine, 6))  # bs*m*6, 01均匀分布的随机浮点数
                    machine_fea1_[i] = machine_fea1[i]  # 最后一个step，就用上一次一样的
                else:
                    machine_fea1_[i] = machine_fea1[i + 1]  # machine_fea1整体左移 = 去掉首位，补上一个最后位，总长度不变
            # 3. 重新计算下一时刻的全局价值
            multi_v_ = self.step_for_net_out_Critic_GAT(
                        net_model=self.global_critic,
                        task_fea=tasks_fea_,
                        graph_pool_avg=graph_pool_avg,
                        adj=adj_,
                        candidate=candidate_,
                        machine_fea1=machine_fea1_,
                        machine_fea2=machine_fea2_,
                        )
            # Todo 3. 计算优势函数
            # 1.局部优势计算
            #目的：利用 Job Actor 和 Machine Actor 自带的 Local Critic来计算优势。
            local_adv_list = self.cal_local_job_machine_reward_GAE(
                        mk_r=mk_r,
                        pt_r=pt_r,
                        tt_r=tt_r,
                        it_r=it_r,
                        jv=job_v,
                        jv_=job_v_,
                        mv=machine_v,
                        mv_=machine_v_,
                        done_operation=done)
            job_adv_mk = local_adv_list[0]  # buffer_step * env_batch
            job_adv_it = local_adv_list[3]
            mac_adv_pt = local_adv_list[1]
            mac_adv_tt = local_adv_list[2]
            job_v_target_mk = job_adv_mk + job_v[:,:,0]
            job_v_target_it = job_adv_it + job_v[:,:,1]
            machine_v_target_pt = mac_adv_pt + machine_v[:, :, 0]  
            machine_v_target_tt = mac_adv_tt + machine_v[:, :, 1]

            #2.全局优势计算
            adv_list = self.separate_cal_4_reward_GAE(
                mk_r=mk_r, 
                pt_r=pt_r, 
                tt_r=tt_r, 
                it_r=it_r,
                v=multi_v,
                v_=multi_v_,
                done_operation=done)

            adv_mk = adv_list[0]
            adv_pt = adv_list[1]
            adv_tt = adv_list[2]
            adv_it = adv_list[3]
            global_v_target_list = [adv_list[i] + multi_v[:, :, i:(i + 1)].squeeze(-1) for i in range(4)]
            Logger.log("Training/buffer_done/adv_targetV", f"Calculate the Local/Global Adv and TargetV based on GAE", print_true=1)

        """
        阶段二：PPO 循环训练
        """
        # 1. 初始化 Loss 字典
        # 记录三个核心网络的 Loss
        loss_dict = {"job_actor_loss": [], "machine_actor_loss": [], "global_critic_loss": []}
        # 2. 初始化 梯度范数 字典
        # 用于监控梯度是否爆炸。如果梯度太大，网络参数会更新飞起，导致崩盘。
        grad_norm_dict = {"a_machine": [], "a_operation": [], "v": []}

        # 3. 开始 K-Epochs 循环。含义：我们将拿着手里这批数据，反复训练 K 次。
        for i_K in range(args['K_epochs']):
            # 4. 清空临时 Loss 列表
            # actor_loss_lst: 存 Machine Actor 的 loss。actor_loss_lst_operation: 存 Job Actor 的 loss
            actor_loss_lst, actor_loss_lst_operation, critic_loss_lst = [], [], []
            # 5. 清空临时 梯度 列表，记录每一次反向传播时的梯度大小
            a_grad_norm_lst, a_grad_norm_lst_operation, v_grad_norm_lst = [], [], []
            # 6. BatchSampler: 批次采样器，它的作用是产生一组组的"索引号" (index)，比如 [0, 5, 8, 12...]
            for index in BatchSampler(SubsetRandomSampler(range(tasks_fea.shape[0])),
                                      mini_bs,
                                      False):
                # 7. SubsetRandomSampler: 随机子集采样器，作用：它会把 0~3199 这些数字完全打乱 (Shuffle)。
                # 8. mini_bs: 小批次大小，例如 64。每次只训练 64 个样本。
                # 9. drop_last=False。如果最后剩下的数据不够 64 个（比如剩 20 个），不丢弃，照样训练。
                # 10. 初始化 机器全图特征，因为数据被打乱了，没有上下文了，所以初始时刻没有"上一时刻的机器特征"，设为 None。
                h_mch_pooled = None
                # 11. 初始化 概率与价值 收集列表，因为我们要把 mini-batch一个个喂给网络，所以用列表先存着结果。
                o_prob_lst = []  # 存 Job Actor 的概率
                m_prob_lst = []  # 存 Machine Actor 的概率
                lst_job_v = []  # 存 Job Critic 的价值
                lst_machine_v = []  # 存 Machine Critic 的价值
                # 12. 遍历 Mini-Batch 里的每一个样本
                for i in range(tasks_fea[index].shape[0]):
                    """
                    阶段三：Actor 更新
                    """
                    # TODO -1.重新采样
                    """----------------- Job Actor 重新采样--- ----------------------"""
                    # 调用 Job Actor 的 forward 函数
                    #目的：虽然Buffer里存了当时的数据，但神经网络参数可能已经变了。我们需要把旧的状态再喂给现在的网络吃一遍，看看现在的网络会输出什么概率。
                    # 输入：这一步的状态 (tasks_fea[index][i])，图结构 (adj[index][i]) 等
                    # 输出：
                    #   prob: 新的动作概率分布
                    #   h_g_o_pooled: 工序图的全局特征 (传给 Machine Actor 用)
                    #   job_v: 本地 Critic 的价值预测
                    _, _, _, prob, h_g_o_pooled, job_v = self.job_actor(
                                x_fea=tasks_fea[index][i],
                                graph_pool_avg=graph_pool_avg,  # 全局只有一个
                                padded_nei=None,
                                adj=adj[index][i],
                                candidate=candidate[index][i],
                                h_g_m_pooled=h_mch_pooled,
                                mask_operation=mask_operation[index][i],
                                use_greedy=False)
                    o_prob_lst.append(prob)  # 输出action的概率：mini_batch_size * env_batch * job
                    lst_job_v.append(job_v)  # 输出自身的value：mini_batch_size * env_batch * 2

                    """----------------- Machine Actor 重新采样 ----------------------"""
                    # 调用 Machine Actor 的 forward 函数
                    # 输出：
                    #   mch_prob (机器选择概率)：在当前状态下，Machine Actor 认为应该选择每一台机器的概率分布。
                    #   h_mch_pooled (机器图池化特征)：这是 GAT 网络处理完所有机器特征后，通过 Pooling 得到的“全车间机器状态向量
                    #   mac_v (机器局部价值)：Machine Actor 自带的 Local Critic 对当前状态的打分。
                    mch_prob, h_mch_pooled, mac_v = self.machine_actor_gcn(
                                machine_fea_1=machine_fea1[index][i],  # 从ReplayBuffer里边查到的！
                                machine_fea_2=machine_fea2[index][i],
                                h_pooled_o=h_g_o_pooled,
                                machine_mask=mask_machine_batch_[index][i])  
                    m_prob_lst.append(mch_prob)  # mini_batch * env_batch * m
                    lst_machine_v.append(mac_v)  # 输出自身的value：mini_batch_size * env_batch * 2
                # TODO -2.概率分布与比率计算
                # 1. 堆叠.把列表里的 tensor 拼起来。形状变回 [mini_bs, env_batch, action_dim]
                j_action_prob = torch.stack(o_prob_lst,dim=0)
                m_action_prob = torch.stack(m_prob_lst, dim=0)
                lst_job_v = torch.stack(lst_job_v, dim=0)
                lst_machine_v = torch.stack(lst_machine_v, dim=0)

                """---------------------- 计算对数概率 (Log Prob) ----------------------"""
                # 2. 构建分布。用新的概率构建一个离散分布对象
                j_dist_now = Categorical(probs=j_action_prob)
                # 3. 计算新策略下，"当年那个动作" 的概率。
                # a_operation[index]: 这是 Buffer 里存的"当年真正选的动作"。
                # log_prob(): 问现在的网络，"如果让你现在选当年那个动作，概率是多少？"
                j_a_logprob_now = j_dist_now.log_prob(a_operation[index])
                # 同理处理 Machine Actor
                m_dist_now = Categorical(probs=m_action_prob)
                m_a_logprob_now = m_dist_now.log_prob(a_machine[index])
                """---------------------- 计算比率 (Importance Sampling Ratio) ----------------------"""
                # 4. 计算 Ratio = exp(New_Log - Old_Log) = New / Old
                # a_logprob_operation[index]: Buffer 里存的"旧概率"
                # .detach(): 旧概率是常数，不需要梯度
                job_ratios = torch.exp(j_a_logprob_now - a_logprob_operation[index].detach())
                machine_ratios = torch.exp(m_a_logprob_now - a_machine_logprob[index].detach())

                # TODO -3.计算 PPO Loss
                """--------------------- job Actor 的全局 PPO Loss ---------------------"""
                # 1. 计算第一路优势
                # job_ratios: 新旧策略比率 (New/Old)。如果 >1 说明新策略更倾向选这个动作。
                # adv_mk[index]: 这是 separate_cal_4_reward_GAE 算出来的时间优势。
                # 含义：如果这个动作能缩短时间 (adv>0)，且新策略更爱选它 (ratio>1)，surr1 就会很大(好事)。
                surr1_job_mk = job_ratios * adv_mk[index]
                # 2. 计算第二路优势
                # torch.clamp: 把 ratio 强行限制在 [0.8, 1.2] 之间 (假设 epsilon=0.2)。
                # 目的：防止 ratio 太大导致一次更新步子迈得太扯。
                surr2_job_mk = torch.clamp(job_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_mk[index]

                surr1_job_pt = job_ratios * adv_pt[index]  
                surr2_job_pt = torch.clamp(job_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_pt[index]

                surr1_job_tt = job_ratios * adv_tt[index]  
                surr2_job_tt = torch.clamp(job_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_tt[index]

                surr1_job_it = job_ratios * adv_it[index]  
                surr2_job_it = torch.clamp(job_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_it[index]

                # 3. 取最小值
                # 核心逻辑：在 surr1 和 surr2 中取较小的那个。
                # 效果：只允许“适度”的优化，拒绝“过激”的跳跃。这是 PPO 稳定的秘诀。
                loss_mk = torch.min(surr1_job_mk, surr2_job_mk)
                loss_pt = torch.min(surr1_job_pt, surr2_job_pt)
                loss_tt = torch.min(surr1_job_tt, surr2_job_tt)
                loss_it = torch.min(surr1_job_it, surr2_job_it)

                """--------------------- 多目标加权融合 (Weighted Sum) ---------------------"""
                # random_weight[index]: 这是一个 [mini_bs, env_batch, 3] 的张量。
                # 它记录了当初采集这条数据时，系统设定的偏好：[时间权重, 能耗权重, 运输权重]。
                # 逐项加权求和：
                global_loss_job_actor = random_weight[index][:,:,0] * loss_mk \
                            + random_weight[index][:,:,1] * (loss_pt + loss_it) \
                            + random_weight[index][:,:,2] * loss_tt   

                """--------------------- job Actor 的局部 PPO Loss ---------------------"""
                surr1_job_mk_local = job_ratios * job_adv_mk[index]
                surr2_job_mk_local = torch.clamp(job_ratios, 1 - self.epsilon, 1 + self.epsilon) * job_adv_mk[index]
                
                surr1_job_it_local = job_ratios * job_adv_it[index]  # 
                surr2_job_it_local = torch.clamp(job_ratios, 1 - self.epsilon, 1 + self.epsilon) * job_adv_it[index]
                
                loss_mk_local = torch.min(surr1_job_mk_local, surr2_job_mk_local)
                loss_it_local = torch.min(surr1_job_it_local, surr2_job_it_local)
                
                #局部 job，Loss 加权求和
                local_loss_job_actor = random_weight[index][:, :, 0] * loss_mk_local \
                                + random_weight[index][:, :, 1] * loss_it_local   

                
                """---------------------Machine Actor 的全局 PPO Loss---------------------"""
                # adv_xx = （buffer_step * env_batch）改为 adv_xx[index] = (mini_batch_size X env_batch)
                surr1_machine_mk = machine_ratios * adv_mk[index]  # 原版重要性采样
                surr2_machine_mk = torch.clamp(machine_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_mk[index] 
                
                surr1_machine_pt = machine_ratios * adv_pt[index]  
                surr2_machine_pt = torch.clamp(machine_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_pt[index]

                surr1_machine_tt = machine_ratios * adv_tt[index]  
                surr2_machine_tt = torch.clamp(machine_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_tt[index]

                surr1_machine_it = machine_ratios * adv_it[index]  
                surr2_machine_it = torch.clamp(machine_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_it[index]

                loss_mk_m = torch.min(surr1_machine_mk, surr2_machine_mk)  # (mini_batch_size X env_batch)
                loss_pt_m = torch.min(surr1_machine_pt, surr2_machine_pt)
                loss_tt_m = torch.min(surr1_machine_tt, surr2_machine_tt)
                loss_it_m = torch.min(surr1_machine_it, surr2_machine_it)


                global_loss_machine = random_weight[index][:,:,0] * loss_mk_m \
                                + random_weight[index][:,:,1] * (loss_pt_m + loss_it_m) \
                                + random_weight[index][:,:,2] * loss_tt_m

                """---------------------Machine Actor 的局部 PPO Loss---------------------"""

                surr1_pt_machine_local = machine_ratios * mac_adv_pt[index]  # 按照machine本地value的ADV计算对应pt和tt指标的surr
                surr2_pt_machine_local = torch.clamp(machine_ratios, 1 - self.epsilon, 1 + self.epsilon) * mac_adv_pt[index]
                
                surr1_tt_machine_local = machine_ratios * mac_adv_tt[index]  
                surr2_tt_machine_local = torch.clamp(machine_ratios, 1 - self.epsilon, 1 + self.epsilon) * mac_adv_tt[index]
                
                loss_pt_m_local = torch.min(surr1_pt_machine_local, surr2_pt_machine_local)
                loss_tt_m_local = torch.min(surr1_tt_machine_local, surr2_tt_machine_local)

                local_loss_machine = random_weight[index][:, :, 1] * loss_pt_m_local  \
                                    + random_weight[index][:, :, 2] * loss_tt_m_local

                # TODO -4.计算 熵和局部评论家损失（Local Critic Loss）来训练 Actor 内部的价值判断能力。
                # 1. 计算 Job Actor 的熵
                # 作用：在最终计算总 Loss 时，我们会 减去这个熵（Loss = ... - coeff * entropy）。
                # 这会迫使网络在训练初期不要过早收敛（即不要过早变得非常有自信），保持一定的随机探索能力，防止陷入局部最优。
                # j_dist_now: 刚才构建的 Job 动作分布 (Categorical Distribution) # .entropy(): 计算公式是 -sum(p * log(p))。
                # 含义：熵越大，概率分布越平坦（越随机）；熵越小，分布越尖锐（越确定）。
                job_dist_entropy = j_dist_now.entropy()  # shape(mini_batch_size * env_batch)
                # 2. 计算 Machine Actor 的熵。同理，计算机器选择策略的随机程度。
                machine_dist_entropy = m_dist_now.entropy()  # shape(mini_batch_size * env_batch)
                #2. 准备多目标权重
                w_mk = random_weight[index][:, :, 0]
                w_ec = random_weight[index][:, :, 1]
                w_tt = random_weight[index][:, :, 2]

                #3. 计算 Job Actor 本地评论家损失 (Job Local Critic Loss)
                # Job Actor 的 Local Critic 负责预测：完工时间 (mk) + 空闲时间 (it)。
                job_critic_loss_mk = self.global_critic_loss_func(w_mk * job_v_target_mk[index],
                                                              w_mk * lst_job_v[:, :, 0:1].squeeze(-1))  
                job_critic_loss_it = self.global_critic_loss_func(w_ec * job_v_target_it[index],
                                                              w_ec * lst_job_v[:, :, 1:2].squeeze(-1))
                #4. 计算 Machine Actor 本地评论家损失 (Machine Local Critic Loss)
                machine_critic_loss_pt = self.global_critic_loss_func(w_ec * machine_v_target_pt[index],
                                                              w_ec * lst_machine_v[:, :, 0:1].squeeze(-1))  
                machine_critic_loss_tt = self.global_critic_loss_func(w_tt * machine_v_target_tt[index],
                                                              w_tt * lst_machine_v[:, :, 1:2].squeeze(-1))  # MSE之后是标量
                #5. 汇总 Local Critic Loss
                job_critic_loss = job_critic_loss_mk + job_critic_loss_it
                machine_critic_loss = machine_critic_loss_pt + machine_critic_loss_tt

                # TODO -5.总 Loss 组装

                # TODO 0108- 消融实验 = 只用全局critic和自身的重要性采样
                # job_actor_loss = -2 * global_loss_job_actor - self.ENTROPY_BETA * job_dist_entropy  # shape(mini_batch_size X env_batch)
                # 1.计算 Job Actor 总损失，这是 Job Actor 的最终考核指标。
                """job_actor_loss = -2 * 全局actor + -1 * 局部actor + 0.5*局部critic(MSE后标量，会自动广播拓展) - deta * entropy"""
                job_actor_loss = -2 * global_loss_job_actor + (-1) * local_loss_job_actor + 0.5 * job_critic_loss - self.ENTROPY_BETA * job_dist_entropy 
                
                # TODO 0108-消融实验 = 只用全局critic和自身的重要性采样
                # machine_actor_loss = -2 * loss_machine - self.ENTROPY_BETA * dist_entropy  # shape(mini_batch_size X env_batch)
                # 2.计算 Machine Actor 总损失
                """machine_actor_loss = -2* 全局actor + -1 * 局部actor + 0.5*局部critic(MSE后标量，会自动广播拓展) - deta * entropy"""
                machine_actor_loss = -2 * global_loss_machine + (-1) * local_loss_machine + 0.5 * machine_critic_loss - self.ENTROPY_BETA * machine_dist_entropy  
                
                Logger.log("Training/buffer_done/actor_loss", f"job_actor_loss={job_actor_loss.shape}, machine_actor_loss={machine_actor_loss.shape}", print_true=1)  # 两个job和machine的网络的loss

                #3.联合反向传播与更新
                # 清空梯度,必须清空！否则梯度会和上一个 Batch 的梯度累加，导致更新方向错误。
                self.job_actor_optimizer.zero_grad()
                # 2. 梯度裁剪 (针对 Job Actor)
                # use_grad_clip: 开关，通常为 True。
                # clip_grad_norm_: 如果梯度向量太长（> CLIP_GRAD），就把它剪短。
                # 作用：防止“梯度爆炸”。如果梯度太大，参数一次更新太多，模型会直接崩盘（Loss 变成 NaN）。
                if args['use_grad_clip']:  # Trick 7: Gradient clip
                    a_grad_norm_operation = torch.nn.utils.clip_grad_norm_(self.job_actor.parameters(),
                                                                           args['CLIP_GRAD'])  # 梯度裁剪，默认是0.5
                    a_grad_norm_lst_operation.append(a_grad_norm_operation)  # 保存minibatch的每一次的梯度缩放因子
                # 3. 清空梯度 (针对 Machine Actor)
                self.machine_actor_optimizer_gcn.zero_grad()
                # 4. 梯度裁剪 (针对 Machine Actor)
                if args['use_grad_clip']:  # Trick 7: Gradient clip
                    a_grad_norm = torch.nn.utils.clip_grad_norm_(self.machine_actor_gcn.parameters(),
                                                                 args['CLIP_GRAD'])  # 梯度裁剪，默认是0.5
                    a_grad_norm_lst.append(a_grad_norm)  # 保存minibatch的每一次的梯度缩放因子

                # 5. 联合 Loss 求和
                # .mean(): 因为前面的 Loss 还是 [mini_bs, env_batch] 的张量，这里求平均变成标量。
                loss = job_actor_loss.mean() + machine_actor_loss.mean()
                # 6. 反向传播
                # 这条指令会沿着计算图回溯，找到 Job Actor 和 Machine Actor 的所有权重，计算它们对 Loss 的贡献。
                loss.backward()
                # 7. 参数更新
                # 优化器根据刚才算出的梯度，修改网络参数。
                self.job_actor_optimizer.step()
                self.machine_actor_optimizer_gcn.step()
                
                Logger.log("Training/buffer_done/update_actor_net", f"loss = job_actor_loss.mean() + machine_actor_loss.mean() = {loss}", print_true=1) # 总loss进行梯度计算
                

                """
                阶段四：全局评论家更新 (Global Critic Update)
                """

                multi_v_s = self.step_for_net_out_Critic_GAT(
                            net_model=self.global_critic,
                            task_fea=tasks_fea[index],
                            graph_pool_avg=graph_pool_avg,
                            adj=adj[index],
                            candidate=candidate[index],
                            machine_fea1=machine_fea1[index],
                            machine_fea2=machine_fea2[index]
                            )  # 按照step重新输入，shape = buffer_step * env_batch * 4
                            
                """
                TODO 新增随机权重的critic的更新方式！！！！因为MSE之后是标量，所以要分别在MSE中的两个变量中乘以权重！！！（先计算W*V，加权和成一个）
                
                需要按照随机权重来搞！
                此时的RW = shape = train_bs * env_bs * 3, 按照Minibatch的index进行提取，然后分别提取对应mk、ec和tt的具体某一列，然后降维squeeze
                    # w1={random_weight[index][:, :, 0].squeeze(-1)}
                    # w2={random_weight[index][:, :, 1].squeeze(-1)}
                    # w3={random_weight[index][:, :, 2].squeeze(-1)},  buffer_step * env_bs, 选取其中的index个，= mini_bs * env_bs
                    
                v_target[index] = v_target（buffer_step * env_batch）--（mini_batch_size * env_batch）
                """
                w_mk = random_weight[index][:,:,0] # shape = minibatch * env_bs
                w_ec = random_weight[index][:,:,1]
                w_tt = random_weight[index][:,:,2]

                # global_v_target_list = [mk_r, pt_r, tt_r, it_r] = 4个（buffer_step * env_batch）的元素
                critic_loss_mk = self.global_critic_loss_func(w_mk * global_v_target_list[0][index],
                                                                w_mk * multi_v_s[:, :, 0:1].squeeze(-1))  # （mini_batch_size * env_batch）
                critic_loss_pt = self.global_critic_loss_func(w_ec * global_v_target_list[1][index],
                                                                w_ec * multi_v_s[:, :, 1:2].squeeze(-1))  
                critic_loss_tt = self.global_critic_loss_func(w_tt * global_v_target_list[2][index],
                                                                w_tt * multi_v_s[:, :, 2:3].squeeze(-1))  
                critic_loss_it = self.global_critic_loss_func(w_ec * global_v_target_list[3][index],
                                                                w_ec * multi_v_s[:, :, 3:4].squeeze(-1))  # TODO MSE之后就是标量了！！！！

                critic_loss = critic_loss_mk + critic_loss_pt + critic_loss_it + critic_loss_tt  # TODO 注意对应好自定义顺序mk+pt+tt+it！！！

                # Update critic
                self.global_critic_optimizer.zero_grad()
                critic_loss.backward()  # 多次计算需要保存计算图为true（选择action和计算adv无梯度计算，就不用保存计算图！）
                
                if args['use_grad_clip']:  # Trick 7: Gradient clip
                    v_grad_norm = torch.nn.utils.clip_grad_norm_(self.global_critic.parameters(),
                                                                 args['CLIP_GRAD'])  # 梯度裁剪，默认是0.5
                    v_grad_norm_lst.append(v_grad_norm)  # 保存minibatch的每一次的梯度缩放因子
                self.global_critic_optimizer.step()
                
                # print("1111:{}".format(get_GPU_usage()[1]))
                
                Logger.log("Training/buffer_done/update_global_critic_net", f"critic_loss={critic_loss}", print_true=1) # 全局critic_loss进行梯度计算
                

                """K_epoch=5遍更新，每次更新需要buffer_step/mini_bs_step=buffer_size轮才能选完buffer，每一轮都会计算一次loss，记录，然后更新网络！"""
                actor_loss_lst_operation.append(job_actor_loss.mean())  # 记录actor的loss，记得求解mean之后再记录！！！
                actor_loss_lst.append(machine_actor_loss.mean())  # 记录actor的loss，记得求解mean之后再记录！！！
                critic_loss_lst.append(critic_loss)  # 记录critic的loss，每次update的时候都会重新记录（TODO MSE后直接标量！）
                
                


                

            # 计算下整个buffer_step都训练完的平均loss,视为1次epoch,dim=0第一维度求均值（计算每一列的均值）
            loss_dict["machine_actor_loss"].append(torch.mean(torch.stack(actor_loss_lst), dim=0))  # 整个buffer_step都训练完，求均值，作为单个epoch的loss
            loss_dict["job_actor_loss"].append(torch.mean(torch.stack(actor_loss_lst_operation), dim=0))  # 先list中tensor堆叠成大tensor，再求mean，也是张量;
            loss_dict["global_critic_loss"].append(torch.mean(torch.stack(critic_loss_lst), dim=0))  
            
            Logger.log("Training/buffer_done/update_1_epoch_done", f"--------Update K_epoch: {i_K+1}/{args['K_epochs']}, ReplayBuffer: every {mini_bs} in {range(tasks_fea.shape[0])},  job_actor_loss={torch.mean(torch.stack(actor_loss_lst_operation), dim=0)}, machine_actor_loss={torch.mean(torch.stack(actor_loss_lst), dim=0)}, global_critic_loss={torch.mean(torch.stack(critic_loss_lst), dim=0)}--------", print_true=1) # log
            
            # 将张量移动到CPU设备，TODO 一次epoch，梯度裁剪相关系数，250425暂时不用
            a_grad_norm_lst_cpu = [tensor.cpu() for tensor in a_grad_norm_lst]
            a_grad_norm_lst_operation_cpu = [tensor.cpu() for tensor in a_grad_norm_lst_operation]
            v_grad_norm_lst_cpu = [tensor.cpu() for tensor in v_grad_norm_lst]
            grad_norm_dict["a_machine"].append(torch.mean(torch.stack(a_grad_norm_lst_cpu), dim=0))  # 先list中tensor堆叠成大tensor，再求mean，也是张量;
            grad_norm_dict["a_operation"].append(torch.mean(torch.stack(a_grad_norm_lst_operation_cpu), dim=0))  
            grad_norm_dict["v"].append(torch.mean(torch.stack(v_grad_norm_lst_cpu), dim=0))  
        
        """记录所有epoch训练之后的平均loss（每个epoch，有buffer_step个样本，分开minibatch个进行随机训练）"""
        train_loss_j_a = torch.mean(torch.stack(loss_dict['job_actor_loss']), dim=0).detach().cpu().numpy()   #所有epoch训练完，取均值作为当前train(update)的训练结果
        train_loss_m_a = torch.mean(torch.stack(loss_dict['machine_actor_loss']), dim=0).detach().cpu().numpy()
        train_loss_c_a = torch.mean(torch.stack(loss_dict['global_critic_loss']), dim=0).detach().cpu().numpy()
        Logger.log("Training/buffer_done/update_all_epoch_done", f"Mean: job_actor_loss={train_loss_j_a}, machine_actor_loss={train_loss_m_a}, global_critic_loss={train_loss_c_a}", print_true=1) # update函数调用一次，反馈的loss
        
        """250425-新增一个Logger，用来记录loss，用于wandb的输出 + loss随着episode发生变化，buffer满了就训练5次，采集5个轨迹就满了，训练每次随机挑mini_bs组成一个轨迹，buffer中5个轨迹用完，算作训练1次；5次epoch训练完，相当于这一轮update结束！记录的loss的5次训练均值，直接喂给wandb，用log的dict形式（DT中episode=iteration，Kepoch=num_steps训练次数） + 验证集是每过10个episode用100组数据来验证，相应的obj直接记录，然后喂给wandb！"""
        # episode里边，buffer_size倍task开始训练，训练kepoch次，每次采用mini_bs=task完成所有buffer_step（即更新网络参数buffer_size次）
        Result_Logger.log_not_str(f"Training/Update/job_actor_loss", train_loss_j_a)  
        Result_Logger.log_not_str(f"Training/Update/machine_actor_loss", train_loss_m_a)
        Result_Logger.log_not_str(f"Training/Update/global_critic_loss", train_loss_c_a)
        
        if args['use_lr_decay']:  # Trick 6:learning rate Decay
            # 线性衰减
            # 固定步数*系数衰减（update一次，衰减一次）
            self.job_actor_lr_decay.step()  # Actor的LR衰减，
            self.machine_actor_lr_decay_gcn.step()  # Actor的LR衰减，
            self.global_critic_lr_decay.step()  # Critic的LR衰减

            for p in self.job_actor_optimizer.param_groups:  # 可以访问优化器参数组的列表。通常情况下，我们只使用一个参数组
                lr_job = p["lr"]  # 获取update之后的lr
            for p in self.machine_actor_optimizer_gcn.param_groups:  # 可以访问优化器参数组的列表。通常情况下，我们只使用一个参数组
                lr_machine = p["lr"]  # 获取update之后的lr
            for p in self.global_critic_optimizer.param_groups:  # 可以访问优化器参数组的列表。通常情况下，我们只使用一个参数组
                lr_critic = p["lr"]
            Logger.log("Training/buffer_done/update_lr_decay", f"lr_job={lr_job}, lr_machine={lr_machine}, lr_critic={lr_critic}", print_true=1) # 获取update之后的lr
        
        loss_mean_lst = [train_loss_j_a, train_loss_m_a, train_loss_c_a]  # 所有kepoch训练完的mean
        loss_std_lst = [torch.std(torch.stack(loss_dict['job_actor_loss']), dim=0).detach().cpu().numpy(),
                        torch.std(torch.stack(loss_dict['machine_actor_loss']), dim=0).detach().cpu().numpy(),
                        torch.std(torch.stack(loss_dict['global_critic_loss']), dim=0).detach().cpu().numpy()]
        
        # return loss_dict, grad_norm_dict  # 返回k_epoch个元素的loss字典 + 总梯度范数字典
        return loss_mean_lst, loss_std_lst  # 返回k_epoch次训练的loss的mean和std，转为数组！ 
    
    # 全局变量清零函数
    def set_to_0(self, env):
        """
        A2C：如果连续运行，这些都需要请0！！！！！！！！！！！！！！！！！！！
        全局变量，记录某些随step变化的值
        """
        self.chosen_taskID_list = []  # 最终选择的task的id    TODO eval的时候使用
        self.chosen_taskID_list_batch = [[] for _ in range(self.batch_size)]  # ! TODO 所有batch中所选择的task的id的, 按照step顺序在对应bs的list中进行append 原chosen_action_list_batch
  
        self.pool_task_dict = {}  # 用作更新的任务池，可选择task还有哪些？   TODO eval的时候使用
        for i in range(self.n_job):
            self.pool_task_dict[i] = self.pool_task_list[i]
        
        self.pool_task_dict_batch = []  # 用作更新的任务池的batch版本，可选择task还有哪些？
        for _ in range(self.batch_size):
            task_dict = {}  # 初始化字典，用来添加到列表!一定要在这里重新初始化！！！！！！！！！！！！！！！！！！！！！！！！！相当于新建内存！否则改一个dict，其他dict全变了！
            for i in range(self.n_job):
                task_dict[i] = self.pool_task_list[i]
            self.pool_task_dict_batch.append(task_dict)  # # 记录batch_size个的各个job的可以选择的task的id = candidate的候选task的id


        self.remaining_m = {}  # 创建一个字典作为每个job的剩余任务数量，用作掩码，排除不能选择的job（即行数）  TODO eval的时候使用
        for i in range(self.n_job):
            self.remaining_m[i] = self.n_machine  # machine默认就是每个job的子任务数量
        
        self.remaining_m_batch = []
        for _ in range(self.batch_size):
            remain_m = {}  # 初始化字典，用来添加到列表!一定要在这里重新初始化！！！！！！！！！！！！！！！！！！！！！！！！！相当于新建内存！
            for j in range(self.n_job):
                remain_m[j] = self.n_machine  # machine默认就是每个job的子任务数量
            # remain_m = {0:4,1:4,2:4,3:4}
            self.remaining_m_batch.append(remain_m)  # # 记录batch_size个的各个job的剩余task个数

        lst = [0.0] * self.n_job  # ! TODO job_mask就是屏蔽当前可以选择的job（一列一列来选择，保证紧凑，所有指标都好！不然就是我旧方法，完全随机选择！）
        self.mask_new = torch.tensor(lst).cuda()  # job选择时的mask机制（第一列完全随机，后续按照candidate最小的先选）     TODO eval的时候使用
        
        self.mask_new_batch = []
        for _ in range(self.batch_size):  #! TODO job_mask就是屏蔽当前可以选择的job（一列一列来选择，保证紧凑，所有指标都好！不然就是我旧方法，完全随机选择！）
            lst1 = [0.0] * self.n_job
            self.mask_new_batch.append(lst1)  # 用于和batch版本的logit进行相加，注意转成tensor
        self.mask_new_batch = torch.tensor(self.mask_new_batch).cuda()  # 转成tensor用于和net中的logit相加， torch.tensor([0.0, 0.0],[0.0, 0.0],...)
        
        
        
        


            
    