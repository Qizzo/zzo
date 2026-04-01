import pickle
import csv
import math
import psutil
import os
import json
import torch
import argparse
import copy
import random
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '/1 DRL/E2E-MAPPO-for-MT-FJSP-main')  # 将a文件夹的路径添加到Python路径,0表示添加到最前边
from algorithm.ppo_algorithm import PPOAlgorithm
from instance.generate_allsize_mofjsp_dataset import Instance_Dataset, Logger, Result_Logger
from trainer.parallel_env import Parallel_env
from trainer.replaybuffer import ReplayBuffer
from model.gcn_mlp import g_pool_cal, aggr_obs
from algorithm.agent_func import select_machine_action
from trainer.validate import validate_cost_gcn_jointActor_GAT, read_MIP_result_from_csv
from trainer.fig_kpi import plot_show
from parameters import args
import wandb
import heapq  # 最小堆，堆中从小到大进行排序，插入会自动保持堆的顺序

def experiment(
        variant,
):
    # TODO one.数据加载与基础配置初始化
    # 1.数据集路径配置
    pth = '/1 DRL/E2E-MAPPO-for-MT-FJSP-main/instance'
    # 动态拼接训练集的绝对路径，格式如 "Instance_J6M6E2.pkl"
    dataset_path = pth + f"/Instance_J{variant['n_job']}M{variant['n_machine']}E{variant['n_edge']}.pkl"
    train_instance = Instance_Dataset(
            generate_true=0,
            dataset_pth=dataset_path)  
    eval_dataset_path = pth + f"/eval_Instance_J{variant['n_job']}M{variant['n_machine']}E{variant['n_edge']}.pkl"
    eval_instance = Instance_Dataset(
            generate_true=0,
            dataset_pth=eval_dataset_path)

    # 2.训练数据加载器
    train_loader = DataLoader(train_instance, 
                               batch_size=variant['env_batch'],#决定了每次抓取多少个算例交给并行环境同时跑。
                               shuffle=True, #每个 epoch 都会把数据顺序打乱
                               num_workers=4)  #开启 4 个子进程在后台默默帮你搬运数据。NumPy 数组转换成 Tensor 张量。

    # 3. 实验日志与监控 (WandB) 命名配置
    specific_name = 'No_LR_decay+Add_eval()'
    group_name = f"Instance_J{variant['n_job']}M{variant['n_machine']}E{variant['n_edge']}" + f"-Weight_{variant['weight_mk']}{variant['weight_carbon']}{variant['weight_tou']}" + f"-BS_{variant['env_batch']}-Seed_{variant['train_seed']}"   # 便于区分实验场景
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'   # 生成一个介于100,000到999,999之间随机整数，作为独立易于识别的实验标识符
    if variant['log_to_wandb']:  # wandb的初始化，就可以在网站看到统计结果
        wandb.init(
            name=exp_prefix + specific_name,
            group=group_name,
            project='E2E-MAPPO_for_MT-FJSP',
            config=variant
        )
    # 4. 实例化强化学习三大核心组件

    # (1)初始化 PPO 算法，内部包含选工序Actor、选机器Actor、全局Critic，以及各自的优化器
    ppo = PPOAlgorithm(args=variant, load_pretrained=False)
    # todo (2)初始化并行调度环境，它的内部同时维护 env_batch 个独立的车间状态
    paral_env = Parallel_env(args=variant)
    # todo (3)初始化经验回放池，用于在交互时收集 (state, action, log_prob, reward) 轨迹，供后续 PPO 计算优势并更新
    replay_buffer = ReplayBuffer(args=variant)

    # 5.知识注入（预训练模型加载）
    # 定位历史“记忆”的存放仓库
    model_pth = "/1 DRL/E2E-MAPPO-for-MT-FJSP-main/trained_model/"
    # PPO 算法有三个网络：选工序的 Actor、选机器的 Actor、全局 Critic。它们是分开保存的。
    job_name = "PPO_job_actor_"
    machine_name = "PPO_machine_actor_"
    critic_name = "PPO_global_critic_"
    # 提取要加载的“历史模型”的规模特征
    # 这里的 Exist_j, Exist_m, Exist_e 代表之前已经训练好的小场景的工件、机器、边数
    size_index = 0  # 确定当前是哪一个场景
    Exist_j = variant['load_pth']['Exist_jme'][size_index][0]  # 之前的小场景训练好的
    Exist_m = variant['load_pth']['Exist_jme'][size_index][1] 
    Exist_e = variant['load_pth']['Exist_jme'][size_index][2] 
    Exist_epi = variant['load_pth']['Exist_epi'][size_index]
    # 根据类型，拼接历史模型的文件名
    if variant['load_model_type'] == 0:
        # 类型 0：加载效果最好（Top 1）的模型
        job_f = job_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + "_top1.pth"    # 选择保存的已存在model的文件名称
        machine_f = machine_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + "_top1.pth" 
        critic_f = critic_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + "_top1.pth"
    elif variant['load_model_type'] == 1:
        # 类型 1：加载训练结束时（Final）保存的模型
        job_f = job_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + '_final.pth'
        machine_f = machine_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + '_final.pth'
        critic_f = critic_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + '_final.pth'
    elif variant['load_model_type'] == 2:
        # 类型 2：加载指定轮次（EP）的模型
        job_f = job_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + f'_EP{Exist_epi}_.pth'
        machine_f = machine_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + f'_EP{Exist_epi}_.pth'
        critic_f = critic_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + f'_EP{Exist_epi}_.pth'
    elif variant['load_model_type'] == 3:
        # 类型 3：加载效果第二好（Top 2）的模型
        job_f = job_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + '_top2.pth'
        machine_f = machine_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + '_top2.pth'
        critic_f = critic_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + '_top2.pth'

    # 生成最终的绝对路径
    path = variant['trained_model_pth']
    job_model = os.path.join(model_pth, job_f)
    mch_model = os.path.join(model_pth, machine_f)
    critic_model = os.path.join(model_pth, critic_f)
    # 执行知识装载
    # 如果允许使用旧模型，且该路径下确实有文件，就把硬盘里的网络参数加载进内存中的网络对象里
    if os.path.exists(job_model) and variant['use_load_model'] == True:
        print('-------------------------------------------load the TRAINNED model--------------------------------------------')
        # 加载 Job Actor 的参数
        ppo.job_actor.load_state_dict(torch.load(job_model))
        # 加载 Machine Actor 的参数
        ppo.machine_actor_gcn.load_state_dict(torch.load(mch_model))  # 按照路径地址加载预训练好的模型
        # 加载 Critic 的参数
        ppo.global_critic.load_state_dict(torch.load(critic_model))  # 按照路径地址加载预训练好的模型
        # 将装载成功的模型参数以日志形式记录下来
        Logger.log("Init/load_model", f"ppo.job_actor={ppo.job_actor.state_dict()}, ppo.machine_actor_gcn={ppo.machine_actor_gcn.state_dict()}, ppo.global_critic={ppo.global_critic.state_dict()}", print_true=1) # 打印出加载后的模型中的所有参数


    #6.辅助工具与记账本就位

    #预计算 GNN 的图池化算子
    graph_pool_avg = g_pool_cal(graph_pool_type=variant['neighbor_pooling_type'],
                                batch_size=variant['env_batch'],
                                n_nodes=variant['n_job'] * variant['n_machine'],
                                device=variant['device'])

    # 读取数学规划 (MIP) 的精确解作为基准线
    # 这些是从 CSV 读取的如 Gurobi 等求解器算出的最优值/下界值。
    # 1. 拼接完整的本地路径，保留 %s 占位符以便填入工件数、机器数等参数
    MIP_result_file = "/1 DRL/E2E-MAPPO-for-MT-FJSP-main/MIP_result/MO_FJSP_MIP_result_(J%s_M%s_seed%s_sample%s_w%s%s%s).csv" % \
                      (variant['n_job'], variant['n_machine'], variant['eval_seed'], 100,
                       int(variant['weight_mk'] * 10), int(variant['weight_carbon'] * 10), int(variant['weight_tou'] * 10))

    # 2. 加上“空字典保护伞”，防止你本地没有这个 CSV 文件导致程序再次崩溃
    try:
        MIP_cost_dict = read_MIP_result_from_csv(MIP_result_file)
    except FileNotFoundError:
        print(f"\n[!] Warning: MIP result file NOT FOUND at: {MIP_result_file}")
        print("[!] Training will continue. Gap calculation in evaluation will use placeholders.\n")
        # 伪造一个全 1 字典，保证 eval 逻辑不报错
        MIP_cost_dict = {
            "Makespan": [1.0] * 101,
            "MachineEC": [1.0] * 101,
            "TransEC": [1.0] * 101,
            "MachineIdleT": [1.0] * 101
        }

    # 初始化所有的参数
    current_bs_idx = 0 # 追踪当前从 DataLoader 里抽到了第几个批次
    all_steps = 0    # 记录智能体总共走过的决策步数，用于驱动学习率衰减等机制

    # 记录 PPO 训练过程的变量：Loss 均值、方差、当前学习率
    loss_mean_lst = []
    loss_std_lst = []
    lr_lst = [variant['LR']]

    # 记录环境交互结果的变量：即时 Reward、最终的四个单目标 Cost、加权综合 Objective

    infor_means_lst = []
    Final_4cost_lst = []
    Obj_lst = []

    # 记录验证集 (Eval) 表现的变量
    eval_cost_lst = []
    cost_best = float('inf')
    top3_obj_heap = []
    traj_lst =[]

    # TODO two.采样

    # 1. 开启训练大循环，i_episode 代表当前是第几轮练习
    for i_episode in range(variant['episode_num']):

        # 2. 判断是否到了“换题”时间
        # resample_freq 定义了每隔多少轮更换一次训练算例（例如每 20 轮换一批）
        if i_episode % variant['resample_freq'] == 0:
            # 3. 打印当前采样进度日志
            print('=' * 250)
            Logger.log("Training/resample_instance", f"--------------------Instance Sample：{(i_episode // variant['resample_freq'] + 1) * variant['env_batch']}/{train_instance.__len__()}---------------", print_true=1)
            # 4. 从数据流水线中提取当前批次的数据
            for bs_idx, bs_data in enumerate(train_loader):
                if bs_idx == current_bs_idx:
                    instance_bs_dict = bs_data
            Logger.log("Training/current_instance", f"bs_idx={bs_idx}, instance_bs_dict['t'].shape={instance_bs_dict['t'].shape}", print_true=1)
            # 5. 计数器加 1，确保下次“换题”时取 DataLoader 的下一个 batch
            current_bs_idx += 1
            # 6. 将新抽取的算例数据注入到并行环境对象中
            # 此时，16 个并行环境（env_batch）各自领到了一个新的工厂排产任务
            paral_env.get_batch(instance_bs_dict)

            # 7. 初始化奖励缩放器
            paral_env.init_RewardScaling_sameBATCH(shape=4)
            Logger.log("Training/resample_status", f"Now is paral_env.get_batch + paral_env.init_RewardScaling_sameBATCH", print_true=1)

        # 8. 获取当前时刻的状态特征
        adj_batch, machine_scheduled_fea_batch, tasks_fea_batch = paral_env.init_DGFJSPEnv_state0()

        # 9. 构造第一拍的候选工序清单
        value_list = []
        for dict in ppo.pool_task_dict_batch:  # 遍历每个环境的待做任务字典
            value_list.append(list(dict.values()))
        candidate_batch = np.array(value_list) - 1

        # 10. 初始化工序 Actor 的动作掩码
        mask_operation_batch = ppo.mask_new_batch.bool()

        # 11. 初始化机器 Actor 的物理约束掩码

        mask_machine_batch = instance_bs_dict["t"].numpy() >= 0
        mask_machine_batch = torch.tensor(mask_machine_batch).to(variant['device'])
        mask_machine_batch0 = ~mask_machine_batch
        h_mch_pooled = None

        # 12. 每一轮（Episode）开始，重置奖励缩放器的累加器
        for i in range(len(paral_env.paral_Rscaling_instance)):
            paral_env.paral_Rscaling_instance[i].reset()

        step_flag_for_v_ = 0

        # TODO three .测试训练
        while True:
            with torch.no_grad(): # 采样阶段不计算梯度，节省显存和计算量
                # 1. 工序智能体 (Job Actor) 调用，选出一个当前最该被加工的工序 (task_index)。
                task_index, action_index, log_a, _, h_g_o_pooled, job_v = ppo.job_actor(
                            x_fea=tasks_fea_batch,
                            graph_pool_avg=graph_pool_avg,
                            padded_nei=None,
                            adj=adj_batch,
                            candidate=candidate_batch,
                            h_g_m_pooled=h_mch_pooled,
                            mask_operation=mask_operation_batch,
                            use_greedy=False
                            )

                # 2. 精准匹配机器掩码
                # 根据刚才选中的工序 (task_index)，从全局机器能力矩阵中提取该工序对应的机器约束。
                mask_machine_batch_ = torch.gather(mask_machine_batch0,
                                                   1,
                                                   task_index.unsqueeze(-1).unsqueeze(-1).expand(mask_machine_batch0.size(0), -1,mask_machine_batch0.size(2)))

                # 3. 提取当前任务的专属机器特征
                machine_candidate_fea_batch = paral_env.cal_cur_task_machine_feature(
                            task_index=task_index,
                            m_mask=mask_machine_batch_,
                            all_task_fea=tasks_fea_batch)

                # 4. 机器智能体 (Machine Actor) 跟进决策
                # 它参考工序智能体给出的“全场战况”(h_g_o_pooled)，决定把这个工序分给哪台机器。
                mch_prob, h_mch_pooled, machine_v = ppo.machine_actor_gcn(
                            machine_fea_1=machine_candidate_fea_batch,  # bs*m*6  对应task的m特征
                            machine_fea_2=machine_scheduled_fea_batch,  # bs*m*8  上一次选完m后的ENV的更新m特征
                            h_pooled_o=h_g_o_pooled,   # task全图张量h_g_o_pooled = bs * hidden 
                            machine_mask=mask_machine_batch_)  # batch * 1 * m machine的mask
                # 5. 最终确定机器动作
                # 根据输出的概率分布 mch_prob 进行随机采样，得到具体的机器编号 m_action。

                m_action, m_action_logprob = select_machine_action(mch_prob)

            # 6. 组装联合动作对
            # 将 (选中的工序, 选中的机器) 配对，交给并行环境执行。
            joint_actions = [x for x in zip(task_index.tolist(), m_action.tolist())]

            # 7. 环境推进一步
            # 环境反馈：新的图结构 (adj_batch_)、奖励信息 (oenv_step_info)、以及更新后的机器和工序特征。
            adj_batch_, oenv_step_info, machine_scheduled_fea_batch_, tasks_fea_batch_ = paral_env.DGFJSPEnv_paral_step(joint_actions)

            # 8. 同步更新
            # 更新每个 Job 的进度，计算下一拍哪些工序变成了“候选人”，并生成新的 Mask。
            candidate_batch_, mask_operation_batch_ = ppo.esa_update_chosenTaskID_CandidateTaskIDx_JobMask(
                        paralenv=paral_env, 
                        action_batch=action_index,
                        mask_value=variant['mask_value'])

            # 9. 提取多目标奖励分量
            o_r = [copy.deepcopy(info[0]) for info in oenv_step_info]  # 综合奖励
            mk = [copy.deepcopy(info[2]) for info in oenv_step_info]  # 位2对应的是finish time
            it = [copy.deepcopy(info[3]) for info in oenv_step_info]  # 位3对应的是idle time
            pt = [copy.deepcopy(info[4]) for info in oenv_step_info]  # 位4对应的是p*t加工能耗
            tt = [copy.deepcopy(info[5]) for info in oenv_step_info]  # 位5对应的是transT
            check_done_operation = [info[1] for info in oenv_step_info]  # 检查 16 个环境是否都排完了

            # 10. 记录下一时刻的价值估计 (V_next)
            # 这是为了后续计算 GAE 优势函数做准备。
            step_flag_for_v_ += 1
            if step_flag_for_v_ > 1:
                replay_buffer.store_v_next(j_v_=job_v,
                                           m_v_=machine_v)
            # 11. 终局处理，计算终点状态的 V
            # 判定所有并行环境是否均已完成调度
            if all(check_done_operation):
                # 开启无梯度模式，仅进行推理计算
                with torch.no_grad():
                    # 计算工序智能体在“排产结束”那一刻的价值
                    _, _, _, _, h_g_o_pooled_, job_v_ = ppo.job_actor(
                                x_fea=tasks_fea_batch_,
                                graph_pool_avg=graph_pool_avg,
                                padded_nei=None,
                                adj=adj_batch_,
                                candidate=candidate_batch_,
                                h_g_m_pooled=h_mch_pooled,
                                mask_operation=mask_operation_batch,
                                use_greedy=False
                                )
                    # 机器特征对齐，因为调度已经结束，没有新的任务可选，所以直接复用最后一次的机器候选特征。
                    machine_candidate_fea_batch_ = machine_candidate_fea_batch
                    # 计算机器智能体在“排产结束”那一刻的价值
                    # 这里同样是为了得到机器端的终点价值 machine_v_。
                    _, _, machine_v_ = ppo.machine_actor_gcn(
                                machine_fea_1=machine_candidate_fea_batch_,
                                machine_fea_2=machine_scheduled_fea_batch_,
                                h_pooled_o=h_g_o_pooled_,
                                machine_mask=mask_machine_batch_)
                    # 将最后时刻的下一状态价值存入经验池
                    replay_buffer.store_v_next(j_v_=job_v_,
                                               m_v_=machine_v_)
                    # 重置价值记录标志位，为下一个 Episode 做准备
                    step_flag_for_v_ = 0

            # 12. 提取当前步骤的权重偏好 (RW)
            rw = [copy.deepcopy(paral_env.paral_env_DG[i_bs].reward_random_weight) for i_bs in range(len(oenv_step_info))]  # env_bs个,遍历不同ENV中的随机权重值，是一个3元素的一维矩阵

            # 13. 写入经验池 (Replay Buffer)
            # 存入所有的 state, action, reward, mask 等，作为后续 PPO 更新的训练材料。
            replay_buffer.store_operation(
                        adj=adj_batch,
                        fea=tasks_fea_batch,
                        candidate=candidate_batch,
                        mask=mask_operation_batch,
                        a_o=action_index,
                        a_o_logprob=log_a,
                        r=o_r,
                        adj_=adj_batch_,
                        fea_=tasks_fea_batch_,
                        candidate_=candidate_batch_,
                        mask_=mask_operation_batch_,
                        mch_fea1=machine_candidate_fea_batch,
                        mch_fea2=machine_scheduled_fea_batch,
                        mch_fea2_=machine_scheduled_fea_batch_,
                        a_m=m_action,
                        a_m_logprob=m_action_logprob,
                        dw=None,
                        done=check_done_operation,
                        mask_machine_=mask_machine_batch_,
                        mk=mk,
                        pt=pt,
                        tt=tt,
                        it=it,
                        rw=rw,
                        j_v=job_v,
                        m_v=machine_v)

            # 14. 触发 PPO 训练逻辑
            # 当经验池中的数据量达到预设的 Buffer 大小时（如采集了 5 条完整轨迹），执行一次网络更新。
            all_steps += 1
            if replay_buffer.count_operation == variant['n_job'] * variant['n_machine'] * variant['buffer_size']:
                # 计算训练批次的参数
                tasks_n = variant['n_job'] * variant['n_machine'] # 总工序数
                mini_batch_size = tasks_n  # 每次从 Buffer 中抽取的“小样”大小设为总工序数
                buffer_size = tasks_n * variant['buffer_size']  # 整个经验池的总容量
                
                Logger.log("Training/while/buffer_done/update", f"++++++++++++++++++++++++++++ ppo update {all_steps//buffer_size}/{variant['episode_num']*tasks_n//buffer_size} buffer_size={buffer_size}||mini_bs={mini_batch_size}||Date_freq_epi={variant['resample_freq']} ++++++++++++++++++++++++++++", print_true=1)

                traj_tuple = replay_buffer.numpy_to_tensor_operation()
                traj_tuple = tuple(tensor.cpu().numpy() for tensor in traj_tuple)
                # 将本次训练的所有数据点（轨迹）存入总列表
                traj_lst.append(traj_tuple)

                # 调用 PPO 算法进行训练
                # 这个方法会拿着 Buffer 里的数据跑 K 次循环，计算 Loss，反向传播并更新网络参数。
                # 它返回本次更新中所有 K 轮训练的 Loss 均值和标准差。
                loss_mean, loss_std = ppo.global_update_JointActions_GAT_selfCritic(  
                                replay_buffer=replay_buffer, 
                                all_steps=all_steps,
                                graph_pool_avg=graph_pool_avg,
                                args=variant,
                                mini_bs=mini_batch_size) 

                # 清空经验池计数器
                # 训练完了，旧经验已经用过，把计数器清零，准备迎接下一波采样。
                replay_buffer.count_operation = 0
                replay_buffer.count_operation_ = 0

                # 记录 Loss 指标
                loss_mean_lst.append(loss_mean)
                loss_std_lst.append(loss_std)

                # 追踪学习率 (LR) 变化
                # PPO 算法内部带有学习率衰减机制。这里从优化器中读取最新的 LR 并存入列表。
                for p in ppo.job_actor_optimizer.param_groups:
                    lr_lst.append(p["lr"])

                Logger.log("Training/while/buffer_done/update_done", f"Update Done on episode {i_episode+1}/{variant['episode_num']}", print_true=1)  # update函数中已log每次update返回的k_epoch个loss的平均值
                print("++" * 250)

            # 15. 状态迭代：把“下一时刻”变成“当前时刻”，准备进行 while 的下一次循环
            adj_batch = adj_batch_
            tasks_fea_batch = tasks_fea_batch_
            candidate_batch = candidate_batch_
            mask_operation_batch = mask_operation_batch_
            machine_scheduled_fea_batch = machine_scheduled_fea_batch_

            # 16. 计算并记录当前步的平均奖励
            infor_means_step = np.array(oenv_step_info).mean(axis=0)
            infor_means_lst.append(infor_means_step)

            #TODO ZZO
            # 17. 检查是否跳出决策循环
            if all(check_done_operation):
                # 从所有并行环境 (env_batch) 中提取 5 个核心调度指标的总和
                # makespan: 完工时间 | total_e1: 加工能耗 | trans_t: 运输时间 | idle_t: 机器空闲时间 | tou_cost: 分时电费
                costs = [sum(getattr(paral_env.paral_env_DG[i_bs], attr) for i_bs in range(variant['env_batch']))
                         for attr in ['makespan_previous_step', 'total_e1_previous_step', 'trans_t_previous_step',
                                      'idle_t_previous_step', 'tou_cost_previous_step']]

                # 将五个指标的总和除以并行环境数，得到本轮 16 个环境的平均表现
                Final_costs = np.array(costs) / variant['env_batch']

                # 依然截取前 4 个原生物理量用于旧版画图 (完工, 加工能耗, 运输时间, 空闲时间)
                Final_4cost = Final_costs[:4]
                Final_4cost_lst.append(Final_4cost)

                # 1. 直接提取我们在底层写好的精确切片电费
                avg_tou_cost = Final_costs[4]

                # 2. 计算机器碳排和物流碳排 (乘上之前在 parameter.py 里定义的碳排因子)
                obj_m_carbon = (Final_4cost[1] + Final_4cost[3]) * variant['carbon_grid']
                obj_agv_carbon = Final_4cost[2] * variant['carbon_agv']

                # 3. 按照全新的 3 权重公式严格合并
                Objective = variant['weight_mk'] * Final_4cost[0] + \
                            variant['weight_carbon'] * (obj_m_carbon + obj_agv_carbon) + \
                            variant['weight_tou'] * avg_tou_cost
                # ==========================================================

                # 记录该综合得分
                Obj_lst.append(Objective)
                
                print('**'*250)
                Logger.log("Training/while/done", f"Trajectory Done (step = tasks) episode {i_episode+1}/{variant['episode_num']} ", print_true=1)
                if i_episode == variant['episode_num'] - 1:
                    paral_env.paral_env_DG[-1].render()

                # 重置 PPO 算法内部的状态变量
                ppo.set_to_0(None)
                for i in range(paral_env.batch_size):
                    paral_env.paral_env_DG[i].reset()
                paral_env.reset_data()
                
                print("**"*250)

                break  # 直接跳出 while循环！

        # TODO four.val
        # 1. 触发条件，每过validate_iter个episode，就进行validate
        if ((i_episode + 1) % variant['eval_freq'] == 0) or (i_episode == variant['episode_num'] - 1):

            print("------------------------------------------------Evaluation-------------------------------------------------------")
            eval_obj_lst = []  # 记录每一数据的综合得分 (Objective)
            eval_4_cost_lst = []  # 用于记录每一个测试数据的4指标的真实值，大lst，每个小lst包含 [mk, pt, transT, idleT]
            gantt_flag = False   # 是否render渲染画图

            # 2.循环遍历eval_sample个验证算例
            for i_eval in range(variant['eval_sample']):
                if i_eval == variant['eval_sample'] - 1:
                    gantt_flag = True
                if i_eval >= 99:
                    gantt_flag = True
                    print('*'*250)
                    print("Validation {}/{}".format((i_eval + 1), variant['eval_sample']))
                    print('*'*250)

                # 3. 调用专用的val类
                # greedy=True：都选概率最大的那个最优动作
                eval_cost_dict_cumsum, eval_4_cost, obj = validate_cost_gcn_jointActor_GAT(
                            ppo=ppo,
                            gantt_flag=gantt_flag,
                            data=eval_instance,  # 传入的是dataset，定义的可以直接.x读取数据
                            data_index=i_eval,
                            data_type=variant['eval_data_type'],  
                            greedy=True,
                            args=variant)

                # 4. 对照MIP算出的最优解
                idea_cost_refer = [MIP_cost_dict["Makespan"][i_eval], MIP_cost_dict["MachineEC"][i_eval],
                                   MIP_cost_dict["TransEC"][i_eval], MIP_cost_dict["MachineIdleT"][i_eval]]

                # 5. 计算相对误差
                # 公式：(RL跑出的值 - MIP理想值) / MIP理想值。这个值越接近 0，说明强化学习的效果越逼近精确解
                new_related_cost = (np.array(eval_4_cost) - np.array(idea_cost_refer)) / np.array(idea_cost_refer)
                # 计算误差的加权总和
                eval_obj_gap = variant['weight_mk'] * new_related_cost[0] + variant['weight_carbon'] * (new_related_cost[1] + new_related_cost[3]) \
                                + variant['weight_tou'] * new_related_cost[2]
                
                eval_obj_lst.append(obj)
                eval_4_cost_lst.append(eval_4_cost)
                
                if i_eval >= 99:  # 最后一个测试集
                    Logger.log("Evaluation/last_instance", f"Objective={obj}, eval_obj_gap={eval_obj_gap}, mk={eval_4_cost[0]}, pt={eval_4_cost[1]}, transT={eval_4_cost[2]}, idleT={eval_4_cost[3]}", print_true=1)  # eval最后一组数据的结果


            # 6. 算均分与方差
            # axis=0 表示对这 100 道题的成绩“按列”求平均值和标准差
            obj_eval_mean = np.mean(np.array(eval_obj_lst), axis=0)
            obj_eval_std = np.std(np.array(eval_obj_lst), axis=0)

            # 此时 eval_4cost_mean 的 4 个槽位分别是：0:mk, 1:m_carbon, 2:agv_carbon, 3:tou_cost
            eval_4cost_mean = np.mean(np.array(eval_4_cost_lst), axis=0)
            eval_4cost_std = np.std(np.array(eval_4_cost_lst), axis=0)

            # 写入 txt 文件，用于后续画图。名称全部换成新的物理量。
            Result_Logger.log_not_str(f"Evaluation/100instances/obj_eval_mean", obj_eval_mean)
            Result_Logger.log_not_str(f"Evaluation/100instances/obj_eval_std", obj_eval_std)
            Result_Logger.log_not_str(f"Evaluation/100instances/mk_eval_mean", eval_4cost_mean[0])
            Result_Logger.log_not_str(f"Evaluation/100instances/mk_eval_std", eval_4cost_std[0])
            Result_Logger.log_not_str(f"Evaluation/100instances/m_carbon_eval_mean", eval_4cost_mean[1])
            Result_Logger.log_not_str(f"Evaluation/100instances/m_carbon_eval_std", eval_4cost_std[1])
            Result_Logger.log_not_str(f"Evaluation/100instances/agv_carbon_eval_mean", eval_4cost_mean[2])
            Result_Logger.log_not_str(f"Evaluation/100instances/agv_carbon_eval_std", eval_4cost_std[2])
            Result_Logger.log_not_str(f"Evaluation/100instances/tou_cost_eval_mean", eval_4cost_mean[3])
            Result_Logger.log_not_str(f"Evaluation/100instances/tou_cost_eval_std", eval_4cost_std[3])

            eval_cost_lst.append(obj_eval_mean)

            # 控制台的“大字报”打印：只显示我们关心的最新均值
            Logger.log("Evaluation/100instances/each_episode_result",
                       f"----------episode={i_episode}, obj_100ins_mean={obj_eval_mean:.2f}, "
                       f"mk_mean={eval_4cost_mean[0]:.2f}, m_carbon_mean={eval_4cost_mean[1]:.2f}, "
                       f"agv_carbon_mean={eval_4cost_mean[2]:.2f}, tou_cost_mean={eval_4cost_mean[3]:.2f}------------------------",
                       print_true=1)
            # 7. 拼接本地 CSV 文件的保存路径

            save_pth = "/1 DRL/E2E-MAPPO-for-MT-FJSP-main/results/"
            result_file = save_pth + f"Obj_100_EvalInstance_J{variant['n_job']}_M{variant['n_machine']}_E{variant['n_edge']}_BS{variant['env_batch']}_Weight{int(variant['weight_mk'] * 10)}{int(variant['weight_carbon'] * 10)}{int(variant['weight_tou'] * 10)}.csv"

            with open(result_file, 'w', newline='') as csvfile1:
                writer = csv.writer(csvfile1)
                writer.writerows([eval_cost_lst])
            id_name = f"J{variant['n_job']}M{variant['n_machine']}E{variant['n_edge']}"
            top_prex = ["_top3.pth", "_top2.pth", "_top1.pth"]

            # 8. 保存最终版模型
            # 只要进了验证环节，不管成绩好坏，先覆盖保存一份 _final.pth。
            model1f = model_pth+job_name+id_name + "_final.pth"
            model2f = model_pth+machine_name+id_name + "_final.pth"
            model3f = model_pth+critic_name+id_name + "_final.pth"
            torch.save(ppo.job_actor.state_dict(), model1f)  # 
            torch.save(ppo.machine_actor_gcn.state_dict(), model2f)
            torch.save(ppo.global_critic.state_dict(), model3f)  # 
            # 9. 准备当前轮次的模型文件路径
            model1 = model_pth + job_name + id_name + f'_EP{i_episode + 1}_.pth'
            model2 = model_pth + machine_name + id_name + f'_EP{i_episode + 1}_.pth'
            model3 = model_pth + critic_name + id_name + f'_EP{i_episode + 1}_.pth'
            # 10. 存入最小堆
            # heapq 是 Python 自带的堆结构。堆顶永远是列表中最小的值。
            heapq.heappush(top3_obj_heap, (-obj_eval_mean, model1, model2, model3))
            torch.save(ppo.job_actor.state_dict(), model1)  # 
            torch.save(ppo.machine_actor_gcn.state_dict(), model2)
            torch.save(ppo.global_critic.state_dict(), model3)  # 

            # 11. 淘汰最差模型
            # 如果堆里的模型超过了 3 个，就把堆顶弹出来。
            if len(top3_obj_heap) > 3:
                _, m1_old, m2_old, m3_old = heapq.heappop(top3_obj_heap)
                os.remove(m1_old)   # 用于删除指定路径的文件
                os.remove(m2_old)   # 用于删除指定路径的文件
                os.remove(m3_old)   # 用于删除指定路径的文件

            # 12. 最后一轮的处理
            if i_episode == variant['episode_num'] - 1:

                # 将最后留在堆里的 3 个模型，按照成绩从小到大排序
                sorted_list = sorted(top3_obj_heap, key=lambda x: x[0])
                for i_tup in range(len(sorted_list)):

                    model1 = model_pth + job_name + id_name + top_prex[i_tup]
                    model2 = model_pth + machine_name + id_name + top_prex[i_tup]
                    model3 = model_pth + critic_name + id_name + top_prex[i_tup]
                    # 使用os.rename()来重命名文件, 保证原文件存在
                    os.rename(sorted_list[i_tup][1], model1)  # list里边不同的元组，元组包含最小cost和3个网络的参数
                    os.rename(sorted_list[i_tup][2], model2)
                    os.rename(sorted_list[i_tup][3], model3)
                
                Logger.log("Evaluation/save model.pth", f"Final and TOP3 Best model parameters saved. Best_model={sorted_list[-1]}", print_true=1)  #  
            

        pth = '/1 DRL/E2E-MAPPO-for-MT-FJSP-main/results/'
        result_f = pth + f"Loss_Cost_new_J{variant['n_job']}M{variant['n_machine']}E{variant['n_edge']}.txt"
        logg = Result_Logger.get_logs(pth=result_f)
        if variant['log_to_wandb']:
            wandb.log(logg)


    pth1 = '/1 DRL/E2E-MAPPO-for-MT-FJSP-main/trajectory/'
    traj_f = pth1 + f"Trajectory_{variant['episode_num']}_J{variant['n_job']}M{variant['n_machine']}E{variant['n_edge']}_Seed{variant['train_seed']}_BS{variant['env_batch']}_Weight{int(variant['weight_mk'] * 10)}{int(variant['weight_carbon'] * 10)}{int(variant['weight_tou'] * 10)}.pkl"
    with open(traj_f, 'wb') as f:
        pickle.dump(traj_lst, f)  # 储存轨迹 = episode_num，同一批batch走buffer_size遍，ppo不断更新进化中
    Logger.log("Training/All_episode/save_buffer_trajectory", f"save trajectory in {traj_f}.pkl'", print_true=1)

    

if __name__ == '__main__':
    
    experiment(variant=vars(args))  # vars内置函数，转为dict，表示参数和其值
    
    pth2 = '/1 DRL/E2E-MAPPO-for-MT-FJSP-main/'
    training_log_file = pth2 + f"/training_log_J{args.n_job}M{args.n_machine}E{args.n_edge}_BS{args.env_batch}_InsSeed{args.train_seed}_Weight{int(args.weight_mk * 10)}{int(args.weight_carbon * 10)}{int(args.weight_tou * 10)}.txt"
    Logger.get_logs(training_log_file)
    print('=' * 250)
    
