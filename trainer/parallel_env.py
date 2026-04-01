import numpy as np
import random
import copy
from instance.generate_allsize_mofjsp_dataset import Logger
from algorithm.ppo_trick import RewardScaling
# ! TODO (LLM) root@01a43040a453:/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/graph-jsp-env# pip install -e .  (需要在新的根目录安装下env的环境)- https://github.com/RKWin93/graph-jsp-env
from graph_jsp_env.disjunctive_graph_jsp_env_singlestep import DisjunctiveGraphJspEnv_singleStep

"""
Parallel_env 类
这个类充当了 PPO 算法 和 底层仿真环境 (DisjunctiveGraphJspEnv_singleStep) 之间的中间层。
输入：来自 dataset 的原始数据（加工时间、能耗、运输时间等）。
输出：打包好的 Tensor/Numpy 数组，直接喂给神经网络。
"""

class Parallel_env(object):
    def __init__(self, args):  # 传入arg参数，已经转成dict形式了
        
        self.njobs = args['n_job']
        self.nmachines = args['n_machine']
        self.ntasks = self.njobs * self.nmachines
        self.nedges = args['n_edge']
        self.batch_size = args['env_batch']
        self.m_scaling = args['m_scaling'] # m选择中的reward放缩比例
        self.reward_dict = args['reward_scaling'] # DG图中direction选择中的reward放缩比例

        self.args = args  # 为了方便传递参数设置

        self.ability_instance = []  # 记录并行的ability instance的所有信息
        self.paral_Rscaling_instance = [] # 记录针对同一样本中的放缩类，相同样本运行多少次，都不会清0，除非换样本
        self.paral_env_DG = []  # 记录并行的DG的环境
        self.oenv_info = []  # 记录每一个batch的reward, done, r_t, r_idle, r_energy_m, r_energy_transM


    """
    调用时机：在 Run.py 的训练循环中，每当 i_episode % resample_freq == 0（即需要换一批新题目做的时候），就会调用它。
    输入的数据结构（按类型分类）：
    t_batch: [工厂1的时间表, 工厂2的时间表, ..., 工厂16的时间表]
    p_batch: [工厂1的能耗表, 工厂2的能耗表, ..., 工厂16的能耗表]transT_batch，edge_batch
    输出（按工厂分类）：batch_size个instance，每个instance有4元素
    instance_1: [时间, 能耗, 运输, 布局]
    instance_2: [时间, 能耗, 运输, 布局]
    """
    def get_batch(self, dataset_dict):
        """
    Todo
        作用：输入：dataset_dict，这是一个字典，包含了一大批（比如 16 个）调度问题的原始数据。
        t: 加工时间矩阵 [Batch, J, M]
        p: 加工功率矩阵 [Batch, J, M] —— 绿色调度重点
        transT: 运输时间矩阵
        edge: 机器布局信息
        核心逻辑：
        数据转 Numpy：先把 Tensor 转成 Numpy 数组，方便后续处理。
        清空旧货：self.ability_instance = []，把上一轮训练用的数据清空。
        重新打包：通过一个 for 循环，把每个样本的 t, p, transT, edge 打包成一个 list，存入 self.ability_instance。
        为什么这么做：底层环境类 (DisjunctiveGraphJspEnv) 每次只能处理一个问题实例，所以这里要把 batch 数据拆开存放，方便后面一对一分发。
        """
        Logger.log("Training/Parallel_env/get_batch_scenario", f"job={self.njobs}, machine={self.nmachines}, edge={self.nedges}, tasks={ self.ntasks}", print_true=1)
        # 先转成array数组
        t_batch = dataset_dict["t"].numpy()
        p_batch = dataset_dict["p"].numpy() 
        transT_batch = dataset_dict["transT"].numpy() 
        edge_batch = dataset_dict["edge"].numpy() 

        self.ability_instance = [] # 既然来了新货，就要把上一批训练用的 16 个工厂数据清空，准备装新的。

        for i_batch in range(self.batch_size):
            
            #【打包】把第 i 号工厂的所有资料（t, p, transT, edge）收集到一起
            instance = [copy.deepcopy(t_batch[i_batch]),
                        copy.deepcopy(p_batch[i_batch]),
                        copy.deepcopy(transT_batch[i_batch]),
                        copy.deepcopy(edge_batch[i_batch])]  # 一次采样的实例信息存到一个list中,元素为np.array
            # 【入库】记录在总的列表中， batch_size个instance，每个instance有4元素
            self.ability_instance.append(copy.deepcopy(instance))

        Logger.log("Training/Parallel_env/get_batch_info", f"self.ability_instance={np.array(self.ability_instance).shape}, t[0].shape={self.ability_instance[0][0].shape}, p[0].shape={self.ability_instance[0][1].shape}, transT[0].shape={self.ability_instance[0][2].shape}", print_true=1)   # bs=1的样本数据 TODO tensor张量用size，array数组用shape，list列表用len（看大小）

    def init_RewardScaling_sameBATCH(self, shape):
        """
    Todo
        作用：为这 16 个并行环境，分别准备一个独立的“奖励缩放器”。
        核心逻辑：循环 batch_size 次，实例化 RewardScaling 类。
        在多目标优化（时间+能耗）中，这个函数非常关键，因为不同目标的数值量级差异巨大（比如时间可能是几十，而能耗可能是几千），如果不进行独立的缩放，数值大的目标会主导训练。
        """
        self.paral_Rscaling_instance = [] # 准备一个列表，用来装 batch_size 个缩放器
        for _ in range(self.batch_size):
            Rscaling = RewardScaling(shape=shape, gamma=self.args['GAMMA'])
            self.paral_Rscaling_instance.append(copy.deepcopy(Rscaling))

    """
    功能：“开局初始化”。它利用之前加载好的数据，创建 batch_size（如 16）个全新的仿真环境对象，并让它们执行 Reset 操作，返回神经网络所需的初始状态（State）。
    输入 (Input)：隐式输入：依赖类属性 self.ability_instance（存放了 16 个工厂的原始数据）和 self.batch_size。
    输出 (Output)： 返回一个包含三个元素的元组 (adj_batch, machine_fea_batch, tasks_fea_batch)，作为 PPO 算法的初始观察值：
    adj_batch (Numpy Array):形状：(Batch_Size, Total_Tasks, Total_Tasks)，例如 (16, 36, 36)。含义：16 个样本的初始析取图邻接矩阵（描述工序间的连接关系）。
    machine_fea_batch (Numpy Array):形状：(Batch_Size, Machines, 8)，例如 (16, 6, 8)。含义：16 个样本里所有机器的初始状态特征（如初始都是空闲的）。
    tasks_fea_batch (Numpy Array):注意形状：(Batch_Size * Total_Tasks, 12)，例如 (576, 12)。含义：所有样本的所有工序特征拼成了一长条。
    """
    def init_DGFJSPEnv_state0(self):

        self.paral_env_DG = []  # 初始直接清0，防错
        adj_batch = []  # 准备列表装 16 个图
        tasks_fea_batch = [] # 准备列表装 16 组工序特征
        machine_fea_batch = [] # 准备列表装 16 组机器特征
        for ii_batch in range(self.batch_size):
            # 1. 取出第 ii 个工厂的原始数据
            # jsp_instance 包含 [时间矩阵, 能耗矩阵]
            jsp_instance = np.array([self.ability_instance[ii_batch][0], self.ability_instance[ii_batch][1]])
            # 2. 实例化仿真环境
            # DisjunctiveGraphJspEnv_singleStep 是真正干活的类
            env = DisjunctiveGraphJspEnv_singleStep(jps_instance=jsp_instance,
                                                    reward_function_parameters=self.reward_dict,
                                                    default_visualisations=["gantt_console", "graph_console"],
                                                    reward_function='wrk', 
                                                    ability_tr_mm=self.ability_instance[ii_batch][2],  # 运输能力矩阵
                                                    perform_left_shift_if_possible=True,  # 打开左移的机制
                                                    configs=self.args
                                                    )

            self.paral_env_DG.append(copy.deepcopy(env))

            _, _, _, adj, _, machine_fea, tasks_fea, *_ = self.paral_env_DG[-1].reset()
            #收集状态数据
            adj_batch.append(copy.deepcopy(adj))
            tasks_fea_batch.append(copy.deepcopy(tasks_fea))
            machine_fea_batch.append(copy.deepcopy(machine_fea))
        #数据堆叠（Stacking）与返回
        adj_batch = np.array(adj_batch)
        tasks_fea_batch = np.concatenate(tasks_fea_batch, axis=0)
        machine_fea_batch = np.array(machine_fea_batch)

        return adj_batch, machine_fea_batch, tasks_fea_batch
    

    def cal_cur_task_machine_feature(self, task_index, m_mask, all_task_fea):
        """
    TODO
        在 PPO 的“双层决策”结构中，第一层网络选好了“我要做哪个工序（Task）”
        这一步就是紧接着的第二层——“为这个工序选哪台机器”。这个函数负责构建机器的特征矩阵，告诉神经网络每一台候选机器的“优缺点”
        输入：
        task_index: 当前每个环境选了哪个工序（工件）。
        m_mask: 哪些机器是可选的。
        all_task_fea: 当前所有工序的状态特征
        输出：(Batch, Machines, 6) 的张量。每台机器有 6 个特征。
        构建的 6 维机器特征：
        t (Time): 机器加工该工序的时间。
        pt (Energy): 加工能耗。
        Trick: 如果机器不能做该工序（值为0或负），代码用 mean_pt (均值) 填充。这是为了防止神经网络读到 0 值产生误导，保持数值分布的平稳。
        transT (Transport): 运输时间。逻辑: 代码通过 all_task_fea 找到当前工序的前置工序是在哪台机器 (m_id) 做的，然后查 tt_ins 表计算运输时间。如果是第一个工序，运输时间为 0。
        Im (Mask): 机器是否可选（取反）。
        p (Power): 加工功率。再次强调了能耗属性。
        edge: 机器的空间位置编号。
        """
        m_feas = np.zeros((self.batch_size, self.nmachines, 6)) #创建一个容器 m_feas，准备装填 6 个特征。
        all_task_fea = all_task_fea.reshape(self.batch_size, -1, self.args['gcn_input_dim'])
        task_index = task_index.cpu().numpy()
        m_mask = m_mask.cpu().numpy()

        #提取原始数据与计算均值（Batch 循环）
        for i in range(self.batch_size):  # bs个并行环境
            t_ins = self.ability_instance[i][0]  # 时间矩阵
            p_ins = self.ability_instance[i][1]  # 功率矩阵
            tt_ins = self.ability_instance[i][2] # 运输时间矩阵
            edge_ins = self.ability_instance[i][3]  # 机器布局信息
            pt_ins = np.multiply(t_ins, np.abs(p_ins)) # 计算能耗矩阵 = 时间 * 功率

            """对那些不能选择的m，能力值用mean表示，防止0无意义的参数"""
            # 找到不为0的元素并计算均值
            over_zero_elements_t = t_ins[task_index[i]][t_ins[task_index[i]] > 0]  # 当前bs选择了哪个task，找这个task的能力t中大于0的元素
            mean_t = np.mean(over_zero_elements_t )  # 计算该工序所有可用机器的平均时间

            over_zero_elements_pt  = pt_ins[task_index[i]][pt_ins[task_index[i]] > 0]
            mean_pt = np.mean(over_zero_elements_pt ) # 计算该工序所有可用机器的平均能耗

            over_zero_elements_p  = p_ins[task_index[i]][p_ins[task_index[i]] > 0]
            mean_p = np.mean(over_zero_elements_p )  # 计算该工序所有可用机器的平均功率

            #构建 6 大特征（Machines 循环）,定义了 Agent 决策的依据
            for m_index in range(self.nmachines): # 遍历每台机器
                #特征 0：加工时间 (t)
                m_feas[i][m_index][0] = t_ins[task_index[i]][m_index] if t_ins[task_index[i]][m_index] > 0 else mean_t
                #特征1：加工能耗(pt)
                m_feas[i][m_index][1] = pt_ins[task_index[i]][m_index] if pt_ins[task_index[i]][m_index] > 0 else mean_pt
                # 特征 2：运输时间
                if task_index[i] % self.nmachines == 0:  # 如果是该 Job 的第一个工序
                    new_avail_transT = 0
                else:  # 查上一个工序是在哪台机器做的
                    new_avail_transT = tt_ins[int(all_task_fea[i][task_index[i]-1][5])-1][m_index]
                m_feas[i][m_index][2] = new_avail_transT    # 查表计算运输时间：从 prev_m 到当前 m_index

                m_feas[i][m_index][3] = 1 - int(m_mask[i][0][m_index])  # 特征 3：机器可用性
                # 特征 4：加工功率
                m_feas[i][m_index][4] = p_ins[task_index[i]][m_index] if p_ins[task_index[i]][m_index] > 0 else mean_p
                #特征 5：机器位置/ID
                m_feas[i][m_index][5] = np.where(edge_ins == m_index)[0][0] + 1
        return m_feas
    
    """并行环境中的env.step, 输入action更新下一时刻的state和产生reward"""
    def DGFJSPEnv_paral_step(self, joint_actions):
        """
    todo
        作用：在 PPO 算法决定了“选哪个工序”和“选哪台机器”之后，
        这个函数负责把这对联合动作Joint Action真正下发给 16 个并行的仿真环境去执行，并收集执行后的新状态 和 奖励 。
        输入：joint_actions。这是一个列表，长度为 batch_size，每个元素是一对 (task_index, machine_index)。
        输出：打包好的新状态（邻接矩阵、机器特征、工序特征）和环境反馈信息（奖励、是否结束）。
        """
        a_lst = joint_actions  # 接收联合动作列表
        adj_batch_ = []  # 存放新时刻的邻接矩阵
        tasks_fea_batch_ = []  # 存放新时刻的工序特征
        machine_fea_batch_ = [] # 存放新时刻的机器特征
        self.oenv_info = []  # 存放每一步的奖励和Done标志，记得初始化清空！
        #并行执行循环
        for l_batch in range(self.batch_size):
            # 解包动作：把 PPO 输出的动作拆分成“工序”和“机器”
            select_task_index = a_lst[l_batch][0]  # 每一个batch中，当前选择的task_index是list的第一个
            select_mch_index = a_lst[l_batch][1]  # 每一个batch中，当前选择的m_id是list的第二个
            joint_action = [select_task_index,select_mch_index]

            # TODO ZZO
            # 环境步进（传进环境模块得到新状态的东西）
            #_, r, o_done, _, rmk, ridle, renergy_m, renergy_transM, _, _, \
                #adj_, _, machine_fea_, tasks_fea_ = self.paral_env_DG[l_batch].step(joint_action)  # env.step
            _, r, o_done, _, rmk, r_carbon_m, r_carbon_agv, r_tou, _, _, \
                adj_, _, machine_fea_, tasks_fea_ = self.paral_env_DG[l_batch].step(joint_action)

            # 安全检查 (Debug 用)
            if self.ability_instance[l_batch][0][select_task_index][select_mch_index] < 0:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"============= 'DGFJSPEnv_paral_step' occur error: chose Minus: t={self.ability_instance[l_batch][0][select_task_index][select_mch_index]}, p= {self.ability_instance[l_batch][1][select_task_index][select_mch_index]}")
            # 收集新状态
            adj_batch_.append(copy.deepcopy(adj_))
            tasks_fea_batch_.append(copy.deepcopy(tasks_fea_))
            machine_fea_batch_.append(copy.deepcopy(machine_fea_))

            # TODO ZZO 奖励动态缩放
            # 1. 组装原始奖励向量：[完工时间, 空闲时间, 加工能耗, 运输能耗]
            #r_vector = np.array([rmk, ridle, renergy_m, renergy_transM])
            r_vector = np.array([rmk, r_carbon_m, r_carbon_agv, r_tou])

            # 2. 调用 RewardScaling 进行归一化
            r_vector_scaling = self.paral_Rscaling_instance[l_batch](r_vector)
            # oenv_step_info = [总奖励, Done, 缩放后的完工, 缩放后的空闲, 缩放后的能耗, 缩放后的运输能耗]
            oenv_step_info = [r, o_done, r_vector_scaling[0], r_vector_scaling[1], r_vector_scaling[2], r_vector_scaling[3]]
            self.oenv_info.append(oenv_step_info)
        #数据堆叠与返回
        adj_batch_ = np.array(adj_batch_)
        tasks_fea_batch_ = np.concatenate(tasks_fea_batch_, axis=0)
        machine_fea_batch_ = np.array(machine_fea_batch_)
        return adj_batch_, self.oenv_info, machine_fea_batch_, tasks_fea_batch_  
    
    def reset_data(self):
        """
        作用： 打扫战场
        """
        self.paral_env_DG = []  # 记录并行的选择DG的环境
        self.oenv_info = []  # 记录每一个batch的reward, done, r_t, r_idle, r_energy_m, r_energy_transM
        

    

    

