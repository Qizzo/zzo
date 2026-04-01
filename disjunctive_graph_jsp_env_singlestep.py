import copy
import random
import gym
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List, Union, Dict, Callable
from trainer.DGenv_func import find_transportT, find_max_arrivaTime_for_currentNode, calculate_idle_t_for_each_machine, calculate_tou_cost_by_slicing
from graph_jsp_env.disjunctive_graph_jsp_visualizer import DisjunctiveGraphJspVisualizer
from graph_jsp_env.disjunctive_graph_logger import log
from algorithm.ppo_trick import RewardScaling


class Variant:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class DisjunctiveGraphJspEnv_singleStep(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'console']}
    """
    输入核心数据：
    jps_instance: 加工时间表和能耗表。
    ability_tr_mm: 运输时间矩阵（Machines * Machines）。用于计算跨机器的运输时间。
    configs: 一个包含所有超参数的字典（来自 parameters.py）。
    强化学习配置：
    reward_function: 奖励函数的名字，默认是 'wrk'（作者自定义的）。
    normalize_observation_space: 是否归一化观测值（0-1之间）。
    action_mode: 动作模式，默认 'task'（选工序）。
    动作策略：
    perform_left_shift_if_possible: 插空策略。如果为 True，当选定一个机器时，插入式。
    """

    def __init__(self,
                 jps_instance: np.ndarray = None, *,  # 代表初始化样本的加工时间和加工功率/能耗的能力矩阵 = 2 * task * m

                 reward_function='nasuta',  # 反馈函数自定义为作者的名字了nasuta
                 custom_reward_function: Callable = None,  # Python中能被调用（called）的东西就是callable，例如function
                 reward_function_parameters: Dict = None,  # Dict字典，无序的、可变的序列，它的元素以“键值对（key-value）”的形式存储。

                 # flat() 方法会按照一个可指定的深度递归遍历数组，并将所有元素与遍历到的子数组中的元素合并为一个新数组返回。
                 normalize_observation_space: bool = True,  # 观测空间，为真，value都是0-1之间
                 flat_observation_space: bool = True,  # 观测空间，为真，扁平化：遍历数组，并组合成新数组返回，否则是矩阵
                 dtype: str = "float32",  # 观测空间的数据类型

                 # parameters for actions
                 action_mode: str = "task",  # 默认动作是子任务，也可以job or node
                 env_transform: str = None,  #
                 perform_left_shift_if_possible: bool = True,
                 # 基于最大完工时间，一个任务（step method期间）加在2个任务之间不改变这两个的开始和结束时间，那就true加进去；否则排在后边；插空进去，时间更短

                 # parameters for rendering
                 c_map: str = "rainbow",  # 渲染画图的matplotlib colormap
                 dummy_task_color="tab:gray",  # 表示源和目的节点dummy node的颜色
                 default_visualisations: List[str] = None,
                 # 渲染时候的可视化，["gantt_window", "gantt_console", "graph_window", "graph_console"]
                 visualizer_kwargs: dict = None,  # 额外的可视化参数
                 verbose: int = 0,  # 决定是否有信息打印在console，0无1重要2全部

                 # wrk 传入信息: 运输时间m*m + 被选的加工能耗j*m
                 ability_tr_mm,  # 初始边的权重，即m到m的运输t的表格 m * m
                 # ability_p2: np.ndarray = np.ones((1, configs.n_machine))
                 ability_p2: np.ndarray = None,
                 # ability_e_tm # 已知的对应被选择m的能耗  j * m

                 configs: dict = None
                 ):

        self.configs = Variant(**configs)  # 将字典转换为Variant实例

        # Note: None-fields will be populated in the 'load_instance' method  没有字段将在'load_instance'方法中填充，都是传进来的？
        # load_instance方法中的自定义变量
        self.size = None
        self.n_jobs = None
        self.n_machines = None
        self.total_tasks_without_dummies = None
        self.total_tasks = None
        self.src_task = None
        self.sink_task = None
        self.longest_processing_time = None
        self.observation_space_shape = None
        self.scaling_divisor = None
        self.machine_colors = None
        self.G = None
        self.machine_routes = None

        # 共有三个地方： 初始化init建立为None，load_instance最初的值为0，step结束记录当前的值，reward函数里用来计算
        self.makespan_previous_step = None
        self.total_e1_previous_step = None
        # self.e2_previous_step = None
        self.trans_t_previous_step = None
        self.idle_t_previous_step = None

        """
        全局统计变量初始化，这些变量用来记录“当前这一步”或者“至今为止”的各种累积指标，用于计算 step 中的奖励。
        作用：这些变量就像是“记分牌”。每走一步 (step)，环境都会更新这些值，然后算出差值作为 Reward 返回给神经网络。
        """
        # 全局变量，记录某些随step变化的值
        self.energy_transport = [0, 0]
        self.exist_list = []

        self.total_e1_this_step = 0  # 至今为止已分配的加工能耗总和
        self.idle_t_this_step = 0  # 至今为止已产生的空闲时间总和
        self.trans_t_this_step = 0  # 至今为止产生的运输时间总和

        self.reward_list = [0, 0, 0, 0, 0]  # 记录4个分项奖励 + 总Cost

        # 记录“累计误差”（用于 Reward 计算公式：上一时刻 - 当前时刻）
        self.reward_t = 0  # 完工时间累计
        self.reward_e1 = 0  # 加工能耗累计
        self.reward_e2 = 0  # 运输能耗累计
        self.reward_idle_t = 0  # 空闲时间累计

        # 记录已经被选择的task，记得清0
        self.selected_action = []
        self.selected_action_machine = []
        self.machines_fea = None  # 初始化一个m节点的特征矩阵
        # 记录每一步选择action后新增的空闲时间idle_t
        self.it_s = []  # 后边在load_instance中变为全0列表

        '''
        奖励函数配置
        '''
        if reward_function not in ['nasuta', 'zhang', 'graph-tassel', 'samsonov', 'zero', 'custom', 'wrk']:  # 判断反馈函数有无
            raise ValueError(f"only 'nasuta', 'zhang', 'graph-tassel', 'samsonov', 'zero', 'custom' "
                             f"are valid arguments for 'reward_function'. {reward_function} is not.")
        if reward_function == 'custom' and custom_reward_function is None:
            raise ValueError(f"if 'reward_function' is 'custom', 'custom_reward_function' must be specified.")

        self.reward_function = reward_function  # reward_function='nasuta',  形参，是传进来的变量；付给init函数中的操作
        self.custom_reward_function = custom_reward_function

        # default reward function params  确定self.reward_function_parameters是什么值
        if reward_function_parameters is None:  # 形参，是个字典
            if reward_function == 'nasuta':
                self.reward_function_parameters = {
                    'scaling_divisor': 1.0  # scaling_divisor缩放因子，放的什么?我感觉是折扣率
                }
            elif reward_function == 'zhang':
                self.reward_function_parameters = {}
            elif reward_function == 'samsonov':
                self.reward_function_parameters = {
                    'gamma': 1.025,
                    't_opt': None,
                }
            elif reward_function == 'graph-tassel':
                self.reward_function_parameters = {
                }
            elif reward_function == 'zero':
                self.reward_function_parameters = {}
            elif reward_function == 'custom':
                self.reward_function_parameters = {}
            elif reward_function == 'wrk':  # 增加自己的reward_function == 'wrk':
                self.reward_function_parameters = {
                    # 放在外边定义了，这里先注释！
                    # 'scaling_divisor': 20.0  # scaling_divisor缩放因子
                }
            else:
                raise ValueError('something went wrong. This error should not be called.')
        else:
            self.reward_function_parameters = reward_function_parameters

        # observation settings   从形参复制到类中的变量，观测空间
        self.normalize_observation_space = normalize_observation_space  # 观测空间是否归一化
        self.flat_observation_space = flat_observation_space  # 观测空间是否扁平化
        self.dtype = dtype  # 观测空间数据类型

        # action setting   从形参复制到类中的变量，动作空间
        self.perform_left_shift_if_possible = perform_left_shift_if_possible  # 是否插空，在两个节点之间：不影响开始和结束时间
        if action_mode not in ['task', 'job']:
            raise ValueError(f"only 'task' and 'job' are valid arguments for 'action_mode'. {action_mode} is not.")
        self.action_mode = action_mode  # 按照zhang的论文，动作就是每一个task，每次选一个o来在schedule表上边分配，从而更新析取图

        if env_transform not in [None, 'mask']:  # env_transform 环境转移？是析取图的方向的改变？没有给出定义！！！
            raise ValueError(f"only `None` and 'mask' are valid arguments for 'action_mode'. {action_mode} is not.")
        self.env_transform = env_transform

        # rendering settings 从形参复制到类中的变量，渲染
        self.c_map = c_map  # matplotlib 画图的配色
        if default_visualisations is None:  # default_visualisations 要显示的东西是什么：甘特控制台 + 甘特窗口 + 图控制台 + 图控制窗口（我run之后console和弹出的界面，4个）
            self.default_visualisations = ["gantt_console", "gantt_window", "graph_console", "graph_window"]
        else:
            self.default_visualisations = default_visualisations
        if visualizer_kwargs is None:  # 额外可视化参数
            visualizer_kwargs = {}
        self.visualizer = DisjunctiveGraphJspVisualizer(**visualizer_kwargs)

        """
        虚拟节点设置  
        在图结构中，起始点 (Source) 和 终点 (Sink) 是虚拟的，不属于任何机器，所以给它们分配了特殊的 ID。
        """
        # values for dummy tasks nedded for the graph structure  画图所需的虚拟节点，source和sink的节点。要改成needed吧，hhh
        self.dummy_task_machine = -2  # 开始和结束节点的m+id
        self.dummy_task_job = -1  # 开始和结束节点隶属哪个job——id
        self.dummy_task_color = dummy_task_color  # 开始和结束节点的颜色，默认灰色

        self.unscheduled_task_m_id = -1  # -1代表着没有被分配m的task
        self.unscheduled_task_color = dummy_task_color  # 一样是灰色
        self.unscheduled_task_duration = 0  # 一样是灰色

        self.verbose = verbose

        """
        数据加载
        """
        self.instance_transT = ability_tr_mm  # 1. 保存运输时间矩阵
        self.jsp_instance = jps_instance  # 2. 保存加工数据
        self.instance_processingEnergy = jps_instance[0] * jps_instance[
            1]  # 3. 预先计算加工能耗 (Power * Time).jps_instance[0] 是时间 t，[1] 是功率 p

        self.reward_random_weight = None
        # 4. 加载实例，构建析取图
        self.load_instance(jsp_instance=self.jsp_instance)
        # 5. 初始化待机功率 (默认全1)
        self.instance_p2 = np.ones((1, self.n_machines))

        #Todo zzo 初始化电费记录变量
        self.instance_power = jps_instance[1]  # 记录真实的加工功率矩阵
        self.tou_cost_this_step = 0.0  # 当前累计电费
        self.tou_cost_previous_step = 0.0  # 上一步的累计电费
    # 建立Disjunctive Graph里边的节点和边
    def load_instance(self, jsp_instance: np.ndarray, *, reward_function_parameters: Dict = None) -> None:

        _, tasks, n_machines = jsp_instance.shape  # 计算有多少 Job，多少 Machine

        n_jobs = tasks // n_machines  # 返回整数部分

        self.size = (n_jobs, n_machines)  # 自定义size = 上步获取到的信息
        # 设置各种计数器
        self.n_jobs = n_jobs  # job个数
        self.n_machines = n_machines  # machine个数
        self.total_tasks_without_dummies = n_jobs * n_machines  # 实际工序数
        self.total_tasks = n_jobs * n_machines + 2  # 全图几个点，算上source和sink
        self.src_task = 0  # source节点的编号
        self.sink_task = self.total_tasks - 1  # sink节点的编号
        # 3. 初始化机器特征矩阵
        self.machines_fea = np.zeros((self.n_machines, 8))
        # 初始化一个列表，长度等于实际工序数。
        self.it_s = [0] * self.total_tasks_without_dummies
        # 定义动作空间,如果模式是选工序（默认）。定义动作空间为 离散空间 ，大小为实际工序数.Agent 输出0-实际工序数的整数，代表选择哪个工序。
        if self.action_mode == 'task':
            self.action_space = gym.spaces.Discrete(self.total_tasks_without_dummies)
        else:  # 如果是选 Job 模式，动作空间就是 Job 数（例如 6）。
            self.action_space = gym.spaces.Discrete(self.n_jobs)

        # 定义观测空间,这部分告诉 Gym 和神经网络：“我的状态（State）长什么样”。
        if self.normalize_observation_space:  # 定义观测矩阵的形状（归一化模式）：行数：工序数。列数：邻接矩阵的一行 + 6 (机器One-hot编码) + 1 (加工时间) 。
            self.observation_space_shape = (self.total_tasks_without_dummies,
                                            self.total_tasks_without_dummies + self.n_machines + 1)
        else:  # 如果不归一化，列数变少（机器ID直接用整数表示，不One-hot）
            self.observation_space_shape = (self.total_tasks_without_dummies, self.total_tasks_without_dummies + 2)

        if self.flat_observation_space:  # 观测空间扁平化
            a, b = self.observation_space_shape  # 维度读取：行+元素
            self.observation_space_shape = (a * b,)  # 一维的，所以shape维度是相乘，平铺开

        # 在自定义观测空间：上边定义了观测空间的维度，这里定义数据范围
        if self.env_transform is None:  # 环境转移标志，是一个str字符串格式 ：默认是None
            self.observation_space = gym.spaces.Box(  # 连续，观测空间
                low=0.0,
                # high=1.0 if self.normalize_observation_space else jsp_instance.max(),  #这里表示我需要状态空间的归一化，低0高1
                high=1.0,  # self.normalize_observation_space我都没改过，一直是True
                shape=self.observation_space_shape,  # 维度
                dtype=self.dtype  # 数据类型
            )
        elif self.env_transform == 'mask':
            self.observation_space = gym.spaces.Dict(
                {  # 观测空间变成字典了：动作标记（low0，high1，12维度，int）+观测（low0，high1，（a，b）的二位维度，数据类型）
                    "action_mask": gym.spaces.Box(0, 1, shape=(self.action_space.n,), dtype=np.int32),
                    # eg，action_space = 12个节点
                    "observations": gym.spaces.Box(
                        low=0.0,
                        # high=1.0 if self.normalize_observation_space else jsp_instance.max(),
                        high=1.0,  # self.normalize_observation_space我都没改过，一直是True
                        shape=self.observation_space_shape,
                        dtype=self.dtype)
                })
        else:
            raise NotImplementedError(f"'{self.env_transform}' is not supported.")

        # 下边都是画图相关的
        c_map = plt.cm.get_cmap(self.c_map)  # select the desired cmap   返回c_map对象，将0-1数值映射成颜色
        arr = np.linspace(0, 1, n_machines,
                          dtype=self.dtype)  # create a list with numbers from 0 to 1 with n items 这个数组的作用是为每个机器分配一个数值，这个数值的范围是0到1之间的连续值
        self.machine_colors = {m_id: c_map(val) for m_id, val in
                               enumerate(arr)}  # 使用字典推导式为每个机器分配一个颜色。遍历`arr`数组中的每个元素，将其作为参数传递给颜色

        """
        初始化一个有向图！
        然后初始化一个最关键的machine_route表示同一个m上的加工顺序，代表了不同的可行解！
        """
        self.G = nx.DiGraph()  # G是一个有向图
        # 初始化 机器路径字典。这是调度结果的容器,结构：{0: [], 1: [], 2: [], ...}.将来调度时，比如把工序 5 分给机器 0，这里就会变成 {0: [5], ...}。
        self.machine_routes = {m_id: np.array([], dtype=int) for m_id in range(n_machines)}  # 创建字典，给每个设备记录其最终的工序路径

        """
        节点与边的构建
        """
        # 创建 Source 起始节点
        self.G.add_node(
            self.src_task,  # 源节点
            pos=(-2, int(-n_jobs * 0.5)),  # 节点在绘图中的位置，这里的pos属性是一个元组，表示节点的x和y坐标
            duration=0,  # 虚拟节点，耗时必须为 0
            machine=self.dummy_task_machine,  # 源节点机器id：-1
            scheduled=True,  # 是否被调度：初始肯定被调度了
            color=self.dummy_task_color,  # 源节点颜色
            job=self.dummy_task_job,  # 源节点job的id：-1
            start_time=0,
            finish_time=0
        )
        """
        循环创建工序节点与内部连边,遍历了样本里的每一个格子（Job × Machine）
        """
        task_id = 0  # 计数器归零

        for i in range(n_jobs):  # 遍历每一个 Job (比如 0 到 5)
            for j in range(n_machines):  # 遍历每一个工序步骤 (比如 0 到 5)
                task_id += 1  # ID 从 1 开始递增 (1, 2, ..., 36)。顺序是：Job0的所有工序 -> Job1的所有工序...
                # 在初始化阶段，我们假装什么都不知道。不知道给哪台机器（-1），也不知道要做多久（0）。这些信息会在 step 函数调度时才填进去。
                m_id = self.unscheduled_task_m_id  # 取值 -1
                dur = self.unscheduled_task_duration  # 取值 0
                # 添加工序节点
                self.G.add_node(  # 因为是循环，所以会每一个task都建立好了，从1开始的，0是源节点
                    task_id,
                    pos=(j, -i),
                    color=self.unscheduled_task_color,
                    duration=dur,
                    scheduled=False,  # 【关键】初始全是未调度
                    machine=m_id,
                    job=i,  # 记录它属于哪个 Job
                    start_time=None,  # start_time理论上应该一直在更新，根据不同的调度顺序，运输时间也不一样
                    finish_time=None
                )

                """
                添加连边,这一段逻辑分三种情况，把孤立的点连成线。
                情况一：Job 的第一个任务
                情况二：Job 的最后一个任务
                情况三：中间的任务
                """
                if j == 0:  # 情况一：Job 的第一个任务
                    self.G.add_edge(
                        self.src_task, task_id,  # 连边：Source -> 当前任务
                        job_edge=True,  # 标记为“工艺边”
                        weight=self.G.nodes[self.src_task]['duration'],  # 权重 = Source的时长 (0)
                        nweight=-self.G.nodes[self.src_task]['duration']  # 负权重(用于关键路径计算)
                    )
                elif j == n_machines - 1:  # 情况二：Job 的最后一个任务

                    # 计算运输时间
                    # 查表：上一个任务(task_id-1) -> 当前任务(task_id) 的潜在运输时间
                    transport_t = find_transportT(self.G, (task_id - 1), task_id, self.instance_transT, self.configs)
                    # print("load instance末位 self.energy_transport",self.energy_transport,j,self.exist_list)

                    self.G.add_edge(
                        task_id - 1, task_id,  # 连边：上一个 -> 当前
                        job_edge=True,
                        # 权重 = 1 + 运输时间
                        # 为什么是 1？因为上一个节点的 duration 初始是 0，作者可能为了防止权重为0加了个基数 1
                        weight=1 + transport_t,
                        nweight=-(1 + transport_t)
                    )
                else:  # 情况三：中间的任务

                    transport_t = find_transportT(self.G, (task_id - 1), task_id, self.instance_transT, self.configs)

                    self.G.add_edge(
                        task_id - 1, task_id,
                        job_edge=True,
                        weight=1 + transport_t,
                        nweight=-(1 + transport_t)
                    )
        self.G.add_node(  # 添加最后的节点
            self.sink_task,
            pos=(n_machines + 1, int(-n_jobs * 0.5)),
            color=self.dummy_task_color,
            duration=0,
            machine=self.dummy_task_machine,
            job=self.dummy_task_job,
            scheduled=True,
            start_time=None,
            finish_time=None
        )
        # 添加每个job的最后一个task的边到结束节点
        for task_id in range(n_machines, self.total_tasks, n_machines):
            self.G.add_edge(
                task_id, self.sink_task,
                job_edge=True,
                weight=1
            )
        """
        初始指标计算与基准线
        图建好了，但在强化学习开始前，必须先算一个**“初始分”**。因为 PPO 的奖励是基于“这一步比上一步好了多少”来算的，第一步需要一个参考系
        """
        initial_makespan = nx.dag_longest_path_length(self.G)  # 使用 NetworkX 自带的动态规划算法找最长路径
        self.makespan_previous_step = initial_makespan

        if self.reward_function == 'wrk':
            """初始化的时候：load_isntance只有在init和reset才会使用，此时都是赋值初始化init/prev的好时间点"""
            if_schedule_lst_init = [0] * self.total_tasks_without_dummies  # 全0列表
            ft_lst_init = [0.0] * self.total_tasks_without_dummies  # 全0列表  完工时间
            st_lst_init = [0.0] * self.total_tasks_without_dummies  # 全0列表  开始时间
            pt_lst_init = [0.0] * self.total_tasks_without_dummies  # 全0列表  加工能耗PE
            st_idea_init, ft_idea_init, pt_idea_init = self.estiamte_st_ft_pt_eachStep_noTransT(
                current_ft=np.array(ft_lst_init).reshape(self.n_jobs, self.n_machines),
                current_st=np.array(st_lst_init).reshape(self.n_jobs, self.n_machines),
                current_pt=np.array(pt_lst_init).reshape(self.n_jobs, self.n_machines),
                if_schedule=np.array(if_schedule_lst_init).reshape(self.n_jobs, self.n_machines))
            # 1. 更新 Makespan 基准：用估算的理想完工时间
            initial_makespan = np.amax(
                ft_idea_init.flatten())  # task个元素，已经被flatten！找最大，就是预估的整体完工时间   amax针对一维数组找最大高效，否则就用通用的mean了
            # initial_makespan = 0
            # self.makespan_previous_step = initial_makespan
            self.makespan_previous_step = initial_makespan
            # 2. 更新能耗基准：用估算的理想总能耗
            self.total_e1_previous_step = np.sum(pt_idea_init.flatten())
            # 3. 运输和空闲基准：设为 0 (理想状态下没有这俩)
            self.trans_t_previous_step = 0
            self.idle_t_previous_step = 0

            # TODO zzo 新增
            self.tou_cost_this_step = 0.0
            self.tou_cost_previous_step = 0.0
        """
        这里记录了reward的缩放因子，不能删掉！！！
        """
        if reward_function_parameters is not None:
            if self.verbose > 1:
                log.info(f"updating reward_function_parameters from '{self.reward_function_parameters}' "
                         f"to '{reward_function_parameters}'")
            self.reward_function_parameters = reward_function_parameters  # reward的缩放因子的赋值！！！！

    def step(self, joint_action: list) -> (np.ndarray, float, bool, dict):
        """
    Todo
        接收指令：Agent 说“把工序 A 给机器 M 做”。
        执行指令：环境更新图结构，算出工序 A 的开始和结束时间。
        计算后果：看看完工时间、能耗有没有增加。
        反馈结果：告诉 Agent 新的状态（State）和得分（Reward）。
        输入 joint_action：这是一个包含两个元素的列表 [task_index, machine_index]。
        task_index: 神经网络选出的工序索引（0 ~ 35）。machine_index: 神经网络选出的机器索引（0 ~ 5）。
        """
        #   Todo 第一部分：解析动作与记录
        # 1. 记录历史动作 (用于调试或画图)
        self.selected_action.append(joint_action[0])  # 记录所有被选择的task的index
        self.selected_action_machine.append(joint_action[1])  # 记录所有被选择的machine的index

        info = {
            'action': joint_action[0]
        }

        # Todo 第二部分：核心调度执行
        if self.action_mode == 'task':
            task_id = joint_action[0] + 1  # 1. 转换 ID：神经网络输出的 index 是从 0 开始的，但图里的节点 ID 是从 1 开始的
            m_id = joint_action[1]  # m的id就是直接从0开始吧

            dur = self.jsp_instance[0][task_id - 1][m_id]  # 2. 获取该工序在该机器上的加工时间 (查表)
            # 3. 执行调度
            # 调用schedule_task 这个函数，它会确定当前工序的开始时间和结束时间（考虑了前置任务约束、机器空闲时间、插空逻辑等），并在图中添加相应的边。
            if self.verbose > 1:
                log.info(f"handling action={joint_action[0]} (Task {task_id})")
            info = {
                **info,
                **self._schedule_task(task_id=task_id,
                                      m_id=m_id,
                                      dur=dur)
            }
        else:
            pass

        #  Todo 第三部分：判断是否结束
        total_length = sum([len(route) for m_id, route in self.machine_routes.items()])  # 计算所有机器上已排任务的总数
        # 如果已排任务数 == 总工序数，说明全部做完了
        done = total_length == self.total_tasks_without_dummies

        # Todo 第四部分：计算当前步骤的各项指标
        # 计算当前所有机器队列中，计算 Makespan ，调用辅助函数 Todo max_finish_time_in_machineRoute
        makespan = self.max_finish_time_in_machineRoute(self.G, self.machine_routes)
        # 计算加工能耗
        task_index = joint_action[0]
        task_id = joint_action[0] + 1  # 当前的任务编号
        m_id = joint_action[1]  # 当前m的编号 = m的索引，m从0开始的
        e1_current = self.instance_processingEnergy[task_index][m_id]  # 查表：当前任务在当前机器上的能耗
        self.total_e1_this_step = self.total_e1_this_step + e1_current  # 累加到本步的总能耗中
        # 计算空闲时间，调用外部函数 Todo calculate_idle_t_for_each_machine 计算当前所有机器的总空闲时间。
        self.idle_t_this_step = calculate_idle_t_for_each_machine(self.G, self.machine_routes, self.instance_p2)

        # TOdo zzo 新增：计算刚排上去的这道工序产生的真实切片电费
        # ====================================================
        curr_task_node = self.G.nodes[task_index + 1]
        st = curr_task_node['start_time']
        ft = curr_task_node['finish_time']
        p_process = self.instance_power[task_index][m_id]
        p_idle = self.instance_p2[0][m_id]

        # 1. 算加工电费
        task_tou = calculate_tou_cost_by_slicing(st, ft, p_process, self.configs.tou_price_table)

        # 2. 算待机电费 (找出这台机器上一个工序的结束时间作为空闲起点)
        route = self.machine_routes[m_id]
        gap_start = 0.0
        if len(route) > 1:
            prev_task_node = self.G.nodes[route[-2]]
            gap_start = prev_task_node['finish_time']
        idle_tou = calculate_tou_cost_by_slicing(gap_start, st, p_idle, self.configs.tou_price_table)

        # 3. 累加到本回合总电费中
        self.tou_cost_this_step += (task_tou + idle_tou)
        # TOdo zzo  ====================================================

        # 计算运输时间
        if joint_action[0] % self.n_machines == 0:
            new_avail_transT = 0  # 如果是 Job 的第一个任务，没有运输
        else:  # 查表：上一个任务 -> 当前任务 的运输时间
            new_avail_transT = find_transportT(self.G, joint_action[0], (joint_action[0] + 1), self.instance_transT,
                                               self.configs)  # action+1=当前的task_id，所以action=上一个的task_id
        self.trans_t_this_step += new_avail_transT

        # Todo 第五部分：获取第二部分调度后的状态
        # 获取新状态，调用 _state_array 获取最新的图结构 (adj) 和特征向量 (fea)
        state, ft_s, it_s, adj_wrk, tasks_fea, machine_fea, tasks_fea_1101, ft_real_estimated, pt_real_estimated = self._state_array()
        # Todo 第六部分：计算奖励。
        # 在计算 Reward 之前，你用 _state_array 返回的预估值覆盖了部分指标，用于 Reward 计算
        makespan = np.amax(ft_real_estimated)
        self.total_e1_this_step = np.sum(pt_real_estimated)
        # 计算这一步的 Reward。get_reward 函数（特别是 wrk 模式）是基于 "上一步指标 - 这一步指标" 的差值来计算的。
        reward, r_t, r_idle, r_pt, r_transT = self.get_reward(
            state=state,
            done=done,  # 传进去这些形参，作为输入
            info=info,
            makespan_this_step=makespan
        )
        # Todo 第七部分：更新 "上一步" 变量
        # 为了给下一次 step 计算 Delta Reward 做准备，需要把当前的指标保存为 "previous"。
        self.makespan_previous_step = makespan
        self.total_e1_previous_step = self.total_e1_this_step  # 上次的加工能耗之和
        self.trans_t_previous_step = self.trans_t_this_step  # 至今为止的运输时间t
        self.idle_t_previous_step = self.idle_t_this_step  # 上次记录当前的idle时间

        # TODO zzo 新增
        self.tou_cost_previous_step = self.tou_cost_this_step

        # Todo 第八部分：结束处理
        # 如果 done 为 True，做一些清理和记录工作。
        if done:
            try:
                cycle = nx.find_cycle(self.G, source=self.src_task)
                log.critical(f"CYCLE DETECTED cycle: {cycle}")
                raise RuntimeError(f"CYCLE DETECTED cycle: {cycle}")
            except nx.exception.NetworkXNoCycle:
                pass
            info["makespan"] = makespan
            info["gantt_df"] = self.network_as_dataframe()
            if self.verbose > 0:
                log.info(f"makespan: {makespan}")
            # 全局变量，记录某些随step变化的值
            self.energy_transport = [0, 0]  # 用来记录运输设备的能耗，这里我直接累加运输设备的总的时间！(运了几次，累加时间)：同一个设备上边的不算次数
            self.exist_list = []  # 用来记录已经查到过的平行边的传输时间t,防止多查
            self.total_e1_this_step = 0  # 用来记录至今为止选了的设备的能耗和
            self.idle_t_this_step = 0  # 用来记录当前的idle时间（machine_route里边已经分配的m的空闲时间）
            self.trans_t_this_step = 0  # 记录至今为止的运输时间，清0
            self.reward_list = [0, 0, 0, 0, 0]  # 对应记录当前的4个reward+总cost
            # 共有2次：init里边初始化为0，reward函数中进行累加；这里清0
            self.reward_t = 0  # 传输时间的累计误差
            self.reward_e1 = 0  # 加工能耗的累计误差
            self.reward_e2 = 0  # 运输能耗的累计误差
            self.reward_idle_t = 0  # 空闲时间的累计误差
            # 记录被选择的action
            self.selected_action = []
            self.selected_action_machine = []
            self.machines_fea = np.zeros((self.n_machines, 8))  # done之后的m节点的状态就清0，reset也清零，后边重新调用__state_array会重新赋值
            self.it_s = [0] * self.total_tasks_without_dummies  # 清0之前先存在某个变量，不影响

        return state, reward, done, info, r_t, r_idle, r_pt, r_transT, ft_s, it_s, adj_wrk, tasks_fea, machine_fea, tasks_fea_1101  # 每一步都会返回（state + reward + done + info），

    def get_reward(self, state: np.ndarray, done: bool, info: Dict, makespan_this_step: float):
        """
    Todo
        这个函数的作用是根据环境当前的配置（reward_function），
        计算并返回给 Agent 的Reward。它就像一个分发器，根据你选择的奖励策略（比如 'nasuta', 'zhang', 还是你自己写的 'wrk'），执行不同的计算逻辑。
        输入参数：
        state (np.ndarray): 当前的环境状态）。done (bool): 标志位，表示当前 Episode 是否结束（所有工序都排完了）。
        info (Dict): 包含额外信息，函数内部也会往里写信息。makespan_this_step (float): 当前这一步计算出来的最大完工时间。
        返回值:
        step_total_r: 给 RL 算法优化的总奖励。
        r_t, r_idle, r_pt, r_transT: 这 4 个是为了记录日志用的，方便你观察模型到底是在优化时间，还是在优化能耗。
        """
        info['reward_function'] = self.reward_function
        reward_function_parameters = self.reward_function_parameters
        # Todo 'nasuta' (稀疏奖励):
        # 逻辑：只有当 done 为 True（做完了）时，才返回 -makespan（负的完工时间）。没做完时奖励为 0。
        # 目的：让 Agent 追求最终结果最小化，中间过程不给反馈。
        if self.reward_function == 'nasuta':
            if not done:
                return 0.0
            else:
                return - makespan_this_step / reward_function_parameters['scaling_divisor']
        # Todo 'zhang'(密集 / 增量奖励):
        # 逻辑：Last_Makespan - Current_Makespan。
        # 目的：每一步都给反馈。如果这一步让完工时间变短了（比如插空成功），给正奖励；变长了，给负奖励。
        elif self.reward_function == 'zhang':
            return 1.0 * self.makespan_previous_step - makespan_this_step
        # Todo 'graph-tassel': 基于机器利用率（甘特图填充面积）来计算。
        elif self.reward_function == 'graph-tassel':
            max_finish_time = 0.0
            total_filled_area = 0.0
            at_least_one_scheduled_node = False
            for m, m_route in self.machine_routes.items():
                if len(m_route):
                    m_ft = self.G.nodes[m_route[-1]]["finish_time"]
                    if m_ft >= max_finish_time:
                        max_finish_time = m_ft
                    m_filled_area = sum([self.G.nodes[task]["duration"] for task in m_route])
                    total_filled_area += m_filled_area
                    at_least_one_scheduled_node = True
                else:
                    pass
            if not at_least_one_scheduled_node:  # needed to avoid division through zero after invalid first action
                return 0.0
            total_gantt_area = max_finish_time * self.n_machines
            machine_utilization = total_filled_area / total_gantt_area  # always between 0 and 1
            return machine_utilization
        # Todo 'samsonov': 基于指数衰减的奖励。
        elif self.reward_function == 'samsonov':
            if not done:
                return 0.0
            else:
                gamma = reward_function_parameters['gamma']
                if reward_function_parameters['t_opt'] is None:
                    raise ValueError(f"'t_opt' must be provided inside 'reward_function_parameters' for the samsonov "
                                     f"reward function.")
                t_opt = reward_function_parameters['t_opt']
                return 1000 * gamma ** t_opt / gamma ** makespan_this_step

        elif self.reward_function == 'zero':
            return 0.0

        elif self.reward_function == 'custom':
            return self.custom_reward_function(
                state,
                done,
                info,
                self.G,
                makespan_this_step,
                self.makespan_previous_step,
                **reward_function_parameters
            )
        elif self.reward_function == 'wrk':
            return self.wrk_reward_function(done,
                                            makespan_this_step)



    # -----------------------------------------------------------
    def wrk_reward_function(self, done: bool, makespan_this_step: float):
        """
        作用：基于“增量”思想设计的：不仅仅看当前状态好不好，更看这一步操作比上一步好（正奖励）还是差（负惩罚）
        核心修改：将原来的(时间, 空闲, 加工, 运输) 4 槽位，完美替换为 (时间, 机器碳排, 物流碳排, 分时电费)。
        """
        # A. 最大完工时间增量 (Makespan) - 保持不变
        r_t = 1.0 * self.makespan_previous_step - makespan_this_step
        self.reward_t = self.reward_t + r_t
        self.reward_list[0] = self.reward_t

        # B. 机器碳排放增量 (Machine Carbon)
        # 物理公式：总耗电量 = 加工耗电(total_e1) + 待机耗电(idle_t)
        # 奖励公式：上一步总碳排 - 这一步总碳排
        carbon_m_prev = (self.total_e1_previous_step + self.idle_t_previous_step) * self.configs.carbon_grid
        carbon_m_curr = (self.total_e1_this_step + self.idle_t_this_step) * self.configs.carbon_grid
        r_carbon_m = carbon_m_prev - carbon_m_curr

        self.reward_e1 = self.reward_e1 + r_carbon_m  # 借用旧变量记录碳排曲线
        self.reward_list[1] = self.reward_e1

        # C. 运输碳排放增量 (AGV Carbon)
        # 物理公式：原代码用 trans_t 代指运输耗电量
        carbon_agv_prev = self.trans_t_previous_step * self.configs.carbon_agv
        carbon_agv_curr = self.trans_t_this_step * self.configs.carbon_agv
        r_carbon_agv = carbon_agv_prev - carbon_agv_curr

        self.reward_e2 = self.reward_e2 + r_carbon_agv  # 借用旧变量记录物流碳排曲线
        self.reward_list[2] = self.reward_e2

        # D. 分时电价增量 (TOU Cost)
        # 重点：这里的 tou_cost_xxx 是我们在 step() 里通过严谨的“时间切片法”算出来的真金白银！
        r_tou = 1.0 * self.tou_cost_previous_step - self.tou_cost_this_step

        self.reward_idle_t = self.reward_idle_t + r_tou  # 借用旧变量记录电费节省曲线
        self.reward_list[3] = self.reward_idle_t

        """
        2. 获取权重 (Weights)
        """
        w_mk = self.configs.weight_mk  # w1: 完工时间权重
        w_carbon = self.configs.weight_carbon  # w2: 碳排放权重
        w_tou = self.configs.weight_tou  # w3: 分时电价权重

        """
        3. 加权求和 (计算最终反馈给 PPO 优化的综合得分)
        完美对应公式: Obj = w1*时间 + w2*(机器碳排 + 物流碳排) + w3*电费
        """
        step_total_r = w_mk * r_t + w_carbon * (r_carbon_m + r_carbon_agv) + w_tou * r_tou

        """
        4. 返回结果
        """
        if not done:
            return step_total_r / self.reward_function_parameters[
                'scaling_divisor'], r_t, r_carbon_m, r_carbon_agv, r_tou
        else:
            return step_total_r / self.reward_function_parameters[
                'scaling_divisor'], r_t, r_carbon_m, r_carbon_agv, r_tou


    def max_finish_time_in_machineRoute(self, G, machineRoute):
        #Todo 作用是 计算当前的 Makespan（最大完工时间）。
        max = 0 # 初始化最大时间为 0
        for value in machineRoute.values():
            for i in range(len(value)):
                if max <= G.nodes[value[i]][
                    'finish_time']:
                    max = G.nodes[value[i]]['finish_time']
        return max

    def reset(self,
              Random_weight_type="01"):
        # TODO 有警告set方法不同，但能运行。这里除非你专门指定，否则默认就是按照configs来走！（eval里边特别指定下！）
        """
    TOdo
        作用：
        将环境恢复到初始状态，以便 Agent 可以开始新的一局游戏。具体包括三个核心任务：
        清理历史数据：清除上一局生成的调度图中的机器连接边，清空机器的调度队列，重置所有计数器（如能耗、时间、Reward 累计值）。
        重载实例：重新加载作业车间的基础数据（工序时间、能耗表），确保图的基础结构（工序节点和工艺约束边）是正确的。
        初始化多目标权重：根据输入的 Random_weight_type，为当前这一局生成一组新的权重（时间 vs 能耗 vs 运输）。
        输入：
        Random_weight_type (str, 默认 "01"):决定了多目标权重（Makespan, Energy, Transport）是如何生成的。
        它会传递给内部函数 self.generate_random_weights。
        "01": 随机生成三个 0-1 之间的数并归一化（和为1）。
        "0.1": 从 [0, 0.1, ... 1.0] 中离散采样。
        "eval": 使用配置文件 中固定的权重（用于测试对比）。
        输出 :
        返回初始时刻的状态（State）。
        """
        machine_edges = [(from_, to_) for from_, to_, data_dict in self.G.edges(data=True) if not data_dict["job_edge"]]
        self.G.remove_edges_from(machine_edges)
        self.machine_routes = {m_id: np.array([]) for m_id in range(self.n_machines)}
        for i in range(1, self.total_tasks_without_dummies + 1):
            node = self.G.nodes[i]
            node["scheduled"] = False
            node["start_time"] = None,
            node["finish_time"] = None
        self.load_instance(jsp_instance=self.jsp_instance)
        self.energy_transport = [0, 0]  # 用来记录运输设备的能耗，这里我直接累加运输设备的总的时间
        self.exist_list = []  # 用来记录已经查到过的平行边的传输时间t,防止多查
        self.total_e1_this_step = 0  # 用来记录至今为止选了的设备的能耗和
        self.idle_t_this_step = 0  # 用来记录当前的idle时间（machine_route里边已经分配的m的空闲时间）
        self.trans_t_this_step = 0  # 记录至今为止的运输时间t，清0
        self.reward_list = [0, 0, 0, 0, 0]  # 对应记录当前的4个reward+总cost
        # 共有2次：init里边初始化为0，reward函数中进行累加
        self.reward_t = 0  # 传输时间的累计误差
        self.reward_e1 = 0  # 加工能耗的累计误差
        self.reward_e2 = 0  # 运输能耗的累计误差
        self.reward_idle_t = 0  # 空闲时间的累计误差
        # 记录已经被选择的task，记得清0
        self.selected_action = []
        self.selected_action_machine = []
        self.machines_fea = np.zeros(
            (self.n_machines, 8))
        self.it_s = [0] * self.total_tasks_without_dummies
        self.generate_random_weights(type=Random_weight_type)
        return self._state_array()


    def generate_random_weights(self, type="01"):
        """
    todo
        作用：为多目标优化问题生成三个目标的权重向量（完工时间、加工能耗、运输时间）
        "01": 随机生成三个 0-1 之间的数并归一化（和为1）。
        "0.1": 从 [0, 0.1, ... 1.0] 中离散采样。
        "eval": 使用配置文件 中固定的权重（用于测试对比）。
        """
        # 分支 1：type == "01"（连续随机权重） 权重是 0 到 1 之间的任意连续小数。
        if type == "01":
            weight_lst = [random.uniform(0, 1) for _ in
                          range(3)]
            self.reward_random_weight = np.array(weight_lst)
            self.reward_random_weight = self.reward_random_weight / np.sum(self.reward_random_weight,
                                                                           axis=-1)
        #分支 2：type == "0.1"
        elif type == "0.1":
            random_numbers = [round(random.uniform(0, 1), 1) for _ in
                              range(3)]
            total = sum(random_numbers)
            normalized_numbers = [round(num / total, 1) for num in random_numbers]
            self.reward_random_weight = np.array(normalized_numbers)
        #分支3：type == "eval"（固定权重）这个模式用于验证或测试阶段。
        elif type == "eval":
            self.reward_random_weight = np.array(
                [self.configs.weight_mk, self.configs.weight_carbon, self.configs.weight_tou])  # 指定的权重比

    def render(self, mode="human", show: List[str] = None, **render_kwargs) -> Union[
        None, np.ndarray]:
        """
        可视化环境状态。它可以画出甘特图
        """
        df = None
        colors = None
        if mode not in ["human", "rgb_array", "console"]:
            raise ValueError(f"mode '{mode}' is not defined. allowed modes are: 'human' and 'rgb_array'.")
        if show is None:
            if mode == "rgb_array":
                show = [s for s in self.default_visualisations if "window" in s]
            elif mode == "console":
                show = [s for s in self.default_visualisations if "console" in s]
            else:
                show = self.default_visualisations
        if "gantt_console" in show or "gantt_window" in show:
            df = self.network_as_dataframe()
            colors = {f"Machine {m_id}": (r, g, b) for m_id, (r, g, b, a) in self.machine_colors.items()}

        if "graph_console" in show:
            self.visualizer.graph_console(self.G, shape=self.size, colors=colors)
        if "gantt_console" in show:
            self.visualizer.gantt_chart_console(df=df, colors=colors)
        if "graph_window" in show:
            if "gantt_window" in show:
                if mode == "human":
                    self.visualizer.render_graph_and_gant_in_window(G=self.G, df=df, colors=colors, **render_kwargs)
                elif mode == "rgb_array":
                    return self.visualizer.gantt_and_graph_vis_as_rgb_array(G=self.G, df=df, colors=colors)
            else:
                if mode == "human":
                    self.visualizer.render_graph_in_window(G=self.G, **render_kwargs)
                elif mode == "rgb_array":
                    return self.visualizer.graph_rgb_array(G=self.G)
        elif "gantt_window" in show:
            if mode == "human":
                self.visualizer.render_gantt_in_window(df=df, colors=colors, **render_kwargs)
            elif mode == "rgb_array":
                return self.visualizer.gantt_chart_rgb_array(df=df, colors=colors)

    def _update_parallel_edge_inSameJob(self):
        """
    Todo
        作用：处理的是同一个 Job 内部，工序 A -> 工序 B -> 工序 C 这种必须按顺序执行的约束边。
        每当调度了一个任务，它的时长确定了，甚至机器确定了，那么它指向下一个任务的“距离”（边的权重）就变了，所以需要更新。
        """
        """
        1. 遍历所有工序
        """
        task_id = 0
        for i in range(self.n_jobs): # 遍历每一个 Job
            for j in range(self.n_machines): # 遍历 Job 里的每一个工序 Step
                task_id += 1  # 维护一个全局的 task_id (从1开始)，是累加的，对应图中节点的 ID（1, 2, 3...）。
                m_id = self.unscheduled_task_m_id
                dur = self.unscheduled_task_duration

                """
                2. 处理工序
                """
                if j == 0: # 如果是一个 Job 的第一个任务（比如 Task 1），它前面没有“同 Job 的上一个任务”。跳过
                    pass

                elif j == self.n_machines - 1:
                    # 计算运输时间：从上一个任务 (task_id - 1) 到当前任务 (task_id)
                    transport_t = find_transportT(self.G, (task_id - 1), task_id, self.instance_transT, self.configs)

                    #注意：如果无脑遍历所有task，那些m_id=-1的会把平行边的权重重置为0（其dur=0，transT限制=0）
                    if self.G.nodes[task_id - 1]['duration'] != 0:  # 关键判断！
                        self.G.add_edge(
                            task_id - 1, task_id,  # 连边：前置 -> 当前
                            job_edge=True,  # 标记为 Job 内部的顺序边
                            # 核心公式：权重 = 前置任务的加工时间 + 运输时间，这条边的权重表示：“当前任务要想开始，至少得等上一个任务结束并运过来”。
                            weight=self.G.nodes[task_id - 1]['duration'] + transport_t,
                            nweight=-(self.G.nodes[task_id - 1]['duration'] + transport_t)
                        )
                else:
                    transport_t = find_transportT(self.G, (task_id - 1), task_id, self.instance_transT, self.configs)
                    if self.G.nodes[task_id - 1]['duration'] != 0:
                        self.G.add_edge(
                            task_id - 1, task_id,
                            job_edge=True,
                            weight=self.G.nodes[task_id - 1]['duration'] + transport_t,
                            nweight=-(self.G.nodes[task_id - 1]['duration'] + transport_t)
                        )
        """
        3. 更新 Sink 边（指向汇点）
        循环结束后，代码单独处理了每个 Job 的最后一个任务指向 Sink 节点的边。
        """
        for task_id in range(self.n_machines, self.total_tasks,
                             self.n_machines):  # task_id 分别是 n, 2n, 3n... 即每个 Job 的最后一个任务 ID
            if self.G.nodes[task_id]['duration'] != 0: # 同样先判断是否已调度
                self.G.add_edge(
                    task_id, self.sink_task,
                    job_edge=True,
                    weight=self.G.nodes[task_id]['duration']
                    # 最后一个任务去 Sink 不需要运输时间，所以只加 duration
                )

    """
    这个函数是调度的总入口，负责逻辑判断（插队还是排队尾）。
    功能：尝试将指定的工序 (task_id) 分配给指定的机器 (m_id)。它会根据约束条件决定将该工序插入到机器队列的头部、中间（插空）还是尾部，并更新图结构（添加边、更新节点时间）。
    输入 (Input)：task_id (int): 目标工序的 ID（图节点编号，从 1 开始）。m_id (int): 目标机器的 ID（从 0 开始）。dur (int): 该工序在该机器上的加工时间。
    输出 (Output)：dict: 一个字典，包含调度结果信息。格式示例：
    {
    "start_time": 10,
    "finish_time": 25,
    "node_id": 5,
    "valid_action": True,       # 是否成功调度
    "scheduling_method": 'left_shift', # 调度方式 ('left_shift', '_insert_at_index_0', '_append_at_the_end')
    "left_shift": 1,            # 是否插队 (1是, 0否)
    } 
    """

    def _schedule_task(self, task_id: int, m_id: int, dur: int) -> dict:
        # 1. 获取并更新节点基础属性
        node = self.G.nodes[task_id]  # 获取图中的节点对象
        # 填写属性：分配给谁？什么颜色？做多久？
        node["machine"] = m_id
        node["color"] = self.machine_colors[m_id]
        node["duration"] = dur
        duration = node["duration"]  # 当前节点的处理时间 （node字典的duration属性）

        # 2. 更新同 Job 的平行边，这个函数会重新遍历所有工序，检查边权重。
        self._update_parallel_edge_inSameJob()

        # 3. 检查：是否重复调度
        # 检查重复调度。如果这个节点已经是 True，说明之前排过了，报错返回
        if node["scheduled"]:
            if self.verbose > 0:
                log.info(f"task {task_id} is already scheduled. ignoring it.")
            return {
                "valid_action": False,  # 被调度了，就false表示不能再选了！！！
                "node_id": task_id,
            }

        m_id = node["machine"]  # 该task也就是node的分配的machine id是什么
        # 4. 检查：前置工序是否完成
        # 找到该节点的入边，取第一个（即 Job 内部的前一个工序）
        # 比如：Task 5 的入边是 Task 4 -> Task 5
        prev_task_in_job_id, _ = list(self.G.in_edges(task_id))[0]
        prev_job_node = self.G.nodes[prev_task_in_job_id]

        # 如果前置节点还没被调度 (scheduled=False)
        if not prev_job_node["scheduled"]:
            # 如果前置工序没做，当前工序不能做（防止死锁）
            if self.verbose > 1:
                log.info(f"the previous task (T{prev_task_in_job_id}) in the job is not scheduled jet. "
                         f"Not scheduling task T{task_id} to avoid cycles in the graph.")
            return {
                "valid_action": False,  # 这里的return啥意思，不是有效的动作选择（我随机产生已经避免了这种情况了，）
                "node_id": task_id,
            }

        # 5. 获取机器当前的排班队列长度
        len_m_routes = len(self.machine_routes[m_id])
        # ================= 分支：机器上有任务，尝试插空 =================
        if len_m_routes:

            if self.perform_left_shift_if_possible:  # 如果开启了插空功能

                # A. 计算该工序的最早可能开始时间
                # 必须等前置工序做完 + 运输过来
                j_lower_bound_st = find_max_arrivaTime_for_currentNode(self.G, task_id, self.instance_transT,
                                                                       self.configs)
                j_lower_bound_ft = j_lower_bound_st + duration  # 当前开始（前置的最晚到达）+ 当前持续 = 当前节点的结束时间
                # B. 尝试插在队首
                m_first = self.machine_routes[m_id][0]  # 机器上当前的第一个任务
                # 计算第一个任务的最早开始时间
                m_first_st = find_max_arrivaTime_for_currentNode(self.G, m_first, self.instance_transT, self.configs)

                # 如果你的结束时间 <= 它的开始时间，说明可以插在它前面
                if j_lower_bound_ft <= m_first_st:
                    # 调用 _insert_at_index_0 插在第一位
                    info = self._insert_at_index_0(task_id=task_id, node=node, m_id=m_id)
                    # 添加机器约束边：当前任务 -> 原第一个任务
                    transport_t = find_transportT(self.G, task_id, m_first, self.instance_transT, self.configs)
                    # 计算空闲时间 (用于边权重)
                    blank = self.G.nodes[m_first]['start_time'] - self.G.nodes[task_id]['finish_time']
                    # 更新新边：当前节点，和，同一个m的第一个节点的
                    self.G.add_edge(
                        task_id, m_first,
                        job_edge=False,  # 判断是否是原先最开始画的平行边（表示同一个job的子任务先后顺序！！）
                        weight=duration + transport_t + blank
                    )
                    return info
                # 如果只有1个任务且插不到前面，就排后面
                elif len_m_routes == 1:
                    return self._append_at_the_end(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)
                # C. 尝试插在中间 (遍历机器任务队列)
                # 检查每两个相邻任务 (m_prev, m_next) 之间的缝隙
                for i, (m_prev, m_next) in enumerate(zip(self.machine_routes[m_id], self.machine_routes[m_id][
                                                                                    1:])):  # 遍历machine是同一个m_id的列表，枚举每一对相邻的约束任务节点:返回是节点id
                    m_temp_prev_ft = self.G.nodes[m_prev]["finish_time"]
                    m_temp_next_st = self.G.nodes[m_next]["start_time"]  # 后一个节点的开始时间是不对的，因为有运输时间的；结束时间
                    # 重新计算后一个任务的最早开始时间 (防止它是被推迟过的)
                    m_next_st = find_max_arrivaTime_for_currentNode(self.G, m_next, self.instance_transT, self.configs)
                    # 条件1：你的结束时间必须 <= 后一个的开始时间
                    if j_lower_bound_ft > m_next_st:  # 当前节点的结束时间 大于 集合中的后一个节点的开始时间：不能插空
                        continue
                    # 条件2：缝隙 (Gap) 必须 >= 你的时长
                    m_gap = m_next_st - m_temp_prev_ft
                    if m_gap < duration:  # 两个节点之间的时间空隙 小于 当前节点的持续时间
                        continue  # 跳过循环，下一轮
                    # === 找到空隙，执行插入 ===
                    replaced_edge_data = self.G.get_edge_data(m_prev, m_next)  # 获取m_prev和m_next之间的边的属性
                    # 1. 计算具体时间
                    task_id_st_forPrev = find_max_arrivaTime_for_currentNode(self.G, task_id, self.instance_transT,
                                                                             self.configs)
                    transport_t = find_transportT(self.G, m_prev, task_id, self.instance_transT,
                                                  self.configs)  # 这不都是同一个的m，铁定等于0，没有运输！
                    # 开始时间 = max(前置工序约束, 机器上一个任务约束)
                    st = max(task_id_st_forPrev, (self.G.nodes[m_prev]["finish_time"] + transport_t))
                    ft = st + duration
                    # 2. 更新节点
                    node["start_time"] = st
                    node["finish_time"] = ft
                    node["scheduled"] = True
                    # 3. 更新边：删除旧边 (Prev->Next)，添加新边 (Prev->You->Next)
                    blank = self.G.nodes[task_id]['start_time'] - self.G.nodes[m_prev]['finish_time']

                    self.G.add_edge(
                        m_prev, task_id,
                        job_edge=replaced_edge_data['job_edge'],  # 判断是否是原先最开始画的平行边，表示同一个job的子任务先后顺序！！
                        weight=self.G.nodes[m_prev]["duration"] + transport_t + blank  # 加上运输时间
                    )
                    transport_t = find_transportT(self.G, task_id, m_next, self.instance_transT,
                                                  self.configs)  # 这不都是同一个的m，铁定等于0，没有运输！

                    """判断调度完成之后，新的边是否需要更新上空闲时间: 后一个st-前一个ft"""
                    blank = self.G.nodes[m_next]['start_time'] - self.G.nodes[task_id][
                        'finish_time']  # 遍历每个列表中的元素：后一个st开始 - 前一个ft结束
                    self.G.add_edge(
                        task_id, m_next,
                        job_edge=False,
                        # weight=duration
                        # wrk
                        weight=duration + transport_t + blank  # 当前task_id节点的持续时间 + 加上查表得的运输时间t
                    )
                    self.G.remove_edge(m_prev, m_next)  # 删除之前的边！！！
                    # 4. 更新机器路径列表
                    self.machine_routes[m_id] = np.insert(self.machine_routes[m_id], i + 1, task_id)

                    if self.verbose > 1:
                        log.info(f"scheduled task {task_id} on machine {m_id} between task {m_prev:.0f} "
                                 f"and task {m_next:.0f}")

                    return {  # 返回节点（子任务的）的开始和结束时间（甘特图就看子任务先后顺序，和同一个m上的先后顺序，先后用时间来表示：gantt上边的抽象时间）
                        "start_time": st,
                        "finish_time": ft,
                        "node_id": task_id,
                        "valid_action": True,
                        "scheduling_method": 'left_shift',
                        "left_shift": 1,
                    }
                # 遍历完了都没地方插，只能排队尾
                else:
                    return self._append_at_the_end(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)
            # 如果没开启插空功能，直接排队尾
            else:
                return self._append_at_the_end(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)
        # ================= 分支：机器是空的 =================
        else:
            return self._insert_at_index_0(task_id=task_id, node=node, m_id=m_id)

    def _append_at_the_end(self, task_id: int, node: dict, prev_job_node: dict, m_id: int) -> dict:
        """
        这是最常用的情况：把任务加到机器队列的最后。
        功能：在 machine_routes[m_id] 的末尾添加任务，并添加从机器上当前最后一个任务指向新任务的边。
        输入 (Input)：task_id (int): 要调度的工序 ID。node (dict): 该工序在 NetworkX 图中的节点对象（引用）。
        prev_job_node (dict): 该工序在 Job 中的前置工序节点。m_id (int): 机器 ID。
        输出 (Output)：dict (调度结果)。
        """
        # 1. 找到机器上当前的最后一个任务
        prev_m_task = self.machine_routes[m_id][-1]  # 当前m_id设备的最后一个子任务的id
        prev_m_node = self.G.nodes[prev_m_task]  # 最后一个子任务的node信息
        # 2. 把新任务加到列表末尾
        self.machine_routes[m_id] = np.append(self.machine_routes[m_id], task_id)  # 将任务ID task_id 插入到当前任务用的m的序列中
        # 3. 计算开始时间
        # A. 工艺约束：必须等 Job 前置工序做完 + 运输
        task_id_st_forPrev = find_max_arrivaTime_for_currentNode(self.G, task_id, self.instance_transT, self.configs)
        # B. 机器约束：必须等机器上一个任务做完 + 运输
        transport_t = find_transportT(self.G, prev_m_task, task_id, self.instance_transT, self.configs)
        # C. 最终开始时间 = Max(工艺约束, 机器约束)
        st = max(task_id_st_forPrev, (prev_m_node["finish_time"] + transport_t))
        # D. 结束时间 = 开始 + 持续
        ft = st + node["duration"]
        # 4. 更新节点状态
        node["start_time"] = st
        node["finish_time"] = ft
        node["scheduled"] = True

        # 5. 添加连边 (机器约束边)
        blank = self.G.nodes[task_id]['start_time'] - self.G.nodes[prev_m_task][
            'finish_time']  # 遍历每个列表中的元素：后一个st开始 - 前一个ft结束

        self.G.add_edge(
            prev_m_task, task_id,  # 添加原先最后一个子任务prev_m_task，到，当前任务task_id的，边(权重都是上一个节点的权重)
            job_edge=False,
            weight=prev_m_node['duration'] + transport_t + blank  # 查表增加初始权重
        )

        return {
            "start_time": st,
            "finish_time": ft,
            "node_id": task_id,
            "valid_action": True,
            "scheduling_method": '_append_at_the_end',
            "left_shift": 0,
        }

    def _insert_at_index_0(self, task_id: int, node: dict, m_id: int) -> dict:
        """
        这种情况发生在机器是空的，或者插队成功插到了最前面。
        功能：在 machine_routes[m_id] 的第 0 位插入任务。因为前面没有机器任务，所以只受 Job 前置工序的约束。
        输入 (Input)：task_id (int): 工序 ID。node (dict): 节点对象。m_id(int): 机器 ID。
        输出 (Output)：dict (调度结果)
        """
        # 1. 把任务插入到列表第 0 位
        self.machine_routes[m_id] = np.insert(self.machine_routes[m_id], 0,
                                              task_id)  # 在 字典machine_routes的m_id设备的array中，第一位插入task_id
        # 2. 计算开始时间
        # 因为是机器上的第一个，没有“机器上一个任务”，所以只受“Job 前置工序”约束
        st = find_max_arrivaTime_for_currentNode(self.G, task_id, self.instance_transT, self.configs)
        ft = st + node["duration"]
        # 3. 更新节点状态
        node["start_time"] = st
        node["finish_time"] = ft
        node["scheduled"] = True
        # return additional info
        return {  # 并返回信息
            "start_time": st,
            "finish_time": ft,
            "node_id": task_id,
            "valid_action": True,
            "scheduling_method": '_insert_at_index_0',
            "left_shift": 0,
        }

    def estiamte_ft_st_eachStep(self, current_ft, current_st, if_schedule):
        # TODO：旧代码好像是
        machine_j_m = copy.deepcopy(self.jsp_instance[0]).astype(int)  # machine保持整数！就2个元素，第一个是选择的m，第二个是对应的加工时间
        processT_j_m = copy.deepcopy(self.jsp_instance[-1])  # 就2个元素，第一个是选择的m，第二个是对应的加工时间
        transT_m_m = copy.deepcopy(self.instance_transT)  # m和m之间的运输时间
        begin_ft = current_ft * if_schedule  # 保留真实值，重新计算其他未调度的task的估计值  (此为有0的新state！！！！)
        update_ft = begin_ft  # 实际更新的ft矩阵
        begin_st = current_st * if_schedule  # 保留真实值，重新计算其他未调度的task的估计值  (此为有0的新state！！！！)
        update_st = begin_st  # 实际更新的st矩阵
        # 傻瓜式遍历更新
        for row in range(current_ft.shape[0]):  # 矩阵的行,job数量
            for col in range(current_ft.shape[1]):  # 矩阵的列，machine数量
                if begin_ft[row][col] == 0:
                    if col != 0:  # 防止是第一列，index会出错

                        update_ft[row][col] = update_ft[row][col - 1] + transT_m_m[machine_j_m[row][col - 1]][
                            machine_j_m[row][col]] + processT_j_m[row][col]  # 运输t都是同job（同一行）中上一个m到当前m
                    else:
                        update_ft[row][col] = 0 + 0 + processT_j_m[row][col]  # 首列，直接就是加工时间！
                # st的初始值，第一列有0存在，但不一定全是0
                if if_schedule[row][col] == 0:  # 说明还没有被调度，所以需要进行估计
                    if col == 0:
                        update_st[row][col] = 0  # 说明此时第一列的首个task，还没被调度，开始时间预估为0
                    else:
                        update_st[row][col] = update_st[row][col - 1] + processT_j_m[row][col - 1] + \
                                              transT_m_m[machine_j_m[row][col - 1]][
                                                  machine_j_m[row][col]]  # 运输t都是同job（同一行）中上一个m到当前m
        # 遍历，重新更新当前step且还没调度的task的ft和st
        return update_ft, update_st

    def estiamte_ft_eachStep_noTransT(self, current_ft, current_st, if_schedule):
        # TODO：旧代码好像是
        # machine_j_m = copy.deepcopy(self.jsp_instance[0]).astype(int)  # machine保持整数！就2个元素，第一个是选择的m，第二个是对应的加工时间
        instance_dur = copy.deepcopy(self.jsp_instance[0])  # task*m的加工时间能力矩阵
        instance_dur[instance_dur < 0] = float("inf")  # 将小于等于0的元素替换为无穷大
        """找到所有task的最小t"""
        min_dur_j_m = np.min(instance_dur, axis=1)  # # 沿着axis=1的方向找到每一行的最小值, 返回的是一维数组啊！总共task个元素
        min_dur_j_m = min_dur_j_m.reshape(self.n_jobs,
                                          self.n_machines)
        begin_ft = current_ft * if_schedule
        update_ft = begin_ft  # 实际更新的ft矩阵
        begin_st = current_st * if_schedule
        update_st = begin_st  # 实际更新的st矩阵
        for row in range(current_ft.shape[0]):  # 矩阵的行,job数量
            for col in range(current_ft.shape[1]):  # 矩阵的列，machine数量
                if begin_ft[row][col] == 0:
                    if col != 0:
                        update_ft[row][col] = update_ft[row][col - 1] + min_dur_j_m[row][col]  # 运输t都是同job（同一行）中上一个m到当前m
                    else:
                        update_ft[row][col] = 0 + min_dur_j_m[row][col]  # 首列，直接就是min的加工时间！
        return update_ft

    def estiamte_st_ft_pt_eachStep_noTransT(self, current_ft, current_st, current_pt, if_schedule):
        """
        作用： 对于那些还没做的工序，神经网络不知道它们的数据是什么。这个函数采用一种“极端乐观的策略”（即下界估计）来填补这些空白：
        时间估算：假设所有未分配任务都由最快的机器执行，且工序之间无缝衔接（无运输、无等待）。
        能耗估算：假设所有未分配任务都由能耗最低的机器执行。
        输入：current_ft, current_st, current_pt: 当前真实的完工时间、开始时间、能耗列表（未调度的位置是 0）。if_schedule: 掩码列表（0 表示未调度，1 表示已调度）。
        输出：一张混合了“历史真实数据”和“未来理想数据”的完整状态表。
        """
        # 第一部分：准备工作：寻找“理论最优值”
        # 1. 复制加工时间矩阵 (Tasks x Machines)
        instance_dur = copy.deepcopy(self.jsp_instance[0])
        # 2. 计算能耗矩阵 = 功率(P) * 时间(T)
        instance_pt = np.multiply(copy.deepcopy(self.jsp_instance[0]), np.abs(copy.deepcopy(self.jsp_instance[1])))
        # 3. 处理无效数据
        # 把小于0的值（代表该机器无法处理该任务）设为无穷大
        instance_dur[instance_dur < 0] = float("inf")
        instance_pt[instance_pt < 0] = float("inf")

        # 4. 找到每个任务的“最短时间”
        # axis=1 表示沿着机器维度找最小值。
        # 结果是一个 (Tasks,) 的一维数组，表示每个任务理论上最快能做多久
        min_dur_j_m = np.min(instance_dur, axis=1)
        min_dur_j_m = min_dur_j_m.reshape(self.n_jobs, self.n_machines)  # 变回 (Jobs, Machines) 形状
        # 5. 找到每个任务的“最小能耗”
        min_pt_j_m = np.min(instance_pt, axis=1)  # 输出每行的最小值，一维数组，task个元素
        min_pt_j_m = min_pt_j_m.reshape(self.n_jobs, self.n_machines)  # 整型成j*m，用来选择预估加工能耗PE

        # 第二部分：初始化更新矩阵
        # current_ft * if_schedule：把未调度的位置全置为 0，已调度的保留真实值
        begin_ft = current_ft * if_schedule  # 保留真实值，重新计算其他未调度的task的估计值  (此为有0的新state！！！！)
        update_ft = copy.deepcopy(begin_ft)  # 这个变量将用来存储“真实+预估”的混合结果

        begin_st = current_st * if_schedule  # 保留真实值，重新计算其他未调度的task的估计值  (此为有0的新state！！！！)
        update_st = copy.deepcopy(begin_st)  # 实际更新的st矩阵

        real_pt = current_pt * if_schedule
        update_pt = copy.deepcopy(real_pt)

        # 第三部分：第一轮循环：推导完工时间 FT
        for row in range(current_ft.shape[0]):  # 遍历 Job
            for col in range(current_ft.shape[1]):  # 遍历 Job 内的工序顺序

                # 如果这个位置是 0 (说明未调度 or 是首位)，需要计算
                if begin_ft[row][col] == 0:
                    if col != 0:  # 如果不是 Job 的第一个任务
                        # 当前预估FT = 上一个任务的FT + 当前任务的最小加工时间
                        # update_ft[row][col-1]：如果上一个任务已调度，这里就是真实FT。如果上一个任务未调度，这里就是刚才循环算出来的预估FT。
                        update_ft[row][col] = update_ft[row][col - 1] + min_dur_j_m[row][col]
                    else:  # 如果是 Job 的第一个任务
                        update_ft[row][col] = 0 + min_dur_j_m[row][col]  # FT = 0 + 最小加工时间

        # 第四部分：第二轮循环：推导 ST 和 PT
        for row in range(current_ft.shape[0]):  # 矩阵的行,job数量
            for col in range(current_ft.shape[1]):  # 矩阵的列，machine数量
                # 只处理未调度的任务
                if if_schedule[row][col] == 0:
                    # 1. 更新 ST (开始时间)
                    if col == 0:
                        update_st[row][col] = 0  # 第一个任务ST为0
                    else:
                        # 当前 ST = 上一个任务的 FT
                        # 这体现了 "noTransT" (无运输时间) 和 "无空闲" 的假设
                        update_st[row][col] = update_ft[row][col - 1]
                    # 2. 更新 PT (能耗)
                    # 直接填入之前算好的最小能耗
                    update_pt[row][col] = min_pt_j_m[row][col]

        return update_st, update_ft, update_pt  # j*m


    def _state_array(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        它负责把当前复杂的图结构和各种内部变量，转换成神经网络能够理解的矩阵（Tensor）格式
        功能：提取当前环境的状态，生成 邻接矩阵、工序特征矩阵 和 机器特征矩阵。
        输入：无显式参数（使用 self 内部状态）。
        输出：返回一个包含 9 个元素的元组。最重要的三个是：
        这个函数一口气返回了 9 个变量。我们来总结一下它们分别是什么，给谁用：
        res (Standard State):身份：Gym 环境的标准 Observation。
        ft_s (Finish Times):身份：当前已调度工序的完工时间列表。
        self.it_s (Idle Times):身份：当前步骤新增的空闲时间。
        adj_wrk (Weighted Adjacency Matrix):身份：加权邻接矩阵（包含了边权重、运输时间、空闲时间等）。
        tasks_fea (Basic Task Features):（3维：时间、能耗、Mask）。
        self.machines_fea (Machine Features):身份：机器特征矩阵（8维：动态统计 + 静态权重）。
        tasks_fea_1101 (Rich Task Features):身份：增强版工序特征（12维：包含入度、预估、权重等）。
        ft_s_estimated (Estimated FT):所有工序的预估完工时间（一维数组）。
        pt_s_estimated (Estimated PT):所有工序的预估/真实加工能耗（一维数组）。
        """
        # TODO 第一部分，零阶矩阵构建
        # 1. 提取原始邻接矩阵
        # [1:-1, 1:-1] 表示切片去掉第 0 行列(Source) 和最后一行列(Sink)，神经网络只关心真实的工序节点
        adj = nx.to_numpy_array(self.G)[1:-1, 1:-1].astype(dtype=int)
        adj_wrk = copy.deepcopy(adj)

        # 2. 权重处理
        for i in range(adj_wrk.shape[0]):
            for j in range(adj_wrk.shape[1]):
                # 判断边是否存在
                if adj_wrk[i, j] != 0:
                    # 获取节点加工时间
                    if self.G.nodes[i + 1]["machine"] < 0:
                        node_dur = 1
                    else:
                        node_dur = self.G.nodes[i + 1].get('duration', 0)
                    # 核心计算：新权重 = 原权重 - 加工时间 + 1
                    adj_wrk[i, j] -= node_dur
                    adj_wrk[i, j] += 1
        # 3. 加自环和转置
        identity_matrix = np.eye(adj_wrk.shape[0])  # 加上单位矩阵，让节点能接收到自己的特征信息
        adj_wrk = adj_wrk + identity_matrix
        adj_wrk = adj_wrk.T

        # TODO 第二部分，节点基础特征提取
        # 1.提取机器和时间映射
        task_to_machine_mapping = np.zeros(shape=(self.total_tasks_without_dummies, 1), dtype=int)
        task_to_duration_mapping = np.zeros(shape=(self.total_tasks_without_dummies, 1), dtype=self.dtype)
        for task_id, data in self.G.nodes(data=True):  # 通过遍历图中的每个任务节点，将任务节点的属性信息填充到相应的数组中。不包括src和sink节点
            if task_id == self.src_task or task_id == self.sink_task:
                continue
            else:
                task_to_machine_mapping[task_id - 1] = data["machine"]  # 此时未调度的task节点的m_id = -2
                task_to_duration_mapping[task_id - 1] = data["duration"]  # 此时未调度的task节点的dur = 0

        # 2. 独热编码处理
        if self.normalize_observation_space:
            task_to_machine_mapping = task_to_machine_mapping.astype(int).ravel()  # 数组转换为整数类型，并将其展平为一维数组
            n_values = self.n_machines
            # 因为未选机器时 m_id < 0，不能直接做 One-Hot
            # 造了一个全 0 矩阵，只有当 val >= 0 (已分配机器) 时才填入 1
            task_to_machine_mapping_zero = np.zeros((len(task_to_machine_mapping), n_values))
            one_hot_array = np.eye(n_values)
            for i, val in enumerate(task_to_machine_mapping):
                if val >= 0:
                    task_to_machine_mapping_zero[i] = one_hot_array[val]

            task_to_machine_mapping = task_to_machine_mapping_zero

            # 3. 拼接与“截断”
            # 1. 拼接：邻接矩阵 + 机器编码 + 时间
            res = np.concatenate((adj, task_to_machine_mapping, task_to_duration_mapping), axis=1, dtype=self.dtype)
            # 2. 【截断】只取前 Tasks 列，也就是只保留了 adj（邻接矩阵）
            out_s = res[:, 0:self.total_tasks_without_dummies]
            # 3. 制作新特征：是否已调度，做一个长度为 36 的列表 task_s，被调度的工序填1，没选过填0
            task_s = [0] * self.total_tasks_without_dummies
            for i in self.selected_action:
                task_s[i] = 1
            # 4. 重新拼接：最终 res = 邻接矩阵 + 1列被选掩码
            out_s = np.column_stack((out_s, task_s))
            res = out_s
            # 初始化完工时间列表，用来存放所有工序的真实完工时间
            ft_s = [0] * self.total_tasks_without_dummies  # 全0列表
            for i in self.selected_action:  # selected_action = action[0-15]正好对应task[1-16]：[0,1,2,3,4,xxxx]都是action不断累加进来的，0-16个数
                ft_s[i] = self.G.nodes[i + 1]['finish_time']  # 收集已调度任务的完工时间
            if self.selected_action:  # 不是空集的时候
                self.it_s[self.selected_action[
                    -1]] = self.idle_t_this_step - self.idle_t_previous_step  # 计算罪证：(现在的总空闲) - (上一步的总空闲)
            ft_s = np.array(ft_s)
            self.it_s = np.array(self.it_s)  # 记录在案：self.it_s记录了每个任务分别导致了多少空闲时间。

            """--------------------------------------Estimate ST + FT + PE(加工能耗) - 采用min最小值！ -----------------------------------------------------"""
            """
            每一个step都会更新一下state
            1、先判断是否被调度
            2、输出已调度的状态 
            3、更新未调度的预估的状态
            最终，作为当前的状态进行输出                                                                   
            """
            # TODO 第三部分：全图状态预估
            # 1. 初始化容器。造了 4 个长度为 总工序数 的全 0 列表，分别用来存：是否调度、完工时间、开始时间、加工能耗。
            if_schedule_lst = [0] * self.total_tasks_without_dummies  # 全0列表
            ft_lst = [0.0] * self.total_tasks_without_dummies  # 全0列表  完工时间
            st_lst = [0.0] * self.total_tasks_without_dummies  # 全0列表  开始时间
            pt_lst = [0.0] * self.total_tasks_without_dummies  # 全0列表  加工能耗PE
            mOrder_lst = [copy.deepcopy(self.n_machines)] * self.total_tasks_without_dummies  # 全max machine数量的列表
            # 2. 填入真实历史数据。每一步都重头开始遍历，确定当前step的状态
            # 下列是当前的已调度的完工时间ft和开始时间st + 已经选择的m的id + 是否被调度（未选的元素统一为0！！）
            """真实值：从已经选好的task的列表中，确定真实的st+ft+Io+pt！！！！！"""
            for i_a in self.selected_action:  # # 遍历每一个已经执行过的动作
                if_schedule_lst[i_a] = 1  # 标记为已调度
                # 从图节点里把真实发生的时间取出来
                ft_lst[i_a] = self.G.nodes[i_a + 1]['finish_time']  # node图中记录的都是task_id, 从1开始的
                st_lst[i_a] = self.G.nodes[i_a + 1]['start_time']  # node图中记录的都是task_id, 从1开始的
                # 计算真实能耗：查表 Power * 机器
                pt_lst[i_a] = self.instance_processingEnergy[i_a][self.G.nodes[i_a + 1]["machine"]]

            # 3. 填补未发生的数据
            st_s_array, ft_s_array, pt_s_array = self.estiamte_st_ft_pt_eachStep_noTransT(
                current_ft=np.array(ft_lst).reshape(self.n_jobs, self.n_machines),
                current_st=np.array(st_lst).reshape(self.n_jobs, self.n_machines),
                current_pt=np.array(pt_lst).reshape(self.n_jobs, self.n_machines),
                if_schedule=np.array(if_schedule_lst).reshape(self.n_jobs, self.n_machines))

            # TODO 第四部分：构建工序特征向量，（3特征版本）
            # 它将“预估的未来”、“已发生的历史”以及“当前的调度目标”混合在一起，为每一个工序节点生成一个长度为 12 的特征向量
            tasks_fea = []
            st_s_estimated = st_s_array.flatten()  # 将实时更新的预估ft二维矩阵，转成1维
            ft_s_estimated = ft_s_array.flatten()  # 将实时更新的预估ft二维矩阵，转成1维
            pt_s_estimated = pt_s_array.flatten()  # 将实时更新的预估ft二维矩阵，转成1维

            for i in range(self.total_tasks_without_dummies):  # 遍历所有节点
                task_fea = []
                task_fea.append(ft_s_estimated[i])  # 特征一：该工序的预计完工时间。,对应节点的ft + 预估ft
                # 特征二：加工能耗
                if self.G.nodes[i + 1]["scheduled"] == True:
                    task_fea.append(self.instance_processingEnergy[i][self.G.nodes[i + 1]["machine"]])  # 将真实的能耗值加入特征列表
                else:
                    task_fea.append(0)  # 没有被调度的，p*t设为0，没有预估！！！！！！！！！
                # 特征三：调度掩码
                task_fea.append(if_schedule_lst[i])  # 对应节点是否被调度

                # 将当前工序特征加入总表
                tasks_fea.append(copy.deepcopy(task_fea))
            tasks_fea = np.array(tasks_fea)  # list转成二维矩阵！

            # TODO 第四部分：构建工序特征向量，（12特征版本）
            tasks_fea_1101 = []  # task * x元素
            for i in range(self.total_tasks_without_dummies):  # 遍历所有节点, task_index
                one_task_fea = []
                one_task_fea.append(st_s_estimated[i])  # 对应节点的st + 预估st  TODO 1 预估ST
                one_task_fea.append(ft_s_estimated[i])  # 对应节点的ft + 预估ft  TODO 2 预估FT
                one_task_fea.append(pt_s_estimated[i])  # 对应节点的pt + 预估pt  TODO 3 预估PT  作为权重就去掉这里！
                one_task_fea.append(if_schedule_lst[i])  # 对应节点是否被调度    TODO 4 被调度I=mask
                one_task_fea.append(len(self.G.in_edges(i + 1)))  # 节点的入边的个数  TODO 5 被调度=in_dedge_n

                if self.G.nodes[i + 1]["scheduled"]:  # True 表示被调度了
                    one_task_fea.append(
                        self.G.nodes[i + 1]["machine"] + 1)  # 对应已调度节点的machine归属：用id来表示！！！！  TODO 6 被调度m_id=1开始
                    one_task_fea.append(self.jsp_instance[0][i][
                                            self.G.nodes[i + 1]["machine"]])  # task*m的t能力，index来定位    TODO 7 被调度加工时间
                    one_task_fea.append(self.jsp_instance[1][i][self.G.nodes[i + 1][
                        "machine"]])  # task*m的p能力，index来定位（task，m）  TODO 8 被调度功率 (p)
                else:  # 没有被调度的，当前都是0！
                    one_task_fea.append(0)  # 对应已调度节点的machine归属：用id来表示！！！！  TODO 6 初始化
                    one_task_fea.append(0)  # task*m的t能力，index来定位  TODO 7 初始化
                    one_task_fea.append(0)  # task*m的p能力，index来定位（task，m）  TODO 8 初始化
                # 添加特征：作业ID
                one_task_fea.append(self.G.nodes[i + 1]["job"] + 1)  # 对应节点的job归属：用id来表示！！！！ TODO 9 固定不变j_id=1开始
                # 添加特征：多目标权重
                one_task_fea.append(self.reward_random_weight[0])  # 对应节点的job归属：用id来表示！！！！ TODO 10 固定不变mk权重 (最大完工时间)
                one_task_fea.append(self.reward_random_weight[1])  # 对应节点的job归属：用id来表示！！！！ TODO 11 固定不变ec权重（加工和等待能耗）
                one_task_fea.append(self.reward_random_weight[2])  # 对应节点的job归属：用id来表示！！！！ TODO 12 固定不变 (运输时间权重)

                # 记录每一个task节点
                tasks_fea_1101.append(copy.deepcopy(one_task_fea))
            tasks_fea_1101 = np.array(tasks_fea_1101)  # list转成二维矩阵！

            # TODO 第五部分：构建机器特征向量
            # 静态特征 (3维)：权重 MK, EC, TT（初始化设定，后续不变）。
            # 动态特征 (5维)：完工时间、累积能耗、累积运输、累积空闲、工序计数（每一步只更新被选中的那台机器）。
            # 1. 准备工作：获取能力矩阵与统计数据
            ability_t = self.jsp_instance[0]  # 获取当前所有工序在所有机器上的加工时间矩阵
            ability_p = self.jsp_instance[1]  # 获取当前所有工序在所有机器上的功率矩阵
            ability_ec = ability_p * ability_t  # 计算加工能耗矩阵
            mean_p = np.mean(ability_p[ability_p > 0])  # 计算整个样本中所有有效功率（大于0）的平均值。

            # 2. 状态更新逻辑：判断是第一步还是后续步骤
            if self.selected_action:  # 当 self.selected_action 不为空（即已经选过动作了），执行此段代码。这意味着只更新当前被选中的那台机器的状态，其他机器状态保持不变
                # 获取最新一次动作选中的 工序索引 (cur_task_index) 和 机器索引 (cur_m_index)。
                cur_task_index = self.selected_action[-1]  # 当前step的task和m的index
                cur_m_index = self.selected_action_machine[-1]

                final_task_id = self.machine_routes[cur_m_index][-1]
                # TODO 1  Feature 1: 完工时间。找到这台机器上排在最后一位的工序 ，取它的完工时间更新
                self.machines_fea[cur_m_index][0] = self.G.nodes[final_task_id]["finish_time"]
                # TODO 2 Feature 2: 累积加工能耗。累加当前工序在这台机器上的能耗。这里做了一个平均化处理（除以总工序数）
                self.machines_fea[cur_m_index][1] += ability_ec[cur_task_index][
                                                         cur_m_index] / self.total_tasks_without_dummies
                # TODO  Feature 3: 累积运输时间
                # 如果是 Job 的第一个工序，运输时间为 0。
                # 如果是中间工序，计算从上一个工序（task_id）到当前工序（task_id + 1）的运输时间。
                if cur_task_index % self.n_machines == 0:
                    new_avail_transT = 0
                else:
                    new_avail_transT = find_transportT(self.G, cur_task_index, (cur_task_index + 1),
                                                       self.instance_transT, self.configs)
                self.machines_fea[cur_m_index][2] += new_avail_transT  # 同一m的累积的运输时间t
                # TODO Feature 4: 累积空闲时间
                self.machines_fea[cur_m_index][3] += self.idle_t_this_step - self.idle_t_previous_step


            else:  # 表明此时没有选择task，就是在初始化！
                # 初始化的时候直接遍历就好了
                for m_index1 in range(self.n_machines):  # 遍历每一个m的节点！
                    self.machines_fea[m_index1][0] = 0  # 没有task，完工时间0      TODO 1 完工时间初始化
                    self.machines_fea[m_index1][1] = 0  # 没有task，没有累加p*t      TODO 2 累积加工能耗初始化
                    self.machines_fea[m_index1][2] = 0  # 没有task，没有累加transT   TODO 3 累积运输时间初始化
                    self.machines_fea[m_index1][3] = 0  # 没有task，没有累加idleT     TODO 4 累积空闲时间初始化
                    # 将当前的三个多目标权重（MK, EC, TT）赋值给机器特征。
                    self.machines_fea[m_index1][5] = self.reward_random_weight[
                        0]  # 对应节点的job归属：用id来表示！！！！ TODO 6 固定不变mk权重
                    self.machines_fea[m_index1][6] = self.reward_random_weight[
                        1]  # 对应节点的job归属：用id来表示！！！！ TODO 7 固定不变ec权重（加工和等待）
                    self.machines_fea[m_index1][7] = self.reward_random_weight[
                        2]  # 对应节点的job归属：用id来表示！！！！ TODO 8 固定不变j_id=1开始

            # ==============================================================================
            # todo 5 zzo 全局电价
            time_unit_hours = 1.0 / 60.0  # 确保与奖励计算中的时间单位一致（分钟转小时）
            for m_id in range(self.n_machines):
                # 1. 获取这台机器自己当前的时间线 (即它上一个工序的完工时间)
                if len(self.machine_routes[m_id]) > 0:
                    last_task_on_m = self.machine_routes[m_id][-1]
                    m_current_time = self.G.nodes[last_task_on_m]["finish_time"]
                else:
                    m_current_time = 0.0  # 如果机器还没开工，当前时间就是 0
                # 2. 换算成现实中的绝对小时数
                abs_time_hours = m_current_time * time_unit_hours
                time_of_day = abs_time_hours % 24.0
                # 3. 查表获取实时电价
                current_price = 0.0
                for (p_start, p_end, price) in self.configs.tou_price_table:
                    if p_start <= time_of_day < p_end:
                        current_price = price
                        break
                # 4. 写入机器特征矩阵的第 5 个维度 (索引为 4)
                # 因为神经网络喜欢归一化的输入，而电价通常在 0.35 到 1.2 之间，大小很合适，直接填入即可。
                self.machines_fea[m_id][4] = current_price
            # ==============================================================================

        else:
            res = np.concatenate((adj, task_to_machine_mapping, task_to_duration_mapping), axis=1, dtype=self.dtype)

        if self.flat_observation_space:
            res = np.ravel(res).astype(self.dtype)
        if self.env_transform == 'mask':
            res = OrderedDict({
                "action_mask": np.array(self.valid_action_mask()).astype(np.int32),
                "observations": res
            })
        return res, ft_s, self.it_s, adj_wrk, tasks_fea, self.machines_fea, tasks_fea_1101, ft_s_estimated, pt_s_estimated


    def network_as_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame([
            {
                'Task': f'Job {data["job"]}', # 列1：任务名称 (例如 "Job 0")
                'Start': data["start_time"],  # 列2：开始时间
                'Finish': data["finish_time"],# 列3：结束时间
                'Resource': f'Machine {data["machine"]}' # 列4：机器资源 (例如 "Machine 1")
            }
            # 遍历图中所有的节点
            for task_id, data in self.G.nodes(data=True)
            # 过滤条件：排除虚拟节点(job != -1) 和 还没排的任务
            if data["job"] != -1 and data["finish_time"] is not None
        ])




    def valid_action_mask(self, action_mode: str = None) -> List[bool]:
        """
    Todo
        作用：告诉 Agent 当前哪些动作是“合法”的，哪些是“非法”的。
        返回类型：一个布尔列表 [True, False, True, ...]，长度等于总工序数。True 代表可以选，False 代表不能选。
        """
        # A.初始化 Mask，一开始假设所有工序都不能选（全为 False），然后我们在循环里找能选的，把它改成 True。
        if action_mode is None:
            action_mode = self.action_mode
        if action_mode == 'task':
            mask = [False] * self.total_tasks_without_dummies
            # B. 遍历所有工序，检查两个条件
            for task_id in range(1, self.total_tasks_without_dummies + 1):
                node = self.G.nodes[task_id]
                # 条件 1：如果自己已经排过了，肯定不能再排
                if node["scheduled"]:
                    continue
                # 寻找同 Job 的上一个工序
                prev_task_in_job_id, _ = list(self.G.in_edges(task_id))[0]
                prev_job_node = self.G.nodes[prev_task_in_job_id]
                # 条件 2：如果上一个工序还没排，那我也不能排
                if not prev_job_node["scheduled"]:
                    continue
                # C. 标记为合法
                mask[task_id - 1] = True
            # D.安全检查
            if True not in mask:
                # 如果发现所有全是 False（没得选了），通常意味着出 Bug 了或者结束了
                if self.verbose >= 1:
                    log.warning("no action options remaining")
                if not self.env_transform == 'mask':
                    raise RuntimeError("something went wrong")  # TODO: remove error?
            return mask
        elif action_mode == 'job':
            # 1. 先获取基础的 Task Mask
            task_mask = self.valid_action_mask(action_mode='task')
            # 2. 按 Job 切分
            masks_per_job = np.array_split(task_mask, self.n_jobs)
            # 3. 只要该 Job 里有一个工序能做，这个 Job 就是可选的
            return [True in job_mask for job_mask in masks_per_job]
        else:
            # 报错处理
            raise ValueError(f"only 'task' and 'job' are valid arguments for 'action_mode'. {action_mode} is not.")

