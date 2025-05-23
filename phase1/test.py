import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque
import datetime
import pandapower as pp
# import math
# import ipython.display import set_matplotlib_formats

plt.rcParams["font.family"] = ["SimHei"]

# netwokx网络建立初始化
# 输出：networkx网络
def initialization(p_dg):
    # 读取excel数据
    nodes = pd.read_excel('data.xlsx', sheet_name='1', header=0)
    edges = pd.read_excel('data.xlsx', sheet_name='2', header=0)

    # networkx 构建电网
    G = nx.Graph()

    # 录入excel中的62名用户，属性: p(功率), cat(分类)
    for idx, row in nodes.iterrows():
        node_id = row['No.']
        p = row['有功P/kW']
        cat = row['分类']

        attrs = {'p': p, 'cat': cat}
        G.add_node(node_id, **attrs)

    # 录入excel中的配电线路，属性: length(长度), resistance(电阻), reactance(电抗), switch
    # switch用来判断是否有开关，若为Si则是开关名，若为0则是没有开关
    for idx, row in edges.iterrows():
        start = row['起点']
        end = row['终点']
        length = row['长度/km']
        resistance = row['电阻/Ω']
        reactance = row['电抗/Ω']
        switch = row['开关']
        G.add_edge(start, end, 
                length=length, 
                resistance=resistance, 
                reactance=reactance, 
                switch=switch)

    # 添加DG，并关联到已有网络，属性: p, cat    
    G.add_node('DG16', cat='dg', p=p_dg)
    G.add_node('DG22', cat='dg', p=p_dg)
    G.add_node('DG32', cat='dg', p=p_dg)
    G.add_node('DG35', cat='dg', p=p_dg)
    G.add_node('DG39', cat='dg', p=p_dg)
    G.add_node('DG48', cat='dg', p=p_dg)
    G.add_node('DG52', cat='dg', p=p_dg)
    G.add_node('DG55', cat='dg', p=p_dg)
    G.add_edge('DG16', 16, length=0, resistance=0, reactance=0, switch=0)
    G.add_edge('DG22', 22, length=0, resistance=0, reactance=0, switch=0)
    G.add_edge('DG32', 32, length=0, resistance=0, reactance=0, switch=0)
    G.add_edge('DG35', 35, length=0, resistance=0, reactance=0, switch=0)
    G.add_edge('DG39', 39, length=0, resistance=0, reactance=0, switch=0)
    G.add_edge('DG48', 48, length=0, resistance=0, reactance=0, switch=0)
    G.add_edge('DG52', 52, length=0, resistance=0, reactance=0, switch=0)
    G.add_edge('DG55', 55, length=0, resistance=0, reactance=0, switch=0)

    #添加CB，并关联到已有网络，属性: p, cat
    G.add_node('CB1', cat='cb', p=2200)
    G.add_node('CB2', cat='cb', p=2200)
    G.add_node('CB3', cat='cb', p=2200)
    G.add_edge('CB1', 1, length=0, resistance=0, reactance=0, switch=0)
    G.add_edge('CB2', 43, length=0, resistance=0, reactance=0, switch=0)
    G.add_edge('CB3', 23, length=0, resistance=0, reactance=0, switch=0)

    #添加跨线开关，但不关联到已有网络，属性: cat, p
    G.add_node('S13-1', cat='link', p=0)
    G.add_node('S29-2', cat='link', p=0)
    G.add_node('S62-3', cat='link', p=0)
    # G.add_edge('S13-1', 13, length=0, resistance=0, reactance=0, switch=0)
    # G.add_edge('S13-1', 43, length=0, resistance=0, reactance=0, switch=0)
    # G.add_edge('S29-2', 19, length=0, resistance=0, reactance=0, switch=0)
    # G.add_edge('S29-2', 29, length=0, resistance=0, reactance=0, switch=0)
    # G.add_edge('S62-3', 23, length=0, resistance=0, reactance=0, switch=0)
    # G.add_edge('S62-3', 62, length=0, resistance=0, reactance=0, switch=0)

    return G

# 绘制电网图函数
# 输入：networkx网络
# 输出：plt.show()输出网络图，同时保存一张网络图graph.png到本地
def draw(G: nx.Graph):

    plt.figure(figsize=(15, 12))
    pos = nx.spring_layout(G, k=0.2, iterations=50, seed=42)

    node_colors = []
    # node_sizes = []
    legend_handles = []
    legend_labels = []
    cat_colors = {
        'res': 'lightblue',
        'com': 'green',
        'gov': 'red',
        'off': 'yellow',
        'link': 'orange',
        'default': 'gray'
    }
    cat_mapping = {
        'res': '居民',
        'com': '商业',
        'gov': '政府',
        'off': '办公',
        'link': '联络开关',
        'dg': '分布式电源',
        'cb': '变电站',
        'unknown': '未知'
    }

    '''
    for node in G.nodes:
        node_cat = G.nodes[node].get('cat', 'unknown')
        # print(f"节点 {node}: 分类={node_cat}\n")
        if node_cat == 'res':
            node_colors.append('lightblue')
        elif node_cat == 'com':
            node_colors.append('green')
        elif node_cat == 'gov':
            node_colors.append('red')
        elif node_cat == 'off':
            node_colors.append('yellow')
        elif node_cat == 'link':
            node_colors.append('orange')
        else:
            node_colors.append('gray')
        '''
        
    for node in G.nodes:
        node_cat = G.nodes[node].get('cat', 'unknown')
        color = cat_colors.get(node_cat, cat_colors['default'])
        node_colors.append(color)
        chinese_cat = cat_mapping.get(node_cat, '未知')
        # 创建图例
        # if node_cat not in legend_labels:
        if chinese_cat not in legend_labels:
            handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
            legend_handles.append(handle)
            legend_labels.append(chinese_cat)

    # 绘制节点和配电线
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.6, edge_color='gray')

    # 节点注释
    # 简单注释
    node_labels = {node: f"{node}\n" for node in G.nodes}
    # 完整注释
    # node_labels = {node: f"{node}\ncat: {G.nodes[node].get('cat', 'unknown')}\np: {G.nodes[node].get('p', 'unknown')}kW" for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # 配电线注释
    # 简单注释
    edge_labels = {(u, v): f"{G[u][v].get('switch', 'unknown')}" for u, v in G.edges}
    # 完整注释
    # edge_labels = {(u, v): f"len: {G[u][v].get('length', 'unknown')}km\nR: {G[u][v].get('resistance', 'unknown')}Ω\nX: {G[u][v].get('reactance', 'unknown')}Ω\nswitch: {G[u][v].get('switch', 'unknown')}" for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # 添加图例
    plt.legend(legend_handles, legend_labels, loc='upper right')

    plt.savefig('graph.png', dpi=300)
    # set_matplotlib_formats('retina')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# nx.draw(G)

# 初始化元件故障率
# 输入：networkx网络
# 输出：所有节点故障率，所有边故障率，所有节点负荷功率，以及一个拥有所有开关的开关列表
def failure_rate(G: nx.Graph) -> Tuple[Dict, Dict, Dict, Dict]:
    node_failure_rates = {}
    for node, data in G.nodes(data=True):
        if data['cat'] in ['res', 'com', 'gov', 'off']:
            node_failure_rates[node] = 0.005
        elif data['cat'] == 'dg':
            node_failure_rates[node] = 0.005
        elif data['cat'] == 'cb':
            node_failure_rates[node] = 0.002
        elif data['cat'] == 'link':
            node_failure_rates[node] = 0.002
    
    edge_failure_rates = {}
    for u, v, data in G.edges(data=True):
        length = data.get('length', 1)
        edge_failure_rates[(u, v)] = 0.002 * length
        if 'switch' in data and data['switch'] != 0:
            edge_failure_rates[(u, v)] += 0.002
    
    # 节点负荷功率
    node_loads = {node: data['p'] for node, data in G.nodes(data=True)
                  if data['cat'] in ['res', 'com', 'gov', 'off']}
    
    # 开关列表，包括普通的开关和联络开关，格式为：switches['S1'] = (1, 2)
    switches = {}
    # 添加普通开关
    for u, v, data in G.edges(data=True):
        if 'switch' in data and data['switch'] != 0:
            switches[data['switch']] = (u, v)
    # 添加联络开关
    for node, data in G.nodes(data=True):
        if data['cat'] == 'link':
            neighbors = list(G.neighbors(node))
            if len(neighbors) >= 2:
                switches[node] = (neighbors[0], neighbors[1])
            # for neighbor in G.neighbors(node):
            #     if G.has_edge(node, neighbor):
            #         switches[node] = (node, neighbor)
            #         break
    
    return node_failure_rates, edge_failure_rates, node_loads, switches

# 潮流分析，用于分析电网中各节点的供电状态
# 输入：networkx网络，所有节点负荷功率，联络开关的最大运载功率
# 输出：一个字典，用来表示节点通电状态，格式：{节点ID: (是否通电，分配的功率)}
def power_analysis(G: nx.Graph, node_loads: Dict, limit: float) -> Dict:
    # G_copy = G.copy()
    node_status = {node: (False, 0) for node in G.nodes()}
    # 获取所有电源的功率
    power_sources = {}
    for node, data in G.nodes(data=True):
        if data.get('cat') == 'cb' or data.get('cat') == 'dg':
            power_sources[node] = data.get('p', 0)

    # BFS，从电源节点开始对所有用户节点进行功率分配
    for src, power in power_sources.items():
        visited = set()
        queue = deque([(src, power)])
        node_status[src] = (True, power)
        visited.add(src)

        while queue:
            cur_node, remain_power = queue.popleft()

            for neighbor in G.neighbors(cur_node):
                if neighbor in visited:
                    continue
            
                # 如果是用户节点，就分配功率
                if G.nodes[neighbor].get('cat') in ['res', 'com', 'gov', 'off']:
                    require_power = node_loads.get(neighbor, 0)
                    allocate_power = min(require_power, remain_power)

                    is_supplied = allocate_power >= require_power
                    node_status[neighbor] = (is_supplied, allocate_power)

                    new_remain = remain_power - allocate_power
                # 如果是联络开关，直接分配功率(p=0)，但联络开关能通过的功率不超过其最大运载功率限制
                elif G.nodes[neighbor].get('cat') == 'link':
                    allocate_power = 0
                    new_remain = min(remain_power, limit)
                    node_status[neighbor] = (True, allocate_power)
                # 如果不是用户节点，直接跳过
                else:
                    allocate_power = 0
                    new_remain = remain_power
                
                # 继续按顺序分配功率
                if new_remain > 0:
                    visited.add(neighbor)
                    queue.append((neighbor, new_remain))

    return node_status

# 模拟单次故障
# 输入：networkx网络，所有节点故障率，所有边故障率，所有节点负荷功率，开关列表，联络开关的最大运载功率
# 输出：经处理的networkx网络，是否有故障(0 or 1)，失负荷量，各节点供电状态的数组
def single_failure(G: nx.Graph,
                   node_failure_rates: Dict,
                   edge_failure_rates: Dict,
                   node_loads: Dict,
                   switches: Dict,
                   limit: float) -> Tuple[nx.Graph, List, float, Dict]:
    G_copy = G.copy()

    # 移除联络开关相关的边
    contact_edges = [
        ('S13-1', 13), ('S13-1', 43),
        ('S29-2', 19), ('S29-2', 29),
        ('S62-3', 23), ('S62-3', 62)
    ]
    for u, v in contact_edges:
        if G_copy.has_edge(u, v):
            G_copy.remove_edge(u, v)

    # 初始化四类故障元件
    user = [node for node in node_failure_rates if G.nodes[node]['cat'] in ['res', 'com', 'gov', 'off']]
    dg = [node for node in node_failure_rates if G.nodes[node]['cat'] == 'dg']
    switch = list(switches.keys())
    edges = list(edge_failure_rates.keys())
    failed_user = None
    failed_dg = None
    failed_switch = None
    failed_edge = None
    is_failed = [] # 判断是否有故障发生

    '''
    # 随机用户故障
    if user and np.random.rand() < sum(node_failure_rates[node] for node in user):
        user_probs = [node_failure_rates[node] for node in user]
        total_user_rate = sum(user_probs)
        user_probs = [p / total_user_rate for p in user_probs]
        failed_user = np.random.choice(user, p=user_probs)
        is_failed.append(failed_user)

    # 随机DG故障
    if dg and np.random.rand() < sum(node_failure_rates[node] for node in dg):
        dg_probs = [node_failure_rates[node] for node in dg]
        total_dg_rate = sum(dg_probs)
        dg_probs = [p / total_dg_rate for p in dg_probs]
        failed_dg = np.random.choice(dg, p=dg_probs)
        is_failed.append(failed_dg)
    
    # 随机开关故障
    if switch and np.random.rand() < sum(node_failure_rates.get(switch, 0) for switch in switch):
        switch_probs = [node_failure_rates.get(switch, 0) for switch in switch]
        total_switch_rate = sum(switch_probs)
        switch_probs = [p / total_switch_rate for p in switch_probs]
        failed_switch = np.random.choice(switch, p=switch_probs)
        is_failed.append(failed_switch)
    
    # 随机线路故障
    if edges and np.random.rand() < sum(edge_failure_rates.values()):
        edge_probs = [edge_failure_rates[edge] for edge in edges]
        total_edge_rate = sum(edge_probs)
        edge_probs = [p / total_edge_rate for p in edge_probs]
        edge_copy = np.arange(len(edges))
        selected_index = np.random.choice(edge_copy, p=edge_probs)
        # failed_edge = np.random.choice(edges, p=edge_probs)
        # is_failed.append(failed_edge)
        failed_edge = edges[selected_index]
        is_failed.append(failed_edge)
    '''

    # 随机用户故障
    if user:
        user_probs = [node_failure_rates[node] for node in user]
        total_user_rate = sum(user_probs)
        if total_user_rate > 0:
            user_probs = [p / total_user_rate for p in user_probs]
            if np.random.rand() < total_user_rate:
                failed_user = np.random.choice(user, p=user_probs)
                is_failed.append(failed_user)

    # 随机DG故障
    if dg:
        dg_probs = [node_failure_rates[node] for node in dg]
        total_dg_rate = sum(dg_probs)
        if total_dg_rate > 0:
            dg_probs = [p / total_dg_rate for p in dg_probs]
            if np.random.rand() < total_dg_rate:
                failed_dg = np.random.choice(dg, p=dg_probs)
                is_failed.append(failed_dg)

    # 随机开关故障
    if switch:
        switch_probs = [node_failure_rates.get(switch, 0) for switch in switch]
        total_switch_rate = sum(switch_probs)
        if total_switch_rate > 0:
            switch_probs = [p / total_switch_rate for p in switch_probs]
            if np.random.rand() < total_switch_rate:
                failed_switch = np.random.choice(switch, p=switch_probs)
                is_failed.append(failed_switch)

    # 随机线路故障
    if edges:
        edge_probs = [edge_failure_rates[edge] for edge in edges]
        total_edge_rate = sum(edge_probs)
        if total_edge_rate > 0:
            edge_probs = [p / total_edge_rate for p in edge_probs]
            if np.random.rand() < total_edge_rate:
                edge_copy = np.arange(len(edges))
                selected_index = np.random.choice(edge_copy, p=edge_probs)
                failed_edge = edges[selected_index]
                is_failed.append(failed_edge)
    
    # 应用故障
    # 如果用户故障，移除故障用户节点
    if failed_user is not None:
        G_copy.remove_node(failed_user)
    
    # 如果DG故障，移除故障DG节点
    if failed_dg is not None:
        G_copy.remove_node(failed_dg)

    # 如果开关故障，移除开关对应的线路
    if failed_switch is not None:
        if failed_switch in switches:
            u, v = switches[failed_switch]
            if G_copy.has_edge(u, v):
                G_copy.remove_edge(u, v)
    
    # 如果线路故障，移除故障线路
    if failed_edge is not None:
        u, v = failed_edge
        if G_copy.has_edge(u, v):
            G_copy.remove_edge(u, v)
    
    #print(is_failed)
    # 如果有故障发生
    if is_failed:
        # for link_switch in [node for node, data in G.nodes(data=True) if data['cat'] == 'link']:
        #     neighbors = list(G.neighbors(link_switch))
        #     if len(neighbors) >= 2:
                # u, v = neighbors[0], neighbors[1]
                # if not G_copy.has_edge(u, v):
                    # G_copy.add_edge(u, v, switch=0, length=0, resistance=0, reactance=0)

        # 把联络开关连接到电路里
        G_copy.add_edge('S13-1', 13, length=0, resistance=0, reactance=0, switch=0)
        G_copy.add_edge('S13-1', 43, length=0, resistance=0, reactance=0, switch=0)
        G_copy.add_edge('S29-2', 19, length=0, resistance=0, reactance=0, switch=0)
        G_copy.add_edge('S29-2', 29, length=0, resistance=0, reactance=0, switch=0)
        G_copy.add_edge('S62-3', 23, length=0, resistance=0, reactance=0, switch=0)
        G_copy.add_edge('S62-3', 62, length=0, resistance=0, reactance=0, switch=0)


            # 如果此时联通开关是断开的，就闭合联通开关
            # if (G_copy.has_edge(link_switch, u) and 
            #        G_copy.has_edge(link_switch, v) and
            #        G_copy[link_switch][u]['switch'] == 0 and
            #        G_copy[link_switch][v]['switch'] == 0):
            #        
            #        G_copy[link_switch][u]['switch'] = 1
            #        G_copy[link_switch][v]['switch'] = 1

        # 通过功率计算各节点供电状态
        node_status = power_analysis(G_copy, node_loads, limit)

        # 计算失负荷量
        lost_load = sum(node_loads[node] for node in node_loads
                        if node in node_status and not node_status[node][0])

        return G_copy, is_failed, lost_load, node_status
    # 如果没有故障发生
    else:
        node_status = power_analysis(G_copy, node_loads, limit)
        lost_load = sum(node_loads[node] for node in node_loads
                        if node in node_status and not node_status[node][0])
        return G_copy, [], lost_load, node_status


    
def get_feeder_nodes(G: nx.Graph, cb_node: str) -> set:
    """
    获取指定CB的供电区域（不包含联络开关节点）
    - G: 当前网络拓扑
    - cb_node: 起始CB节点（CB1/CB2/CB3）
    - 返回值: {供电区域节点}
    """
    visited = set()
    queue = deque([cb_node])
    visited.add(cb_node)
    
    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                # 排除联络开关节点
                if G.nodes[neighbor].get('cat') != 'link':
                    visited.add(neighbor)
                    queue.append(neighbor)
    return visited

def check_overload(G: nx.Graph, node_loads: Dict, ol_hazard_num: float,contact_limit: float = 1.5e3) -> float:
    """
    检查馈线是否过载，并通过联络开关转移功率
    - G: 当前网络拓扑
    - node_loads: 节点负荷功率字典
    - contact_limit: 联络开关最大转供功率（默认1.5MW）
    - 返回值: 过载危害值
    """
    overload_hazard = 0.0
    V_line = 10.0  # kV
    I_threshold = 242.0  # A:过负荷阈值
    threshold = I_threshold * V_line * np.sqrt(3)  # kW:过载阈值对应的功率
    # 获取各CB的供电区域
    # feeder1_nodes = get_feeder_nodes(G, 'CB1')
    # feeder2_nodes = get_feeder_nodes(G, 'CB2')
    # feeder3_nodes = get_feeder_nodes(G, 'CB3')
    # print(f"CB1节点：{feeder1_nodes}"
    #       f"\nCB2节点：{feeder2_nodes}"
    #       f"\nCB3节点：{feeder3_nodes}")

    # 各馈线的净功率 = 负荷总和 - DG总出力
    total_power = {
        'CB1': 0.0,
        'CB2': 0.0,
        'CB3': 0.0
    }

    excess = {}
    capacity = {}
    for cb in ['CB1', 'CB2', 'CB3']:
        feeder_nodes = get_feeder_nodes(G, cb) # 获取各CB的供电区域
        load_sum = sum(node_loads.get(node, 0) for node in feeder_nodes) # 计算各CB的总负荷
        dg_sum = 0.0
        for dg_node in ['DG16', 'DG22', 'DG32', 'DG35', 'DG39', 'DG48', 'DG52', 'DG55']:
            if dg_node in feeder_nodes:
                dg_sum += G.nodes[dg_node]['p'] # 各CB的DG出力
        total_power[cb] = max(0,load_sum - dg_sum)  # 净功率 = 负荷总和 - DG出力 同时考虑功率不能倒送

        # 处理DG过剩功率转移
        # 过剩功率 = DG出力 - 负荷（若DG出力 > 负荷）
        excess[cb] = max(0.0, dg_sum - load_sum)
        # 剩余容量 = 馈线最大允许功率 - 净功率
        capacity[cb] = threshold - total_power[cb]

    # 迭代处理过载，直到无法转移或所有馈线不超载
    changed = True
    while changed:
        changed = False
        # DG出力过多时，多余的出力向其它馈线转移
        for from_cb in ['CB1', 'CB2', 'CB3']:
            if excess[from_cb] > 0:
                # 找到相邻馈线，按剩余容量排序（优先转移给剩余容量大的）
                if from_cb == 'CB1':
                    adjacent = ['CB2', 'CB3']
                elif from_cb == 'CB2':
                    adjacent = ['CB1', 'CB3']
                else:
                    adjacent = ['CB1', 'CB2']
                # 按剩余容量从高到低排序
                sorted_adj = sorted(adjacent, key=lambda x: capacity[x], reverse=True)
                for to_cb in sorted_adj:
                    transferable = min(excess[from_cb], capacity[to_cb], contact_limit)
                    if transferable > 0:
                        # 执行转移
                        excess[from_cb] -= transferable
                        capacity[to_cb] -= transferable
                        # 更新净功率
                        total_power[from_cb] += transferable  # 出方净功率增加（因转移出去）
                        total_power[to_cb] -= transferable    # 接收方净功率减少（因接收功率）
                        changed = True
                        break  # 转移后跳出循环，继续处理下一个馈线
        
    # 步骤2：处理过载（与之前逻辑相同）
    for cb in ['CB1', 'CB2', 'CB3']:
        if total_power[cb] > threshold:
            overload_amount = total_power[cb] - threshold
            overload_hazard += overload_amount * ol_hazard_num

    return overload_hazard

# 蒙特卡洛模拟，进行失负荷计算
# 输入：networkx网络，所有节点故障率，所有边故障率，所有节点负荷功率，开关列表，不同类型用户的危害系数，联络开关的最大运载功率，模拟次数
# 输出：平均失负荷风险值，平均失负荷功率值，失负荷的节点列表
def montecarlo(
        G: nx.Graph,
        node_failure_rates: Dict,
        edge_failure_rates: Dict,
        node_loads: Dict,
        switches: Dict,
        hazard_num: Dict,
        ol_hazard_num: float,
        limit: float,
        simulation_times: int = 10000) -> Tuple[float, float, List[float],float,float,float]:
    total_hazard = 0.0
    total_lost_load = 0.0
    hazard_list = []
    total_overload_hazard = 0.0 # 过负荷风险
    lostload_times = 0  # 失负荷次数
    overload_times = 0  # 过负荷次数

    for i in range(simulation_times):
        # 生成DG的随机出力（正态分布）
        G_temp = G.copy()
        for dg_node in ['DG16', 'DG22', 'DG32', 'DG35', 'DG39', 'DG48', 'DG52', 'DG55']:
            mean = G_temp.nodes[dg_node]['p']
            std = mean * 0.2    # 波动标准差设为额定容量的20%
            perturbed_p = np.random.normal(mean, std) # 设置DG出力为正态分布
            perturbed_p = max(0, perturbed_p)  # 确保功率不为负
            G_temp.nodes[dg_node]['p'] = perturbed_p
        
        # 模拟单次故障
        G_copy, is_failed, lost_load, node_status = single_failure(
            G_temp, node_failure_rates, edge_failure_rates, node_loads, switches, limit)
        
        
        # 如果用户没有供电(node_status=False)，计算危害度C，并将危害度累加起来
        hazard = 0.0
        for node in node_loads:
            if node in node_status and not node_status[node][0]:
                user_cat = G.nodes[node]['cat']
                hazard += node_loads[node] * hazard_num[user_cat] # C=节点功率*危害度系数
        
        if hazard > 0:  # 如果发生失负荷，失负荷次数加1
            lostload_times += 1
        
        total_hazard += hazard
        total_lost_load += lost_load
        hazard_list.append(hazard)

        overload_hazard = check_overload(G_copy, node_loads, ol_hazard_num ,contact_limit=1.5e6)
        if overload_hazard > 0:  # 如果发生过载，过载次数加1
            overload_times += 1
        total_overload_hazard += overload_hazard

        # if (i+1) % 1000 == 0:
        #    print(f"完成{i+1}/{simulation_times}次模拟")

        stride(i + 1, simulation_times)
    
    # 计算平均失负荷风险，计算平均失负荷量
    avg_hazard = total_hazard / simulation_times
    avg_lost_load = total_lost_load / simulation_times
    # 计算平均过负荷风险
    avg_overload_hazard = total_overload_hazard / simulation_times
    # 计算失负荷和过负荷频率，并用频率趋近概率
    lostload_prob = lostload_times / simulation_times
    overload_prob = overload_times / simulation_times
    #print(overload_times)

    return avg_hazard, avg_lost_load, hazard_list, avg_overload_hazard,lostload_prob, overload_prob

# 一个用来打印进度条的函数
# 输入：当前循环次数，总循环次数
# 输出：一个进度条
def stride(iter, total, length=100):
    # percent = "{:.1f}".format(100 * (iter / float(total)))
    filled_length = int(length * iter // total)
    bar = '#' * filled_length + ' ' * (length - filled_length)
    print(f'\r进度: |{bar}| {iter}/{total}', end='', flush=True)
    if iter == total:
        print()

def simulate_and_draw(init_p_dg, step, max_p_dg, simulation_times, limit, hazard_num, ol_hazard_num):
    dg_capacity = np.arange(init_p_dg, max_p_dg + step, step)
    risk_list = []
    lostload_prob_list = []
    overload_prob_list = []
    overload_risk_list = []
    r_sys_list = []

    for p_dg in dg_capacity:
        print(f"当前DG容量: {p_dg} kW")

        G = initialization(p_dg)
        # draw(G)
        node_failure_rates, edge_failure_rates, node_loads, switches = failure_rate(G)
        # print(node_failure_rates)
        # print(edge_failure_rates)
        # print(node_loads)
        # print(switches)

        # G_copy, is_failed, lost_load, node_status = single_failure(G, node_failure_rates, edge_failure_rates, node_loads, switches)
        # print(G_copy)
        # print(is_failed)
        # print(lost_load)
        # print(node_status)

        risk, avg_lost, hazard_list,overload_risk,lostload_prob,overload_prob = montecarlo(
        G, node_failure_rates, edge_failure_rates, node_loads, switches, hazard_num, ol_hazard_num, simulation_times=simulation_times, limit=limit)

        r_sys = risk + overload_risk

        print(f"系统失负荷风险: {risk:.2f} kW·危害系数")
        print(f"系统过负荷风险: {overload_risk:.2f} kW·危害系数")
        # draw(G_copy)
        print(f"失负荷概率: {lostload_prob:.4f}")
        print(f"过负荷概率: {overload_prob:.4f}")
        print(f"系统风险：{r_sys:.4f} kW·危害系数")

        risk_list.append(risk)
        lostload_prob_list.append(lostload_prob)
        overload_risk_list.append(overload_risk)
        overload_prob_list.append(overload_prob)
        r_sys_list.append(r_sys)

    # _, axes = plt.subplots(3, 1, figsize=(12, 6))

    
    # 绘制失负荷风险和失负荷概率图
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('DG容量 (kW)')
    ax1.set_ylabel('失负荷风险 (kW·危害系数)', color='black')
    line1 = ax1.plot(dg_capacity, risk_list, label='失负荷风险', color='blue')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.set_ylabel('失负荷概率', color='black')
    line2 = ax2.plot(dg_capacity, lostload_prob_list, label='失负荷概率', color='red')
    ax2.tick_params(axis='y', labelcolor='black')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    ax1.set_title('失负荷风险和失负荷概率')
    plt.savefig('lostload.png', dpi=300)
    plt.show()

    # 绘制过负荷风险和过负荷概率图
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    ax3.set_xlabel('DG容量 (kW)')
    ax3.set_ylabel('过负荷风险 (kW·危害系数)', color='black')
    line3 = ax3.plot(dg_capacity, overload_risk_list, label='过负荷风险', color='blue')
    ax3.tick_params(axis='y', labelcolor='black')

    ax4 = ax3.twinx()
    ax4.set_ylabel('过负荷概率', color='black')
    line4 = ax4.plot(dg_capacity, overload_prob_list, label='过负荷概率', color='red')
    ax4.tick_params(axis='y', labelcolor='black')

    lines = line3 + line4
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')

    ax3.set_title('过负荷风险和过负荷概率')
    plt.savefig('overload.png', dpi=300)
    plt.show()

    # 绘制系统风险图
    plt.figure(figsize=(6, 6))
    plt.plot(dg_capacity, r_sys_list, label='系统风险')
    plt.xlabel('DG容量 (kW)')
    plt.ylabel('系统风险 (kW·危害系数)')
    plt.title('系统风险')
    plt.legend()
    plt.savefig('rsys.png', dpi=300)
    plt.show()
    
    '''
    # 失负荷风险和失负荷概率
    ax1 = axes[0]
    ax1_r = ax1.twinx()
    ax1.plot(dg_capacity, risk_list, label='失负荷风险 (kW·危害系数)', color='blue')
    ax1_r.plot(dg_capacity, lostload_prob_list, label='失负荷概率', color='red')
    ax1.set_xlabel('DG容量 (kW)')
    ax1.set_ylabel('失负荷风险 (kW·危害系数)')
    ax1_r.set_ylabel('失负荷概率')
    ax1.set_title('失负荷风险和失负荷概率')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1_r.legend(lines + lines2, labels + labels2, loc='upper right')

    # 过负荷风险和过负荷概率
    ax2 = axes[1]
    ax2_r = ax2.twinx()
    ax2.plot(dg_capacity, overload_risk_list, label='过负荷风险 (kW·危害系数)', color='blue')
    ax2_r.plot(dg_capacity, overload_prob_list, label='过负荷概率', color='red')
    ax2.set_xlabel('DG容量 (kW)')
    ax2.set_ylabel('过负荷风险 (kW·危害系数)')
    ax2_r.set_ylabel('过负荷概率')
    ax2.set_title('过负荷风险和过负荷概率')
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2_r.legend(lines + lines2, labels + labels2, loc='upper right')

    # 系统风险
    ax3 = axes[2]
    ax3.plot(dg_capacity, r_sys_list, label='系统风险 (kW·危害系数)', color='blue')
    ax3.set_xlabel('DG容量 (kW)')
    ax3.set_ylabel('系统风险 (kW·危害系数)')
    ax3.set_title('系统风险')
    ax3.legend(loc='upper right')

    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()
    '''

    return risk_list, overload_risk_list, lostload_prob_list, overload_prob_list, r_sys_list


if __name__ == '__main__':

    # 危害度系数初始化
    # 失负荷危害度系数
    hazard_num = {
        'res': 1.0,
        'com': 1.0,
        'gov': 1.0,
        'off': 1.0
    }
    ol_hazard_num = 1.0 # 过负荷危害系数，此处不区分故障节点，因为过负荷是整条线路的故障

    simulation_times = 1000000 # 模拟次数
    limit = 1500 # 联络开关的最大负载能力(kW)
    p_dg = 300 # 每个DG的输出功率(kW)

    # starttime = datetime.datetime.now()
    
    G = initialization(p_dg)
    draw(G)
    risk, overload_risk, lostload_prob, overload_prob, r_sys = simulate_and_draw(p_dg, 0.3*p_dg, 3*p_dg, simulation_times, limit, hazard_num, ol_hazard_num)

    # print(f"设定模拟次数{simulation_times}次，联络开关最大负载能力{limit} kW，初始DG功率{p_dg} kW：")

    # endtime = datetime.datetime.now()

    # print(f"运行时间: {endtime-starttime}")

#######################
#    2025 MCM, ZJU    #
#      Question A     #
#    YUAN, YANG, LI   #
# All rights reserved #
#######################