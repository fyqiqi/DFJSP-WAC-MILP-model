from gurobipy import GRB
from matplotlib import pyplot as plt

from DFJSPDATAREAD import read_and_convert_file
from DFJSPAGVWACMIP import MIPModel

filename = 'FJSSPinstances/MK/instance3.txt'
Data = read_and_convert_file(filename)
print('data_j', Data['J'], Data['OJ'])
print('DATA_operations_machines', Data['operations_machines'])
print('DATA_operations_machines', Data['operations_times'])

# num_operation = []
# for f in Data['F']:
#     for i in Data['J']:
#         num_operation.append(Data['OJ'][(f,i)][-1])
# print(num_operation)
# num_operation_max = np.array(num_operation).max()
#
# time_window = np.zeros(shape=(Data['n'], num_operation_max, Data['m']))
#
# for i in range(len(num_operation)):
#     for j in range(num_operation[i]):
#         mchForJob = Data['operations_machines'][(i + 1, j + 1)]
#         for k in mchForJob:
#             time_window[i][j][k - 1] = Data['operations_times'][(i + 1, j + 1, k)]
# print(time_window)
# map_window = get_travel_time()
F = Data['F']
n = Data['n']
m = Data['m']
J = Data['J']
OJ = Data['OJ']
A = Data['A']
M = Data['M']
W = {}
W = [list(range(1, 3)) for _ in range(len(F))]
Data['W'] = W
operations_machines = Data['operations_machines']
operations_times = Data['operations_times']
largeM = Data['largeM']

mipmodel = MIPModel(Data)
mipmodel.setParam('OutputFlag', 1)  # 启用输出
mipmodel.optimize()

# 确保即使没有最优解也能绘制甘特图
if mipmodel.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
    # # 提取机器调度信息
    # print("\n检查决策变量x的值：")
    # for f in F:
    #     print(f"\n工厂 {f}:")
    #     for j in J:
    #         print(f"\n  作业 {j}:")
    #         for i in OJ[(f, j)]:
    #             print(f"    工序 {i}:")
    #             found = False
    #             for k in operations_machines[f, j, i]:
    #                 var = mipmodel.getVarByName(f"x_{j}_{i}_{f}_{k}")
    #                 if var.X > 0.5:
    #                     print(f"      在机器 {k} 上加工")
    #                     found = True
    #             if not found:
    #                 print("      未找到分配的机器")
    machine_schedule = {}
    for f in F:
        for j in J:
            for i in OJ[(f, j)]:
                for k in operations_machines[(f, j, i)]:
                    if mipmodel.getVarByName(f"Y_{j}_{i}_{f}_{k}").X > 0.5:
                        start = mipmodel.getVarByName(f"s_{j}_{i}_{f}").X
                        duration = operations_times[(f, j, i, k)]
                        machine_schedule.setdefault(f"{f}_{k}", []).append({
                            'job': j,
                            'operation': i,
                            'factory': f,
                            'start': start,
                            'end': start + duration,
                            'duration': duration
                        })

    # 提取AGV运输信息
    agv_schedule = {}
    for f in F:
        for j in J:
            for i in OJ[(f, j)]:
                if mipmodel.getVarByName(f"eta_{j}_{i}_{f}").X > 0.5:
                    for r in A[f-1]:
                        if mipmodel.getVarByName(f"Z_{j}_{i}_{f}_{r}").X > 0.5:
                            start = mipmodel.getVarByName(f"ST_{j}_{i}_{f}").X
                            duration = mipmodel.getVarByName(f"Tt_{j}_{i}_{f}").X
                            agv_schedule.setdefault(f"{f}_{r}", []).append({
                                'job': j,
                                'operation': i,
                                'factory': f,
                                'start': start,
                                'end': start + duration,
                                'duration': duration
                            })

    # 提取工人调度信息
    worker_schedule = {}
    for f in F:
        for j in J:
            for i in OJ[(f, j)]:
                for w in W[f-1]:
                    var = mipmodel.getVarByName(f"V_{j}_{i}_{f}_{w}")
                    for k in operations_machines[(f, j, i)]:
                        var2 = mipmodel.getVarByName(f"Y_{j}_{i}_{f}_{k}")
                        if var is not None and var.X > 0.5 and var2.X > 0.5:
                            start = mipmodel.getVarByName(f"SR_{j}_{i}_{f}").X
                            duration = operations_times[f,j,i,k]/2  # monitoring_time
                            worker_schedule.setdefault(f"{f}_{w}", []).append({
                                'job': j,
                                'operation': i,
                                'factory': f,
                                'start': start,
                                'end': start + duration,
                                'duration': duration,
                                'type': 'monitor'
                            })

    worker_transport_schedule = {}
    for f in F:
        for j in J:
            for i in OJ[(f, j)]:
                for w in W[f-1]:
                    var = mipmodel.getVarByName(f"W_{j}_{i}_{f}_{w}")
                    if var is not None and var.X > 0.5:
                        start = mipmodel.getVarByName(f"ST_{j}_{i}_{f}").X
                        duration = mipmodel.getVarByName(f"Tt_{j}_{i}_{f}").X
                        worker_schedule[f"{f}_{w}"].append({
                            'job': j,
                            'operation': i,
                            'factory': f,
                            'start': start,
                            'end': start + duration,
                            'duration': duration,
                            'type': 'transport'
                        })

    # 绘制甘特图（为每个工厂创建单独的图）
    for f in F:
        plt.figure(figsize=(12, 8))
        plt.title(f'Schedule Gantt Chart - Factory {f}')

        y_ticks = []
        y_labels = []
        colors = plt.cm.tab20.colors
        current_y = 0

        # 绘制该工厂的机器
        for k in M[f-1]:
            if f"{f}_{k}" in machine_schedule:
                y_ticks.append(current_y)
                y_labels.append(f"Machine {k}")
                for op in machine_schedule[f"{f}_{k}"]:
                    plt.barh(current_y, op['duration'], left=op['start'],
                             color=colors[op['job'] % 20], edgecolor='black')
                current_y += 2

        # 绘制该工厂的AGV
        for r in A[f-1]:
            if f"{f}_{r}" in agv_schedule:
                y_ticks.append(current_y)
                y_labels.append(f"AGV {r}")
                for task in agv_schedule[f"{f}_{r}"]:
                    plt.barh(current_y, task['duration'], left=task['start'],
                             color=colors[task['job'] % 20], edgecolor='black', alpha=0.7)
                current_y += 2

        # 绘制该工厂的工人
        # 修改工人任务的显示
        for w in W[f - 1]:
            y_ticks.append(current_y)
            y_labels.append(f"Worker {w}")
            if f"{f}_{w}" in worker_schedule:
                for task in worker_schedule[f"{f}_{w}"]:
                    pattern = '//' if task['type'] == 'monitor' else '\\\\'
                    bar = plt.barh(current_y, task['duration'], left=task['start'],
                                   color=colors[task['job'] % 20], edgecolor='black',
                                   alpha=0.7, hatch=pattern)
            current_y += 2

        plt.yticks(y_ticks, y_labels)
        plt.xlabel('Time')
        plt.ylabel('Resources')
        plt.grid(True)
        plt.show()

else:
    print("No solution found")
