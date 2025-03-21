from gurobipy import Model, GRB, quicksum
import sys
from map import get_travel_time

def MIPModel(Data):
    n = Data['n']
    m = Data['m']
    J = Data['J']
    A = Data['A']
    OJ = Data['OJ']
    W = Data['W']
    F = Data['F']
    M = Data['M']
    operations_machines = Data['operations_machines']
    operations_times = Data['operations_times']
    largeM = Data['largeM']
    TT = get_travel_time()
    monitoring_time = 10
    model = Model("FJSP_PPF")
    model.Params.OutputFlag = 0
    # Decision Variables
    x = {}      # φ[j,i,k]
    eta = {}    # η[j,i]
    mu = {}     # μ[j,i,r]
    z = {}      # z[j1,i1,j2,i2,r]
    s = {}      # SO[j,i]
    ST = {}     # ST[j,i]
    Tt = {}     # Tt[j,i]
    msq = {}
    cmax = model.addVar(vtype=GRB.INTEGER, name="cmax")
    gamma = {}  # Worker assignment to operations
    beta = {}  # Worker sequence between operations
    MT = {}  # Monitoring start time
    theta = {}  # Whether operation is the first task for a worker
    mu_w = {}  # Whether transport task is assigned to worker
    # Initialize worker-related variables
    # 新决策变量定义
    mon_trans = {}  # 监控任务早于运输任务
    trans_mon = {}  # 运输任务早于监控任务
    # Define a new variable to assign jobs to factories
    fac_y = {}  # y[j,f] = 1 if job j is assigned to factory f, 0 otherwise
    for j in J:
        for f in F:  # F is the set of factories
            fac_y[j, f] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{f}")

    # Each job must be assigned to exactly one factory
    for j in J:
        model.addConstr(
            quicksum(fac_y[j, f] for f in F) == 1,
            f"job_factory_assignment_{j}"
        )
    for f in F:
        for w in W[f-1]:
            for j1 in J:
                for i1 in OJ[(f,j1)]:
                    for j2 in J:
                        for i2 in OJ[(f,j2)]:
                            if (j1, i1) != (j2, i2):
                                mon_trans[j1, i1, j2, i2, f, w] = model.addVar(vtype=GRB.BINARY,
                                                                            name=f"mon_trans_{j1}_{i1}_{j2}_{i2}_{w}")
                                trans_mon[j1, i1, j2, i2, f, w] = model.addVar(vtype=GRB.BINARY,
                                                                            name=f"trans_mon_{j1}_{i1}_{j2}_{i2}_{w}")
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                MT[j, i, f] = model.addVar(vtype=GRB.INTEGER, name=f"MT_{j}_{i}_{f}") # Monitoring start time
                for w in W[f-1]:
                    gamma[j, i,f, w] = model.addVar(vtype=GRB.BINARY, name=f"gamma_{j}_{i}_{f}_{w}") # Worker assignment
                    theta[j, i,f, w] = model.addVar(vtype=GRB.BINARY, name=f"theta_{j}_{i}_{w}") # First task for worker
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                s[j, i,f] = model.addVar(vtype=GRB.INTEGER, name=f"s_{j}_{i}_{f}")
                ST[j, i,f] = model.addVar(vtype=GRB.INTEGER, name=f"ST_{j}_{i}_{f}")
                Tt[j, i, f] = model.addVar(vtype=GRB.INTEGER, name=f"Tt_{j}_{i}_{f}")
                eta[j, i, f] = model.addVar(vtype=GRB.BINARY, name=f"eta_{j}_{i}_{f}")
                for k in operations_machines[f,j, i]:
                    x[j, i,f, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{i}_{f}_{k}")
                for r in A[f-1]:
                    mu[j, i,f,r] = model.addVar(vtype=GRB.BINARY, name=f"mu_{j}_{i}_{f}_{r}")
                for w in W[f-1]:
                    mu_w[j,i,f,w] = model.addVar(vtype=GRB.BINARY, name=f"mu_w_{j}_{i}_{f}_{w}")

    # Define beta variable
    for f in F:
        for j1 in J:
            for j2 in J:
                for i1 in OJ[(f, j1)]:
                    for i2 in OJ[(f, j2)]:
                        for w in W[f-1]:
                            beta[j1, i1, j2, i2, f, w] = model.addVar(vtype=GRB.BINARY, name=f"beta_{j1}_{i1}_{j2}_{i2}_{w}")
    for f in F:
        for r in A[f-1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f,j1)]:
                        for i2 in OJ[(f,j2)]:
                            z[j1, i1, j2, i2, f, r] = model.addVar(vtype=GRB.BINARY, name=f"z_{j1}_{i1}_{j2}_{i2}_{r}")
    for f in F:
        for k in M[f-1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f,j1)]:
                        for i2 in OJ[(f,j2)]:
                            msq[j1, i1, j2, i2, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{j1}_{i1}_{j2}_{i2}_{k}")  #
    # Objective
    model.setObjective(cmax, GRB.MINIMIZE)
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                model.addConstr(
                    quicksum(gamma[j, i, f, w] for w in W[f-1]) == fac_y[j,f],
                    f"worker_assignment_{j}_{i}"
                )

    #     # 2. Monitoring time must be during operation processing
    # for f in F:
    #     for j in J:
    #         for i in OJ[(f,j)]:
    #             # Monitoring starts after operation start
    #             model.addConstr(
    #                 MT[j, i,f] >= s[j, i, f],
    #                 f"monitoring_start_{j}_{i}"
    #             )
            # Monitoring completes before operation ends
    for f in F:
        for w in W[f - 1]:
            # 每个工人最多有一个首任务
            model.addConstr(
                quicksum(theta[j, i, f, w] for j in J for i in OJ[(f, j)]) <= 1,
                f"worker_first_task_upper_{w}"
            )
            # 如果工人被分配了任务，则必须有一个首任务
            model.addConstr(
                quicksum(theta[j, i, f, w] for j in J for i in OJ[(f, j)]) >=
                quicksum(gamma[j, i, f, w] + mu_w[j, i, f, w] for j in J for i in OJ[(f, j)]) / largeM,
                f"worker_first_task_lower_{w}"
            )
        # 4. Worker task sequencing constraints
    # for f in F:
    #     for j in J:
    #         for i in OJ[(f,j)]:
    #             # theta is 1 if no prior tasks and assigned to w
    #             model.addConstr(
    #                 theta[j, i,f, w] <= gamma[j, i,f, w], # If assigned to worker
    #                 f"theta_upper_{j}_{i}_{w}"
    #             )
    #             model.addConstr(
    #                 theta[j, i, f, w] <= 1 - (quicksum(beta[j1, i1, j, i, f, w]
    #                                                    for j1 in J for i1 in OJ[(f, j1)]
    #                                                    if (j1, i1) != (j, i))),
    #                 f"theta_no_prior_{j}_{i}_{w}"
    #             )
    #             model.addConstr(
    #                 theta[j, i, f,w] >= gamma[j, i,f, w] - quicksum(beta[j1, i1, j, i,f, w]
    #                                                             for j1 in J for i1 in OJ[(f,j1)]
    #                                                             if (j1, i1) != (j, i)),
    #                 f"theta_lower_{j}_{i}_{w}"
    #             )

        # 3.2 Initial movement from depot to first task
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                for w in W[f-1]:
                    for k in operations_machines[f,j, i]:
                        model.addConstr(
                            MT[j, i,f] >= (TT[0][k] - largeM * (3 - theta[j, i, f, w] - x[j, i, f, k] - gamma[j, i, f, w])),
                            f"initial_movement_{j}_{i}_{w}_{k}"
                        )
    for f in F:
        for w in W[f-1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f,j1)]:
                        for i2 in OJ[(f,j2)]:
                            if (j1, i1) != (j2, i2):  # Avoid duplicate constraints
                                model.addConstr(
                                    beta[j1, i1, j2, i2, f, w] + beta[j2, i2, j1, i1, f, w] <= fac_y[j1,f],
                                    f"worker_seq_mutex_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    beta[j1, i1, j2, i2, f, w] + beta[j2, i2, j1, i1, f, w] <= fac_y[j2, f],
                                    f"worker_seq_mutex_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    beta[j1, i1, j2, i2, f, w] + beta[j2, i2, j1, i1, f, w] >= fac_y[j1,f] + fac_y[j2,f] + gamma[j1, i1, f, w] + gamma[j2, i2, f, w]-3,
                                    f"worker_seq_mutex_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
# 3.3 Worker task sequencing constraints
    for f in F:
        for w in W[f-1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f,j1)]:
                        for i2 in OJ[(f,j2)]:
                            if (j1, i1) != (j2, i2):
                                for k1 in operations_machines[f,j1, i1]:
                                    for k2 in operations_machines[f,j2, i2]:
                                        model.addConstr(
                                            MT[j2, i2,f] >= (MT[j1, i1,f] + operations_times[f,j1,i1,k1]/2 + TT[k1][k2]
                                                           - largeM * (7 - beta[j1, i1, j2, i2, f, w]
                                                                       - gamma[j1, i1,f, w] - gamma[j2, i2,f, w]
                                                                       - x[j1, i1, f, k1] - x[j2, i2, f, k2]
                                                                       - fac_y[j1,f] - fac_y[j2,f])),
                                            f"worker_move_{j1}_{i1}_{j2}_{i2}_{w}_{k1}_{k2}"
                                        )
    for f in F:
        for w in W[f - 1]:  # 遍历所有工人
            # 收集所有可能分配到工人w的工序
            ops_on_k = [
                (j, i)
                for j in J
                for i in OJ.get((f, j), [])
            ]               # 生成所有可能的工序对
            for idx1 in range(len(ops_on_k)):
                for idx2 in range(idx1 + 1, len(ops_on_k)):
                    j1, i1 = ops_on_k[idx1]
                    j2, i2 = ops_on_k[idx2]
                    wsq = model.addVar(vtype=GRB.BINARY, name=f"wsq_{j1}_{i1}_{j2}_{i2}_{w}")
                    for k in M[f-1]:  # 遍历所有机器
                        if k in operations_machines[f, j1, i1] and k in operations_machines[f, j2, i2]:
                            model.addConstr(
                                MT[j2, i2, f] >= MT[j1, i1, f] + operations_times[f,j1,i1,k]/2  * gamma[j1, i1, f, w]
                                - largeM * (3 - x[j1,i1,f,k] - gamma[j1, i1, f, w] - gamma[j2, i2, f, w]) - largeM * (1 - wsq),
                                name=f"worker_order_forward_{j1}_{i1}_{j2}_{i2}_{w}"
                            )
                            model.addConstr(
                                MT[j1, i1, f] >= MT[j2, i2, f] + operations_times[f,j2,i2,k]/2 * gamma[j2, i2, f, w]
                                - largeM * (3 - x[j2,i2,f,k] - gamma[j1, i1, f, w] - gamma[j2, i2, f, w]) - largeM * wsq,
                                name=f"worker_order_backward_{j1}_{i1}_{j2}_{i2}_{w}"
                            )
    for f in F:
        for j in J:
            for i in OJ[(f, j)]:
                # Ensure worker monitoring starts exactly when the operation starts
                model.addConstr(
                    MT[j, i, f] == s[j, i, f] * fac_y[j, f],
                    f"monitoring_starts_with_operation_{j}_{i}"
                )
    # Constraints
    # 1. 机器分配 (Sub-problem 1)
    for f in F:
        for j in J:
            for i in OJ[(f, j)]:
                model.addConstr(
                    quicksum(x[j, i, f, k] for k in operations_machines[f, j, i]) == fac_y[j,f],
                    f"machine_assignment_{j}_{i}_{f}"
                )
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                for k in operations_machines[f,j, i]:
                    model.addConstr(
                        x[j, i, f, k] <= fac_y[j, f],
                        f"factory_machine_assignment_{j}_{i}_{f}_{k}"
                    )
        # 7. 同一机器上的工序顺序约束（新增）
    for f in F:
        for k in M[f-1]:  # 遍历所有机器
            # 收集所有可能分配到机器k的工序
            ops_on_k = [
                (j, i)
                for j in J
                for i in OJ.get((f, j), [])
                if k in operations_machines.get((f, j, i), [])
            ]            # 生成所有可能的工序对
            for idx1 in range(len(ops_on_k)):
                for idx2 in range(idx1 + 1, len(ops_on_k)):
                    j1, i1 = ops_on_k[idx1]
                    j2, i2 = ops_on_k[idx2]
                    # 创建顺序变量 sq
                    sq = model.addVar(vtype=GRB.BINARY, name=f"sq_{j1}_{i1}_{j2}_{i2}_{k}")

                    # 约束1: j1在j2之前
                    model.addConstr(
                        s[j2, i2,f] >= s[j1, i1,f] + x[j1, i1, f,k] * operations_times[f,j1, i1, k]
                        - largeM * (2 - x[j1, i1, f,k] - x[j2, i2, f, k]) - largeM * (1 - sq),
                        name=f"machine_order_forward_{j1}_{i1}_{j2}_{i2}_{k}"
                    )
                    # 约束2: j2在j1之前
                    model.addConstr(
                        s[j1, i1,f] >= s[j2, i2,f] + x[j2, i2, f, k] * operations_times[f,j2, i2, k]
                        - largeM * (2 - x[j1, i1, f, k] - x[j2, i2, f, k]) - largeM * sq,
                        name=f"machine_order_backward_{j1}_{i1}_{j2}_{i2}_{k}"
                    )

    # # 2. 运输任务存在性 (η_j)
    for f in F:
        for j in J:
            # 首工序必须存在运输任务
            first_op = OJ[(f,j)][0]
            model.addConstr(eta[ j, first_op,f] == fac_y[j,f], f"eta_first_{j}")
            # 非首工序
            for idx in range(1, len(OJ[(f,j)])):
                i = OJ[(f,j)][idx]
                prev_i = OJ[(f,j)][idx-1]
                model.addConstr(
                    eta[j, i,f] == fac_y[j,f] - quicksum(
                        x[j,i,f,k]*x[j,prev_i,f,k]
                        for k in operations_machines[f,j, prev_i]
                        if k in operations_machines[f,j, i]
                    ),
                    f"eta_{j}_{i}"
                )

    #不运输时保证ST与Tt为0
    #不运输不用考虑ST的值，随便给一个
    # for f in F:
    #     for j in J:
    #         for i in OJ[(f,j)]:
    #             model.addConstr(
    #                 (eta[j, i,f] == 0) >> (ST[j, i,f] == 0),
    #                 f"ST_zero_when_no_transport_{j}_{i}"
    #             )
    #             model.addConstr(
    #                 (eta[j, i,f] == 0) >> (Tt[j, i,f] == 0),
    #                 f"Tt_zero_when_no_transport_{j}_{i}"
    #             )
    # 3. 运输时间计算 (Tt) y为i 是否在k_pre和k_curr上面加工
    for f in F:
        for j in J:
            for idx, i in enumerate(OJ[(f,j)]):
                if idx == 0:
                    model.addConstr(Tt[j, i,f] == quicksum(x[j, i,f, k] * TT[0][k] for k in operations_machines[f,j, i]))
                else:
                    prev_i = OJ[(f,j)][idx - 1]
                    y = {}
                    for k_prev in operations_machines[f,j, prev_i]:
                        for k_curr in operations_machines[f,j, i]:
                            y[j, i, f,k_prev, k_curr] = model.addVar(vtype=GRB.BINARY)
                            model.addConstr(y[j, i, f,k_prev, k_curr] <= x[j, prev_i, f,k_prev])
                            model.addConstr(y[j, i, f,k_prev, k_curr] <= x[j, i, f,k_curr])
                            model.addConstr(y[j, i, f,k_prev, k_curr] >= x[j, prev_i,f, k_prev] + x[j, i, f,k_curr] - 1)
                    model.addConstr(
                        Tt[j, i,f] == quicksum(
                            y[j, i,f, k_prev, k_curr] * TT[k_prev][k_curr]
                            # (x[j,i,k_curr]*x[j,prev_i,k_prev]) * TT[k_prev][k_curr]
                            for k_prev in operations_machines[f,j, prev_i]
                            for k_curr in operations_machines[f,j, i]
                        ),
                        name=f"Tt_{j}_{i}"
                    )

    # 4. AGV分配 (Sub-problem 3)
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                model.addConstr(
                    quicksum(mu[j, i,f, r] for r in A[f-1])+quicksum(mu_w[j, i,f, w] for w in W[f-1]) == eta[j, i,f],
                    f"AGV_assignment_{j}_{i}"
                )
    # 定义辅助变量 当且仅当 mu[j1,i1,r] == 1 且 mu[j2,i2,r] == 1 时，w = 1：
    w = {}
    for f in F:
        for r in A[f-1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f,j1)]:
                        for i2 in OJ[(f,j2)]:
                            if (j1, i1) != (j2, i2):
                                w[j1, i1, j2, i2,f, r] = model.addVar(
                                    vtype=GRB.BINARY,
                                    name=f"w_{j1}_{i1}_{j2}_{i2}_{r}"
                                )
    for f in F:
        for r in A[f-1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f,j1)]:
                        for i2 in OJ[(f,j2)]:
                            if  (j1, i1) != (j2, i2):
                                # w <= mu[j1,i1,r]
                                model.addConstr(
                                    w[j1, i1, j2, i2,f, r] <= mu[j1, i1,f, r],
                                    name=f"w_upper1_{j1}_{i1}_{j2}_{i2}_{r}"
                                )
                                # w <= mu[j2,i2,r]
                                model.addConstr(
                                    w[j1, i1, j2, i2,f, r] <= mu[j2, i2,f, r],
                                    name=f"w_upper2_{j1}_{i1}_{j2}_{i2}_{r}"
                                )
                                # w >= mu[j1,i1,r] + mu[j2,i2,r] - 1
                                model.addConstr(
                                    w[j1, i1, j2, i2,f, r] >= mu[j1, i1,f, r] + mu[j2, i2,f, r] - 1,
                                    name=f"w_lower_{j1}_{i1}_{j2}_{i2}_{r}"
                                )
    # 5. AGV任务排序 (Sub-problem 4)
    #todo  z决策变量的值定义有误
    # 5. AGV任务排序约束（修正后的运输时间计算）\
    for f in F:
        for r in A[f-1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f,j1)]:
                        for i2 in OJ[(f,j2)]:
                            if (j1, i1) == (j2, i2):
                                continue  # 跳过相同任务
                            # 获取j2的前序工序索引
                            if i2 == OJ[(f,j2)][0]:  # 首工序
                                # 首工序的运输起点是depot（0）
                                for k1 in operations_machines[f,j1, i1]:
                                    # AGV从j1,i1的机器k1移动到depot（0），再运输到当前机器k2
                                    for k2_curr in operations_machines[f,j2, i2]:
                                        # 运输时间TT[k1][0] + 当前工序运输时间Tt[j2,i2]（已包含在ST）
                                        model.addConstr(
                                            (z[j1, i1, j2, i2,f, r] == 1) >> (
                                                    ST[j2, i2,f] >= ST[j1, i1,f] + Tt[j1, i1,f] + TT[k1][0]
                                                    - largeM * (2 - x[j1, i1,f, k1] - x[j2, i2,f, k2_curr])
                                            ),
                                            f"AGV_order_{j1}_{i1}_{j2}_{i2}_first_{r}"
                                        )
                            else:
                                # 非首工序，前序工序的机器k_prev
                                prev_i2 = OJ[(f,j2)][OJ[(f,j2)].index(i2) - 1]
                                # 遍历可能的前序机器k_prev和当前机器k_curr
                                for k1 in operations_machines[f,j1, i1]:
                                    for k_prev in operations_machines[f, j2, prev_i2]:
                                        for k_curr in operations_machines[f, j2, i2]:
                                            # AGV从j1,i1的机器k1移动到前序机器k_prev，再运输到k_curr
                                            model.addConstr(
                                                (z[j1, i1, j2, i2,f, r] == 1) >> (
                                                        ST[j2, i2,f] >= ST[j1, i1,f] + Tt[j1, i1,f] + TT[k1][k_prev]
                                                        - largeM * (3 - x[j1, i1,f, k1] - x[j2, prev_i2,f, k_prev] - x[
                                                    j2, i2,f, k_curr])
                                                ),
                                                f"AGV_order_{j1}_{i1}_{j2}_{i2}_{r}"
                                            )
                            # # 确保顺序互斥
                            # if (j1, i1) < (j2, i2):  # 避免重复处理
                            #     model.addConstr(
                            #         z[j1, i1, j2, i2,f, r] + z[j2, i2, j1, i1,f, r] <= 1,
                            #         name=f"AGV_order_mutex_{j1}_{i1}_{j2}_{i2}_{r}"
                            #     )
                            if (j1, i1) != (j2, i2):
                                model.addConstr(
                                    (w[j1, i1, j2, i2, f, r] == 1) >> (
                                            z[j1, i1, j2, i2, f, r] + z[j2, i2, j1, i1, f, r] == 1
                                    ),
                                    name=f"AGV_order_mutex_{j1}_{i1}_{j2}_{i2}_{r}"
                                )
                                # j1,i1先于j2,i2的约束
                                model.addConstr(
                                    z[j1, i1, j2, i2, f, r] <= mu[j1, i1, f, r],
                                    name=f"z_upper1_{j1}_{i1}_{j2}_{i2}_{r}"
                                )
                                model.addConstr(
                                    z[j1, i1, j2, i2, f, r] <= mu[j2, i2, f, r],
                                    name=f"z_upper2_{j1}_{i1}_{j2}_{i2}_{r}"
                                )
                                # j2,i2先于j1,i1的约束
                                model.addConstr(
                                    z[j2, i2, j1, i1, f, r] <= mu[j2, i2, f, r],
                                    name=f"z_upper1_{j2}_{i2}_{j1}_{i1}_{r}"
                                )
                                model.addConstr(
                                    z[j2, i2, j1, i1, f, r] <= mu[j1, i1, f, r],
                                    name=f"z_upper2_{j2}_{i2}_{j1}_{i1}_{r}"
                                )
    #保证加工顺序
    for f in F:
        for j in J:
            for idx in range(1, len(OJ[(f,j)])):  # 从第二道工序开始
                i = OJ[(f,j)][idx]
                prev_i = OJ[(f,j)][idx - 1]
                model.addConstr(
                    s[j, i,f] >= s[j, prev_i,f] + quicksum(
                        x[j, prev_i,f, k] * operations_times[f,j, prev_i,k]
                        for k in operations_machines[f,j, prev_i]
                    ),
                    name=f"op_sequence_{j}_{i}"
                )
    #运输时间在加工完成之后
    for f in F:
        for j in J:
            for idx in range(1, len(OJ[(f,j)])):
                i = OJ[(f,j)][idx]
                prev_i = OJ[(f,j)][idx - 1]
                model.addConstr(
                    ST[j, i,f] >= s[j, prev_i,f] + quicksum(
                        x[j, prev_i,f, k] * operations_times[f,j, prev_i, k]
                        for k in operations_machines[f,j, prev_i]
                    ) - largeM * (1 - eta[j, i,f]),
                    name=f"ST_after_prev_op_{j}_{i}"
                )
    # Calculate worker transport times (similar to AGV transport time)
    # 为每个工人创建辅助变量来表示运输时间

    worker_transport_time = {}
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                worker_transport_time[j, i,f] = model.addVar(vtype=GRB.INTEGER, name=f"worker_tt_{j}_{i}")

                if i == OJ[(f,j)][0]:  # 首工序
                    # 创建辅助变量表示每个工人和各机器的组合
                    aux_tt = {}
                    for w in W[f-1]:
                        for k in operations_machines[f, j, i]:
                            aux_tt[w, k,f] = model.addVar(vtype=GRB.INTEGER, name=f"aux_tt_{j}_{i}_{w}_{k}")
                            # 线性化约束：当工人w负责运输且使用机器k时
                            model.addConstr(aux_tt[w, k,f] <= TT[0][k] * mu_w[j, i,f, w], f"aux_tt_bound1_{j}_{i}_{w}_{k}")
                            model.addConstr(aux_tt[w, k,f] <= largeM * x[j, i,f, k], f"aux_tt_bound2_{j}_{i}_{w}_{k}")
                            model.addConstr(aux_tt[w, k,f] >= TT[0][k] * mu_w[j, i,f, w] - largeM * (1 - x[j, i,f, k]),
                                            f"aux_tt_bound3_{j}_{i}_{w}_{k}")

                    # 将所有可能的工人和机器组合贡献加入总运输时间
                    model.addConstr(worker_transport_time[j, i,f] == quicksum(
                        aux_tt[w, k,f] for w in W[f-1] for k in operations_machines[f,j, i]
                    ), f"worker_tt_first_{j}_{i}")
                else:
                    prev_i = OJ[(f,j)][OJ[(f,j)].index(i) - 1]

                    # 处理非首工序 - 需要考虑前序工序机器到当前工序机器的运输
                    # 第一步：创建辅助变量表示机器分配的组合
                    y = {}
                    for k_prev in operations_machines[f,j, prev_i]:
                        for k_curr in operations_machines[f,j, i]:
                            y[k_prev, k_curr,f] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{prev_i}_{i}_{k_prev}_{k_curr}")
                            # 确保y只在两个机器都被分配时为1
                            model.addConstr(y[k_prev, k_curr,f] <= x[j, prev_i,f, k_prev],
                                            f"y_bound1_{j}_{i}_{k_prev}_{k_curr}")
                            model.addConstr(y[k_prev, k_curr,f] <= x[j, i, f,k_curr], f"y_bound2_{j}_{i}_{k_prev}_{k_curr}")
                            model.addConstr(y[k_prev, k_curr,f] >= x[j, prev_i,f, k_prev] + x[j, i,f, k_curr] - 1,
                                            f"y_bound3_{j}_{i}_{k_prev}_{k_curr}")

                    # 第二步：创建辅助变量表示工人、机器组合的运输时间
                    aux_tt = {}
                    for w in W[f-1]:
                        for k_prev in operations_machines[f,j, prev_i]:
                            for k_curr in operations_machines[f,j, i]:
                                aux_tt[w, k_prev, k_curr,f] = model.addVar(vtype=GRB.INTEGER,
                                                                         name=f"aux_tt_{j}_{i}_{w}_{k_prev}_{k_curr}")

                                # 线性化三元乘积：mu_w[j,i,w] * y[k_prev,k_curr] * TT[k_prev][k_curr]
                                model.addConstr(aux_tt[w, k_prev, k_curr,f] <= largeM * mu_w[j, i,f, w],
                                                f"aux_tt_bound1_{j}_{i}_{w}_{k_prev}_{k_curr}")
                                model.addConstr(aux_tt[w, k_prev, k_curr,f] <= largeM * y[k_prev, k_curr,f],
                                                f"aux_tt_bound2_{j}_{i}_{w}_{k_prev}_{k_curr}")
                                model.addConstr(aux_tt[w, k_prev, k_curr,f] <= TT[k_prev][k_curr],
                                                f"aux_tt_bound3_{j}_{i}_{w}_{k_prev}_{k_curr}")
                                model.addConstr(aux_tt[w, k_prev, k_curr,f] >= TT[k_prev][k_curr] -
                                                largeM * (2 - mu_w[j, i,f, w] - y[k_prev, k_curr,f]),
                                                f"aux_tt_bound4_{j}_{i}_{w}_{k_prev}_{k_curr}")

                    # 将所有可能的工人和机器组合贡献加入总运输时间
                    model.addConstr(worker_transport_time[j, i,f] == quicksum(
                        aux_tt[w, k_prev, k_curr,f]
                        for w in W[f-1]
                        for k_prev in operations_machines[f,j, prev_i]
                        for k_curr in operations_machines[f,j, i]
                    ), f"worker_tt_{j}_{i}")
    for f in F:
        for w in W[f-1]:
            for j1 in J:
                for i1 in OJ[(f,j1)]:
                    for j2 in J:
                        for i2 in OJ[(f,j2)]:
                            if (j1, i1) != (j2, i2):
                                # 当j1,i1是监控任务，j2,i2是运输任务，且j1,i1在j2,i2之前时，mon_trans=1
                                model.addConstr(
                                    mon_trans[j1, i1, j2, i2,f, w] <= gamma[j1, i1,f, w],
                                    f"mon_trans_1_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    mon_trans[j1, i1, j2, i2,f, w] <= mu_w[j2, i2,f, w],
                                    f"mon_trans_2_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    mon_trans[j1, i1, j2, i2,f, w] + mon_trans[j2, i2, j1, i1,f, w] >= gamma[j1, i1,f, w] + mu_w[j2, i2,f, w] - 1,
                                    f"mon_trans_4_{j1}_{i1}_{j2}_{i2}_{w}"
                                )

                                # 当j1,i1是运输任务，j2,i2是监控任务，且j1,i1在j2,i2之前时，trans_mon=1
                                model.addConstr(
                                    trans_mon[j1, i1, j2, i2,f, w] <= mu_w[j1, i1,f, w],
                                    f"trans_mon_1_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    trans_mon[j1, i1, j2, i2,f, w] <= gamma[j2, i2,f, w],
                                    f"trans_mon_2_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    trans_mon[j1, i1, j2, i2,f, w]+ trans_mon[j2, i2, j1, i1,f, w] >= mu_w[j1, i1,f, w] + gamma[j2, i2,f, w]  - 1,
                                    f"trans_mon_4_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    trans_mon[j1, i1, j2, i2,f, w] +trans_mon[j2, i2, j1, i1,f, w] <= 1,
                                )
                                model.addConstr(
                                    mon_trans[j1, i1, j2, i2,f, w] + mon_trans[j2, i2, j1, i1,f, w] <= 1,
                                )
    # 正确处理运输→监控的转移时间
    for f in F:
        for w in W[f-1]:
            for j1 in J:
                for i1 in OJ[(f,j1)]:  # 运输任务
                    for j2 in J:
                        for i2 in OJ[(f,j2)]:  # 监控任务
                            if (j1, i1) != (j2, i2):
                                # 确保只有当第一个任务真的是运输任务时才应用此约束
                                for k_dest in operations_machines[f,j1, i1]:  # 运输目的地机器
                                    for k_monitor in operations_machines[f,j2, i2]:  # 监控机器
                                        # 添加条件确保第一个任务是运输任务
                                        model.addConstr(
                                            (mu_w[j1, i1,f, w] == 1) >> (
                                                    MT[j2, i2,f] >= ST[j1, i1,f]  + TT[k_dest][
                                                k_monitor]
                                                    - largeM * (4 - gamma[j2, i2,f, w] - x[j1,i1,f,k_dest] - x[
                                                j2, i2,f, k_monitor] - trans_mon[j1, i1, j2, i2,f, w])
                                            ),
                                            f"worker_transport_to_monitor_{j1}_{i1}_{j2}_{i2}_{w}_{k_dest}_{k_monitor}"
                                        )

    # 计算工人从监控任务到运输任务的转移时间
    for f in F:
        for w in W[f-1]:
            for j1 in J:
                for i1 in OJ[(f,j1)]:  # 监控任务
                    for j2 in J:
                        for i2 in OJ[(f,j2)]:  # 运输任务
                            if (j1, i1) != (j2, i2):
                                # 确定监控机器和运输起点机器
                                for k_monitor in operations_machines[f, j1, i1]:  # 监控机器
                                    if i2 == OJ[(f,j2)][0]:  # 首工序运输，起点是仓库(0)
                                        model.addConstr(
                                            (gamma[j1, i1,f, w] == 1) >> (
                                                    ST[j2, i2,f] >= MT[j1, i1,f] + x[j1, i1, f,k_monitor] * operations_times[f,j1, i1, k_monitor]/2 + TT[k_monitor][0]
                                                    - largeM * (3 - mu_w[j2, i2,f, w] - x[j1, i1,f, k_monitor] - mon_trans[j1, i1, j2, i2,f, w])
                                            ),
                                            f"worker_monitor_to_transport_first_{j1}_{i1}_{j2}_{i2}_{w}_{k_monitor}"
                                        )
                                    else:  # 非首工序运输，起点是前序工序机器
                                        prev_i2 = OJ[(f,j2)][OJ[(f,j2)].index(i2) - 1]
                                        for k_prev in operations_machines[f,j2, prev_i2]:  # 前序工序机器
                                            model.addConstr(
                                                (gamma[j1, i1, f, w] == 1) >> (
                                                        ST[j2, i2,f] >= MT[j1, i1,f] + x[j1, i1, f,k_monitor] * operations_times[f,j1, i1, k_monitor]/2 + TT[k_monitor][k_prev]
                                                        - largeM * (4 - mu_w[j2, i2,f, w] - x[j1, i1,f, k_monitor]
                                                                    - x[j2, prev_i2,f, k_prev] - mon_trans[j1, i1, j2, i2,f, w])
                                                ),
                                                f"worker_monitor_to_transport_{j1}_{i1}_{j2}_{i2}_{w}_{k_monitor}_{k_prev}"
                                            )
    # 确保工人不能同时监控和运输
    for f in F:
        for w in W[f-1]:
            for j1 in J:
                for i1 in OJ[(f,j1)]:
                    for j2 in J:
                        for i2 in OJ[(f,j2)]:
                            if (j1, i1) != (j2, i2):
                                for k in operations_machines[f,j1, i1]:
                                    # 如果两个任务都分配给同一工人，一个为运输一个为监控，必须确保时序合理
                                    model.addConstr(
                                        MT[j2, i2,f] >= MT[j1, i1,f] + operations_times[f,j1,i1,k]/2 -
                                        largeM * (4 - gamma[j1, i1,f, w] - mu_w[j2, i2,f, w] - beta[j1, i1, j2, i2, f,w] - x[j1,i1,f,k]),
                                        f"worker_monitor_transport_seq1_{j1}_{i1}_{j2}_{i2}_{w}"
                                    )
                                    model.addConstr(
                                        ST[j2, i2,f] >= MT[j1, i1,f] + operations_times[f,j1,i1,k]/2  -
                                        largeM * (4 - gamma[j1, i1,f, w] - mu_w[j2, i2,f, w] - beta[j1, i1, j2, i2,f, w] - x[j1,i1,f,k]),
                                        f"worker_monitor_transport_seq2_{j1}_{i1}_{j2}_{i2}_{w}"
                                    )

    # 修改现有约束，将工人运输时间纳入考虑
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                model.addConstr(
                    s[j, i,f] >= ST[j, i,f] + Tt[j, i,f] * (1 - eta[j,i,f]),
                    f"transport_time_combined_{j}_{i}"
                )

    worker_z = {}
    for f in F:
        for w in W[f-1]:
            for j1 in J:
                for i1 in OJ[(f,j1)]:
                    for j2 in J:
                        for i2 in OJ[(f,j2)]:
                            if (j1, i1) != (j2, i2):
                                worker_z[j1, i1, j2, i2,f, w] = model.addVar(vtype=GRB.BINARY,
                                                                           name=f"worker_z_{j1}_{i1}_{j2}_{i2}_{w}")
    # 工人运输任务排序约束
    for f in F:
        for w in W[f-1]:
            for j1 in J:
                for i1 in OJ[(f,j1)]:
                    for j2 in J:
                        for i2 in OJ[(f,j2)]:
                            if (j1, i1) != (j2, i2):
                                # 当两个任务都分配给同一工人运输时，必须有一个顺序
                                model.addConstr(
                                    worker_z[j1, i1, j2, i2,f, w] + worker_z[j2, i2, j1, i1,f, w] <= 1,
                                    f"worker_transport_seq_mutex_{j1}_{i1}_{j2}_{i2}_{w}"
                                )

                                # 如果都分配给同一工人运输，必须有一个顺序
                                model.addConstr(
                                    worker_z[j1, i1, j2, i2,f, w] + worker_z[j2, i2, j1, i1,f, w] >=
                                    mu_w[j1, i1,f, w] + mu_w[j2, i2,f, w] - 1,
                                    f"worker_transport_seq_required_{j1}_{i1}_{j2}_{i2}_{w}"
                                )

                                # 运输时间顺序约束
                                model.addConstr(
                                    ST[j2, i2,f] >= ST[j1, i1,f]+ Tt[j1,i1,f] -
                                    largeM * (2 - mu_w[j1, i1,f, w] - mu_w[j2, i2,f, w]) -
                                    largeM * (1 - worker_z[j1, i1, j2, i2,f, w]),
                                    f"worker_transport_timing_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
    # Create worker transport sequencing variables (similar to z for AGVs)
    # 1. 监控→监控任务序列约束
    # for f in F:
    #     for w in W[f-1]:
    #         for j1 in J:
    #             for i1 in OJ[(f,j1)]:
    #                 for j2 in J:
    #                     for i2 in OJ[(f,j2)]:
    #                         if (j1, i1) != (j2, i2):
    #                             for k1 in operations_machines[f,j1, i1]:
    #                                 for k2 in operations_machines[f,j2, i2]:
    #                                     # 若工人w先执行j1,i1监控，再执行j2,i2监控
    #                                     model.addConstr(
    #                                         MT[j2, i2,f] >= MT[j1, i1,f] + x[j1, i1, f,k1] * operations_times[f,j1, i1, k1]/2 + TT[k1][k2]
    #                                         - largeM * (5 - beta[j1, i1, j2, i2,f, w]
    #                                                     - gamma[j1, i1,f, w] - gamma[j2, i2,f, w]
    #                                                     - x[j1, i1,f, k1] - x[j2, i2,f, k2]),
    #                                         f"worker_monitor_to_monitor_{j1}_{i1}_{j2}_{i2}_{w}_{k1}_{k2}"
    #                                     )

    # 2. 运输→运输任务序列约束
    for f in F:
        for w in W[f-1]:
            for j1 in J:
                for i1 in OJ[(f,j1)]:  # 第一个运输任务
                    for j2 in J:
                        for i2 in OJ[(f,j2)]:  # 第二个运输任务
                            if (j1, i1) != (j2, i2):
                                # 确定第一个运输任务的目的地机器
                                for k1_dest in operations_machines[f,j1, i1]:
                                    # 根据第二个运输任务是否为首工序，确定起始位置
                                    if i2 == OJ[(f,j2)][0]:  # 第二个是首工序，起点是仓库(0)
                                        model.addConstr(
                                            ST[j2, i2,f] >= ST[j1, i1,f] + Tt[j1,i1,f] + TT[k1_dest][0]
                                            - largeM * (3 - mu_w[j1, i1,f, w] - mu_w[j2, i2,f, w]
                                                        - worker_z[j1, i1, j2, i2,f, w]),
                                            f"worker_transport_to_transport_first_{j1}_{i1}_{j2}_{i2}_{w}_{k1_dest}"
                                        )
                                    else:  # 第二个非首工序，起点是前序工序机器
                                        prev_i2 = OJ[(f,j2)][OJ[(f,j2)].index(i2) - 1]
                                        for k2_prev in operations_machines[f,j2, prev_i2]:
                                            model.addConstr(
                                                ST[j2, i2,f] >= ST[j1, i1,f] + Tt[j1,i1,f] + TT[k1_dest][
                                                    k2_prev]
                                                - largeM * (4 - mu_w[j1, i1,f, w] - mu_w[j2, i2,f, w]
                                                            - worker_z[j1, i1, j2, i2,f, w] - x[j2, prev_i2,f, k2_prev]),
                                                f"worker_transport_to_transport_{j1}_{i1}_{j2}_{i2}_{w}_{k1_dest}_{k2_prev}"
                                            )
    # 6. 处理与运输时序耦合
    for f in F:
        for j in J:
            for idx, i in enumerate(OJ[(f, j)]):
                if idx == 0:  # 首工序
                    model.addConstr(
                        s[j, i, f] >= ST[j, i, f] + Tt[j, i, f] - largeM * (1 - fac_y[j, f]),
                        f"coupling_first_{j}_{i}"
                    )
                else:  # 非首工序
                    prev_i = OJ[(f, j)][idx - 1]
                    # 当η=1时，处理任务在运输后开始
                    model.addConstr(
                        s[j, i, f] >= ST[j, i, f] + Tt[j, i, f] - largeM * (1 - eta[j, i, f]) - largeM * (
                                    1 - fac_y[j, f]),
                        f"coupling_transport_{j}_{i}"
                    )
                    # η=0时：紧接前序工序结束
                    model.addConstr(
                        (eta[j, i, f] == 0) >> (
                                s[j, i, f] >= s[j, prev_i, f] + quicksum(
                            x[j, prev_i, f, k] * operations_times[f,j, prev_i,  k]
                            for k in operations_machines[f, j, prev_i]
                        ) - largeM * (1 - fac_y[j, f])
                        ),
                        f"coupling_no_transport_{j}_{i}"
                    )

    # 7. Makespan定义
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                model.addConstr(
                    cmax >= s[j, i,f] + quicksum(
                        x[j, i, f,k] * operations_times[f,j, i, k]
                        for k in operations_machines[f,j, i]
                    ),
                    f"cmax_{j}"
                )
    # model.Params.MIPFocus = 2  # 让求解器专注于尽可能多地改进当前的最优解
    # 调整求解器参数
    # model.Params.Presolve = 2  # 加强预处理
    # model.Params.Cuts = 3  # 激进生成切割
    # model.Params.Heuristics = 0.8  # 增加启发式搜索
    # model.Params.MIPGap = 0.05  # 接受5%的gap提前终止
    # model.Params.Threads = 4  # 使用多线程
    model.Params.TimeLimit = 20000
    model.update()
    return model
