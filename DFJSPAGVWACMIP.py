from gurobipy import Model, GRB, quicksum
import sys
from map import get_travel_time

def MIPModel(Data):
    n = Data['n']
    m = Data['m']
    J = Data['J']
    A = Data['A']
    OJ = Data['OJ']
    Worker = Data['W']
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
    Y = {}
    LP= {}     # λ[j1,i1,j2,i2,r]
    eta = {}    # η[j,i]
    Z = {}     # μ[j,i,r]
    z = {}      # LP[j1,i1,j2,i2,r]
    SO = {}      # SO[j,i]
    ST = {}     # ST[j,i]
    Tt = {}     # Tt[j,i]
    YP = {}
    cmax = model.addVar(vtype=GRB.INTEGER, name="cmax")
    V = {}  # Worker assignment to operations
    VP = {}  # Worker sequence between operations
    SR = {}  # Monitoring start time
    theta = {}  # Whether operation is the first task for a worker
    W = {}  # Whether transport task is assigned to worker
    # Initialize worker-related variables
    # 新决策变量定义
    MT = {}  # 监控任务早于运输任务
    TM = {}  # 运输任务早于监控任务
    # Define a new variable to assign jobs to factories
    X = {}  # y[j,f] = 1 if job j is assigned to factory f, 0 otherwise
    for j in J:
        for f in F:  # F is the set of factories
            X[j, f] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{f}")

    # Each job Zst be assigned to exactly one factory

    #监控与运输顺序变量
    for f in F:
        for w in Worker[f-1]:
            for j1 in J:
                for i1 in OJ[(f,j1)]:
                    for j2 in J:
                        for i2 in OJ[(f,j2)]:
                            if (j1, i1) != (j2, i2):
                                MT[j1, i1, j2, i2, f, w] = model.addVar(vtype=GRB.BINARY,
                                                                            name=f"MT_{j1}_{i1}_{j2}_{i2}_{w}")
                                TM[j1, i1, j2, i2, f, w] = model.addVar(vtype=GRB.BINARY,
                                                                            name=f"TM_{j1}_{i1}_{j2}_{i2}_{w}")
    #工人任务变量
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                SR[j, i, f] = model.addVar(vtype=GRB.INTEGER, name=f"SR_{j}_{i}_{f}") # Monitoring start time
                for w in Worker[f-1]:
                    V[j, i,f, w] = model.addVar(vtype=GRB.BINARY, name=f"V_{j}_{i}_{f}_{w}") # Worker assignment
                    # theta[j, i,f, w] = model.addVar(vtype=GRB.BINARY, name=f"theta_{j}_{i}_{w}") # First task for worker

    #初始化加工与运输操作
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                SO[j, i,f] = model.addVar(vtype=GRB.INTEGER, name=f"s_{j}_{i}_{f}")
                ST[j, i,f] = model.addVar(vtype=GRB.INTEGER, name=f"ST_{j}_{i}_{f}")
                Tt[j, i, f] = model.addVar(vtype=GRB.INTEGER, name=f"Tt_{j}_{i}_{f}")
                eta[j, i, f] = model.addVar(vtype=GRB.BINARY, name=f"eta_{j}_{i}_{f}")
                for k in operations_machines[f,j, i]:
                    Y[j, i, f, k] = model.addVar(vtype=GRB.BINARY, name=f"Y_{j}_{i}_{f}_{k}")
                for r in A[f-1]:
                    Z[j, i,f,r] = model.addVar(vtype=GRB.BINARY, name=f"Z_{j}_{i}_{f}_{r}")
                for w in Worker[f-1]:
                    W[j,i,f,w] = model.addVar(vtype=GRB.BINARY, name=f"W_{j}_{i}_{f}_{w}")

    # Define VP variable
    for f in F:
        for j1 in J:
            for j2 in J:
                for i1 in OJ[(f, j1)]:
                    for i2 in OJ[(f, j2)]:
                        for w in Worker[f-1]:
                            VP[j1, i1, j2, i2, f, w] = model.addVar(vtype=GRB.BINARY, name=f"VP_{j1}_{i1}_{j2}_{i2}_{w}")
    for f in F:
        for r in A[f-1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f,j1)]:
                        for i2 in OJ[(f,j2)]:
                            LP[j1, i1, j2, i2, f, r] = model.addVar(vtype=GRB.BINARY, name=f"z_{j1}_{i1}_{j2}_{i2}_{r}")
    for f in F:
        for k in M[f-1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f,j1)]:
                        for i2 in OJ[(f,j2)]:
                            YP[j1, i1, j2, i2,f, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{j1}_{i1}_{j2}_{i2}_{f}_{k}")
    # Objective
    model.setObjective(cmax, GRB.MINIMIZE)

    #辅助变量L，用于计算LP（同一机器上的先后加工关系）
    L = {}
    for f in F:
        for k in M[f - 1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f, j1)]:
                        for i2 in OJ[(f, j2)]:
                            if (j1, i1) != (j2, i2):
                                L[j1, i1, j2, i2, f, k] = model.addVar(
                                    vtype=GRB.BINARY,
                                    name=f"L_{j1}_{i1}_{j2}_{i2}_{k}"
                                )
    for f in F:
        for k in M[f - 1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f, j1)]:
                        for i2 in OJ[(f, j2)]:
                            if (j1, i1) != (j2, i2) and k in operations_machines[f,j1, i1] and k in operations_machines[f,j2, i2]:
                                # w <= LP[j1,i1,r]
                                model.addConstr(
                                    L[j1, i1, j2, i2, f, k] <= Y[j1, i1, f, k],
                                    name=f"w_upper1_{j1}_{i1}_{j2}_{i2}_{k}"
                                )
                                # w <= Z[j2,i2,r]
                                model.addConstr(
                                    L[j1, i1, j2, i2, f, k] <= Y[j2, i2, f, k],
                                    name=f"w_upper2_{j1}_{i1}_{j2}_{i2}_{k}"
                                )
                                # w >= LP[j1,i1,r] + Z[j2,i2,r] - 1
                                model.addConstr(
                                    L[j1, i1, j2, i2, f, k] >= Y[j1, i1, f, k] + Y[j2, i2, f, k] - 1,
                                    name=f"w_lower_{j1}_{i1}_{j2}_{i2}_{k}"
                                )
    for f in F:
        for k in M[f - 1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f, j1)]:
                        for i2 in OJ[(f, j2)]:
                            if (j1, i1) != (j2, i2):
                                # Ensure YP is 1 only when L[j1, i1, j2, i2, f, r] == 1
                                model.addConstr(
                                    (L[j1, i1, j2, i2, f, k] == 1) >> (
                                            YP[j1, i1, j2, i2, f, k] + YP[j2, i2, j1, i1, f, k] == 1
                                    ),
                                    name=f"YP_active_{j1}_{i1}_{j2}_{i2}_{f}_{k}"
                                )

    #VP约束，
    for f in F:
        for w in Worker[f-1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f,j1)]:
                        for i2 in OJ[(f,j2)]:
                            if (j1, i1) != (j2, i2):  # Avoid duplicate constraints
                                model.addConstr(
                                    VP[j1, i1, j2, i2, f, w] + VP[j2, i2, j1, i1, f, w] <= X[j1,f],
                                    f"worker_seq_Ztex_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    VP[j1, i1, j2, i2, f, w] + VP[j2, i2, j1, i1, f, w] <= X[j2, f],
                                    f"worker_seq_Ztex_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    VP[j1, i1, j2, i2, f, w] + VP[j2, i2, j1, i1, f, w] >= X[j1,f] + X[j2,f] + V[j1, i1, f, w] + V[j2, i2, f, w]-3,
                                    f"worker_seq_Ztex_{j1}_{i1}_{j2}_{i2}_{w}"
                                )


    # Constraints

    for f in F:
        for j in J:
            for idx, i in enumerate(OJ[(f,j)]):
                if idx == 0:
                    model.addConstr(Tt[j, i,f] == quicksum(Y[j, i,f, k] * TT[0][k] for k in operations_machines[f,j, i]))
                else:
                    prev_i = OJ[(f,j)][idx - 1]
                    y = {}
                    for k_prev in operations_machines[f,j, prev_i]:
                        for k_curr in operations_machines[f,j, i]:
                            y[j, i, f,k_prev, k_curr] = model.addVar(vtype=GRB.BINARY)
                            model.addConstr(y[j, i, f,k_prev, k_curr] <= Y[j, prev_i, f,k_prev])
                            model.addConstr(y[j, i, f,k_prev, k_curr] <= Y[j, i, f,k_curr])
                            model.addConstr(y[j, i, f,k_prev, k_curr] >=Y[j, prev_i,f, k_prev] + Y[j, i, f,k_curr] - 1)
                    model.addConstr(
                        Tt[j, i,f] == quicksum(
                            y[j, i,f, k_prev, k_curr] * TT[k_prev][k_curr]
                            # (x[j,i,k_curr]*x[j,prev_i,k_prev]) * TT[k_prev][k_curr]
                            for k_prev in operations_machines[f,j, prev_i]
                            for k_curr in operations_machines[f,j, i]
                        ),
                        name=f"Tt_{j}_{i}"
                    )

    for f in F:
        for w in Worker[f-1]:
            for j1 in J:
                for i1 in OJ[(f,j1)]:
                    for j2 in J:
                        for i2 in OJ[(f,j2)]:
                            if (j1, i1) != (j2, i2):
                                # 当j1,i1是监控任务，j2,i2是运输任务，且j1,i1在j2,i2之前时，MT=1
                                model.addConstr(
                                    MT[j1, i1, j2, i2,f, w] <= V[j1, i1,f, w],
                                    f"MT_1_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    MT[j1, i1, j2, i2,f, w] <= W[j2, i2,f, w],
                                    f"MT_2_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    MT[j1, i1, j2, i2,f, w] + MT[j2, i2, j1, i1,f, w] >= V[j1, i1,f, w] + W[j2, i2,f, w] - 1,
                                    f"MT_4_{j1}_{i1}_{j2}_{i2}_{w}"
                                )

                                # 当j1,i1是运输任务，j2,i2是监控任务，且j1,i1在j2,i2之前时，TM=1
                                model.addConstr(
                                    TM[j1, i1, j2, i2,f, w] <= W[j1, i1,f, w],
                                    f"TM_1_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    TM[j1, i1, j2, i2,f, w] <= V[j2, i2,f, w],
                                    f"TM_2_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    TM[j1, i1, j2, i2,f, w]+ TM[j2, i2, j1, i1,f, w] >= W[j1, i1,f, w] + V[j2, i2,f, w]  - 1,
                                    f"TM_4_{j1}_{i1}_{j2}_{i2}_{w}"
                                )
                                model.addConstr(
                                    TM[j1, i1, j2, i2,f, w] +TM[j2, i2, j1, i1,f, w] <= 1,
                                )
                                model.addConstr(
                                    MT[j1, i1, j2, i2,f, w] + MT[j2, i2, j1, i1,f, w] <= 1,
                                )

    #约束1
    for j in J:
        model.addConstr(
            quicksum(X[j, f] for f in F) == 1,
            f"job_factory_assignment_{j}"
        )
    #约束2
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                model.addConstr(
                    quicksum(V[j, i, f, w] for w in Worker[f-1]) == X[j,f],
                    f"worker_assignment_{j}_{i}"
                )

    # 约束3 机器分配 (Sub-problem 1)
    for f in F:
        for j in J:
            for i in OJ[(f, j)]:
                model.addConstr(
                    quicksum(Y[j, i, f, k] for k in operations_machines[f, j, i]) == X[j,f],
                    f"machine_assignment_{j}_{i}_{f}"
                )
    #约束4，5
    # # 2. 运输任务存在性 (η_j)
    for f in F:
        for j in J:
            # 首工序必须存在运输任务
            first_op = OJ[(f,j)][0]
            model.addConstr(eta[ j, first_op,f] == X[j,f], f"eta_first_{j}")
            # 非首工序
            for idx in range(1, len(OJ[(f,j)])):
                i = OJ[(f,j)][idx]
                prev_i = OJ[(f,j)][idx-1]
                model.addConstr(
                    eta[j, i,f] == X[j,f] - quicksum(
                        YP[j, prev_i, j, i,f, k]
                        for k in operations_machines[f,j, prev_i]
                        if k in operations_machines[f,j, i]
                    # eta[j,i,f] == X[j,f] - quicksum(YP[j,prev_i,j,i,f,k] for k in M[f-1]
                    ),
                    f"eta_{j}_{i}"
                )
    #约束6
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                model.addConstr(
                    quicksum(Z[j, i,f, r] for r in A[f-1])+quicksum(W[j, i,f, w] for w in Worker[f-1]) == eta[j, i,f],
                    f"AGV_assignment_{j}_{i}"
                )
    # 约束7
    for f in F:
        for k in M[f - 1]:
            ops_on_k = [
                (j, i)
                for j in J
                for i in OJ.get((f, j), [])
                if k in operations_machines.get((f, j, i), [])
            ]
            for idx1 in range(len(ops_on_k)):
                for idx2 in range(idx1 + 1, len(ops_on_k)):
                    j1, i1 = ops_on_k[idx1]
                    j2, i2 = ops_on_k[idx2]
                    # 原有约束：j1在j2之前
                    model.addConstr(
                        SO[j2, i2, f] >= SO[j1, i1, f] + operations_times[f, j1, i1, k] - largeM * (
                                    1 - YP[j1, i1, j2, i2, f, k]),
                        name=f"machine_order_forward_{j1}_{i1}_{j2}_{i2}_{k}"
                    )
                    # 新增约束：j2在j1之前
                    model.addConstr(
                        SO[j1, i1, f] >= SO[j2, i2, f] + operations_times[f, j2, i2, k] - largeM * (
                                    1 - YP[j2, i2, j1, i1, f, k]),
                        name=f"machine_order_backward_{j1}_{i1}_{j2}_{i2}_{k}"
                    )
    #约束8
    #运输时间在加工完成之后
    for f in F:
        for j in J:
            for idx in range(1, len(OJ[(f,j)])):
                i = OJ[(f,j)][idx]
                prev_i = OJ[(f,j)][idx - 1]
                for k in operations_machines[f,j, prev_i]:
                    model.addConstr(
                        ST[j, i,f] >= SO[j, prev_i,f] + operations_times[f,j, prev_i, k] - largeM * (1 - Y[j,prev_i,f,k]),
                        name=f"ST_after_prev_op_{j}_{i}"
                    )

    # 约束9 ，10  AGV任务排序约束（修正后的运输时间计算）\
        # 辅助变量w
    # 定义辅助变量 当且仅当 LP[j1,i1,r] == 1 且 Z[j2,i2,r] == 1 时，w = 1：
    w = {}
    for f in F:
        for r in A[f - 1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f, j1)]:
                        for i2 in OJ[(f, j2)]:
                            if (j1, i1) != (j2, i2):
                                w[j1, i1, j2, i2, f, r] = model.addVar(
                                    vtype=GRB.BINARY,
                                    name=f"w_{j1}_{i1}_{j2}_{i2}_{r}"
                                )
    for f in F:
        for r in A[f - 1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f, j1)]:
                        for i2 in OJ[(f, j2)]:
                            if (j1, i1) != (j2, i2):
                                # w <= LP[j1,i1,r]
                                model.addConstr(
                                    w[j1, i1, j2, i2, f, r] <= Z[j1, i1, f, r],
                                    name=f"w_upper1_{j1}_{i1}_{j2}_{i2}_{r}"
                                )
                                # w <= Z[j2,i2,r]
                                model.addConstr(
                                    w[j1, i1, j2, i2, f, r] <= Z[j2, i2, f, r],
                                    name=f"w_upper2_{j1}_{i1}_{j2}_{i2}_{r}"
                                )
                                # w >= LP[j1,i1,r] + Z[j2,i2,r] - 1
                                model.addConstr(
                                    w[j1, i1, j2, i2, f, r] >= Z[j1, i1, f, r] + Z[j2, i2, f, r] - 1,
                                    name=f"w_lower_{j1}_{i1}_{j2}_{i2}_{r}"
                                )
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
                                                    ST[j2, i2,f] >= ST[j1, i1,f] + Tt[j1, i1,f] + TT[k1][0]
                                                    - largeM * (3 - LP[j1, i1, j2, i2,f, r] - Y[j1, i1,f, k1] - Y[j2, i2,f, k2_curr]
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
                                                        ST[j2, i2,f] >= ST[j1, i1,f] + Tt[j1, i1,f] + TT[k1][k_prev]
                                                        - largeM * (4 - LP[j1, i1, j2, i2,f, r] - Y[j1, i1,f, k1] -Y[j2, prev_i2,f, k_prev] - Y[j2, i2,f, k_curr]
                                                ),
                                                f"AGV_order_{j1}_{i1}_{j2}_{i2}_{r}"
                                            )
                            if (j1, i1) != (j2, i2):
                                model.addConstr(
                                    (w[j1, i1, j2, i2, f, r] == 1) >> (
                                            LP[j1, i1, j2, i2, f, r] + LP[j2, i2, j1, i1, f, r] == 1
                                    ),
                                    name=f"AGV_order_Ztex_{j1}_{i1}_{j2}_{i2}_{r}"
                                )


    # 约束11 处理与运输时序耦合
    for f in F:
        for j in J:
            for idx, i in enumerate(OJ[(f, j)]):
                    model.addConstr(
                        SO[j, i, f] >= ST[j, i, f] + Tt[j, i, f]  - largeM * (1 - X[j, f]),
                        f"coupling_transport_{j}_{i}"
                    )
    #约束12，13
    for f in F:
        for j in J:
            for i in OJ[(f, j)]:
                # Ensure worker monitoring starts exactly when the operation starts
                model.addConstr(
                    SR[j, i, f] == SO[j, i, f],
                    f"monitoring_starts_with_operation_{j}_{i}"
                )

    # 约束14
    for f in F:
        for w in Worker[f - 1]:
            for j1 in J:
                for j2 in J:
                    for i1 in OJ[(f, j1)]:
                        for i2 in OJ[(f, j2)]:
                            if (j1, i1) != (j2, i2):
                                for k1 in operations_machines[f, j1, i1]:
                                    for k2 in operations_machines[f, j2, i2]:
                                        model.addConstr(
                                            SR[j2, i2, f] >= (SR[j1, i1, f] + operations_times[f, j1, i1, k1] / 2 +
                                                              TT[k1][k2]
                                                              - largeM * (3 - VP[j1, i1, j2, i2, f, w]
                                                                          - V[j1, i1, f, w] - V[j2, i2, f, w]
                                                                          )),
                                            f"worker_move_{j1}_{i1}_{j2}_{i2}_{w}_{k1}_{k2}"
                                        )
    # 约束15
    # 运输完成之后不会在0区域，因此不用考虑第一道工序的问题
    for f in F:
        for w in Worker[f-1]:
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
                                            (W[j1, i1,f, w] == 1) >> (
                                                    SR[j2, i2,f] >= ST[j1, i1,f]  + TT[k_dest][k_monitor]
                                                    - largeM * (4 - V[j2, i2,f, w] - Y[j1,i1,f,k_dest] - Y[j2, i2,f, k_monitor] - TM[j1, i1, j2, i2,f, w])
                                            ),
                                            f"worker_transport_to_monitor_{j1}_{i1}_{j2}_{i2}_{w}_{k_dest}_{k_monitor}"
                                        )
    # 约束16，17
    for f in F:
        for w in Worker[f-1]:
            for j1 in J:
                for i1 in OJ[(f,j1)]:  # 监控任务
                    for j2 in J:
                        for i2 in OJ[(f,j2)]:  # 运输任务
                            if (j1, i1) != (j2, i2):
                                # 确定监控机器和运输起点机器
                                for k_monitor in operations_machines[f, j1, i1]:  # 监控机器
                                    if i2 == OJ[(f,j2)][0]:  # 首工序运输，起点是仓库(0)
                                        model.addConstr(
                                            (V[j1, i1,f, w] == 1) >> (
                                                    ST[j2, i2,f] >= SR[j1, i1,f] + operations_times[f,j1, i1, k_monitor]/2 + TT[k_monitor][0]
                                                    - largeM * (2 - Y[j1, i1,f, k_monitor] - MT[j1, i1, j2, i2,f, w])
                                            ),
                                            f"worker_monitor_to_transport_first_{j1}_{i1}_{j2}_{i2}_{w}_{k_monitor}"
                                        )
                                    else:  # 非首工序运输，起点是前序工序机器
                                        prev_i2 = OJ[(f,j2)][OJ[(f,j2)].index(i2) - 1]
                                        for k_prev in operations_machines[f,j2, prev_i2]:  # 前序工序机器
                                            model.addConstr(
                                                (V[j1, i1, f, w] == 1) >> (
                                                        ST[j2, i2,f] >= SR[j1, i1,f] + operations_times[f,j1, i1, k_monitor]/2 + TT[k_monitor][k_prev]
                                                        - largeM * (3 - Y[j1, i1,f, k_monitor]
                                                                    - Y[j2, prev_i2,f, k_prev] - MT[j1, i1, j2, i2,f, w])
                                                ),
                                                f"worker_monitor_to_transport_{j1}_{i1}_{j2}_{i2}_{w}_{k_monitor}_{k_prev}"
                                            )

    # 7. Makespan定义
    for f in F:
        for j in J:
            for i in OJ[(f,j)]:
                model.addConstr(
                    cmax >= SO[j, i,f] + quicksum(
                        Y[j, i, f,k] * operations_times[f,j, i, k]
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
