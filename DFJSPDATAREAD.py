def convert_data(lines):
    result = {}
    operations_machines={}#the available machines for each operation of jobs
    operations_times={}#the processing time for each operation of jobs
    OJ={} # the number of operations for each job
    i = 0
    infor = lines[i].strip().split()
    job_num,factory_num,mch_num = int(infor[0]),int(infor[1]),int(infor[2])
    numonjobs = [[] for i in range(factory_num)]
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        parts = lines[i].strip().split()
        if len(parts) == 3 and i!= 0:
            fac_id, job_id, op_num = map(int, parts)
            i += 1
            numonjobs[fac_id-1].append(op_num)
            # Process jobs for this factory
            job_count = 0
            while i < len(lines) and len(lines[i].strip().split()) > 3:  # Job description lines
                job_parts = lines[i].strip().split()
                op_index = int(job_parts[0])
                mch_count = int(job_parts[1])
                job_machines = []

                job_processingtime = []
                # Process operations for this job
                op_data = job_parts[2:]
                for mch_index in range(1, mch_count + 1):
                    machine_index = int(op_data[(mch_index - 1) * 2])
                    proc_time = int(op_data[(mch_index - 1) * 2 + 1])
                    job_machines.append(machine_index)
                    job_processingtime.append(proc_time)
                    # Add to result dictionary
                    key = (fac_id, job_id, op_index, machine_index)
                    result[key] = proc_time
                operations_machines[(fac_id,job_id, op_index)] = job_machines
                for l in range(len(job_machines)):
                    operations_times[(fac_id,job_id, op_index,job_machines[l])] = job_processingtime[l]
                i += 1

                # Break if we've processed all jobs for this factory
                if op_index >= op_num:  # Assuming 5 jobs per factory based on data pattern
                    break
        else:
            i += 1

    F = list(range(1, factory_num + 1))  # define the index of factory
    J = list(range(1, job_num + 1))  # define the index of jobs
    M = [list(range(1, mch_num+1)) for _ in range(factory_num)]
    A = [list(range(1, 2)) for _ in range(factory_num)]
    W = [list(range(1, 3)) for _ in range(factory_num)]
    for fac in range(factory_num):
        for j in range(job_num):
            OJ[(fac+1,J[j])]=list(range(1,numonjobs[fac][j]+1))
    largeM=0
    for fac in F:
        for job in J:
            for op in OJ[(fac,job)]:
                protimemax=0
                for l in operations_machines[(fac,job,op)]:
                    if protimemax<operations_times[(fac,job,op,l)]:
                        protimemax=operations_times[(fac,job,op,l)]
                largeM+=protimemax*10
    Data = {
        'n': job_num,
        'm': mch_num,
        'J': J,
        'M': M,
        'A': A,
        'OJ': OJ,
        'W': W,
        'F': F,
        'operations_machines': operations_machines,
        'operations_times': operations_times,
        'largeM': largeM,
    }
    return Data


# Reading from a file
def read_and_convert_file(filename):
    with open(filename, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()

        # Convert the data
        result_dict = convert_data(lines)
        return result_dict


