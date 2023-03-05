
"""
For comparing the performance of different models. 
Runs the test day in train_fair.py for each model, with the given beta values.
"""
import os
import argparse

os.chdir('../')

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--start', type=int, default=0)
parser.add_argument('-e', '--end', type=int, default=24)
parser.add_argument('-c', '--cap', type=int, default=4)
parser.add_argument('-n', '--numvehs', type=int, default=1000)
args = parser.parse_args()

c= args.cap
n=args.numvehs
start = args.start
end = args.end
model_loc = "../models/RFTrain/hourly_models/"

tag = 'Test/ModelComparison6Hours'
logdir = f"../logs/{n}veh_{c}cap_ny{start}-{end}"
ran = []

betas = [0.0, 2.0] #list of betas
# models = [
#     'nyenvt_pathnn0_beta0.0_1000veh_4cap_0-24_209619.0_GPUTrainGRebalance2.h5',
#     'nyenvt_pathnn1_beta0.0_1000veh_4cap_0-24_212994.0_GPUTrainGRebalance2.h5',
#     'nyenvt_pathnn2_beta0.0_1000veh_4cap_0-24_213760.0_GPUTrainGRebalance2.h5',
#     'nyenvt_pathnn3_beta0.0_1000veh_4cap_0-24_214822.0_GPUTrainGRebalance2.h5',
#     'nyenvt_pathnn4_beta0.0_1000veh_4cap_0-24_221171.0_GPUTrainGRebalance2.h5',
#     'nyenvt_pathnn5_beta0.0_1000veh_4cap_0-24_222959.0_GPUTrainGRebalance2.h5',
#     'nyenvt_pathnn6_beta0.0_1000veh_4cap_0-24_225857.0_GPUTrainGRebalance2.h5',
#     'nyenvt_pathnn7_beta0.0_1000veh_4cap_0-24_227838.0_GPUTrainGRebalance2.h5',
# ]
models = os.listdir(model_loc)
models.sort()

for beta in betas:
    for i, model in enumerate(models):
        model_tag = tag+f'/Model{i}'
        log_loc = f'{logdir+model_tag}/alpha0.0/beta{beta}/plusRequests/'
        print(log_loc)
        print(os.path.isdir(log_loc))
        if not(os.path.isdir(log_loc)):
            ran.append([model, beta])
            print(f'cap{c}({start}-{end}),a0.0,b{beta} plusRequests Running')
            os.system(f'python train_fair.py -n {n} -c {c} -v plusreq -m {model_loc+model} -s {start} -e {end} -b {beta} --tag {model_tag}')

#Save index-to-model mapping
model_dict = {i:m for i,m in enumerate(models)}
with open(logdir+tag+'/model_mapping.txt', 'w') as outfile:
    outfile.write(str(model_dict))
