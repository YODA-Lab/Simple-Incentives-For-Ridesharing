"""
!! ONLY USE WITH test_driver_fair.py !!

10/26/22
Written to run experiments of combinations of alpha, beta and delta, 
with 0.9999 driver discount, warm start of 100, 0.999 sr discount

plus driver and plusreq: plusdrive with some beta
plus driver and allreq: allreq with some delta
all driver and plusreq: plusreq with some delta
all driver and allreq: alldrive with some beta
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--start', type=int, default=8)
parser.add_argument('-e', '--end', type=int, default=12)
parser.add_argument('-c', '--cap', type=int, default=4)
parser.add_argument('-n', '--numvehs', type=int, default=1000)
parser.add_argument('-giff', '--giff', type=bool, default=False)
args = parser.parse_args()

c= args.cap
n=args.numvehs
if c ==4:
    if n==1000 or 1:
        model_name = "pathnn_1000n_4c_233362.0.h5"
elif c==10:
    if n==1000:
        model_name = "pathnn_1000n_10c_255518.0.h5"

model_loc = "../models/"+model_name

start = args.start
end = args.end

tag = 'Test/GIFF' if args.giff else 'Test'
logdir = f"../logs/{n}veh_{c}cap_ny{start}-{end}"
ran = []

vfs_dict = {
    'areq':'alphaRequests',
    'plusreq':'plusRequests',
    'adrive':'alphaDrivers',
    'plusdrive':'plusDrivers'
}

driver_fair = True
passenger_fair = False

# deltas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 0.0]
# driver fair
if driver_fair:
    alphads = [0.0, 1.0]
    deltas = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    betas = [0.0]
    alphas = [0.0]

    if args.giff:
        betas = [0.0]
        alphads = [0.0, 1.0] # when used with GIFF, becomes a trigger for GIFF (+))
        deltas = [0.0, 0.1, 0.2, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 1.0]
        # deltas = [0.995]
        alphas = [0.0] # This is the degree of advantage correction
        # alphas = [0.000001]


# passenger fair
if passenger_fair:
    alphads = [0.0, 1.0]
    deltas = [0.0]
    betas = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    alphas = [0.0, 1.0]

    if args.giff:
        deltas = [0.0]
        alphads = [0.0, 1.0] # when used with GIFF, becomes a trigger for GIFF (+))
        betas = [0.0, 0.1, 0.2, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 1.0]
        # betas = [0.9999, 0.99995, 0.99999, 0.999995, 0.999999, 0.9999995]
        # betas = [0.99995]
        # betas = [0.0, 1.0]
        alphas = [0.0, 0.1]
        # alphas = [0.000001]

test = False

for alphad in alphads:
    for delta in deltas:
        for alpha in alphas:
            for beta in betas:
                log_loc = f'{logdir}{tag}/alpha_d{alphad}/delta{delta}/alpha{alpha}/beta{beta}/'
                print(log_loc)
                print(os.path.isdir(log_loc))
                if not(os.path.isdir(log_loc)) or test:
                    ran.append([delta, alpha, beta])
                    print(f'cap{c}({start}-{end}),delta{delta}/alpha{alpha}/beta{beta}/ Running')
                    command = f"python test_driver_fair.py -m {model_loc} -v pathnn -c {c} -n {n} -s {start} -e {end} --tag {tag} -d {delta} -a {alpha} -b {beta} -ad {alphad}"
                    if args.giff:
                        command += " --giff True"
                    os.system(command)
