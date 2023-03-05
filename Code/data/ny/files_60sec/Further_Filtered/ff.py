import random


def ff(fn):
    with open(fn, 'r') as f:
        a = ''.join(f.readlines())

    a = a.split('Flows:')
    header = a[0]
    new_tails = []
    for i in range(1, len(a)):
        epoch = a[i].strip().split('\n')
        head = epoch[0]
        tail = epoch[1:]
        random.shuffle(tail)
        tail = tail[:len(tail) // 10]
        ffed_epoch = '\n'.join([head] + tail + [''])
        new_tails.append(ffed_epoch)
    new_a = 'Flows:'.join([header] + new_tails)

    with open(fn, 'w') as f:
        f.write(new_a)


fns = [f'filtered_test_flow_5000_{i}.txt' for i in range(1, 21)]

for fn in fns:
    ff(fn)

