import os
from collections import defaultdict

import torch


def parse_proxy(fname, scale):
    f = open(fname, 'r')
    layer_dict = {}
    for line in f:
        if 'proxy error' in line:
            line = line.rstrip()
            line = line[line.find('layer'):]
            proxy_error = float(line[line.find(':') + 1:])
            layer_name = ' '.join(line.split(' ')[1:3])
            layer_dict[layer_name] = {scale: proxy_error}
    return layer_dict


total = None
files = ['075', '080', '085', '090', '095', '100', '103', '105']
for key in files:
    res = parse_proxy(f'/work/albert/two_bit_quant/slurm_out/e8p_s{key}.log',
                      key)
    if total is None:
        total = res
    else:
        for key in res:
            total[key].update(res[key])

hist = defaultdict(int)
best_layer = {}
for layer in total:
    best = float('inf')
    best_scale = None
    for scale in total[layer]:
        if total[layer][scale] < best:
            best = total[layer][scale]
            best_scale = scale
    best_layer[layer] = best_scale
    hist[best_scale] += 1

print(hist)
exit()

ckpt_path = '/work/albert/two_bit_quant/checkpoints'
out_path = os.path.join(ckpt_path, 'e8p_best_scale')
os.system(f'rm -rf {out_path}')
os.system(f'mkdir {out_path}')

os.system('cp {} {}'.format(
    os.path.join(ckpt_path, f'e8p_s{files[0]}', 'config.pt'), out_path))

for layer in best_layer:
    src = os.path.join(ckpt_path, f'e8p_s{best_layer[layer]}',
                       '{}.pt'.format(layer.replace(' ', '_')))
    tgt = out_path
    os.system(f'cp {src} {tgt}')
