"""t test
"""


import argparse
import numpy as np
from scipy import stats
from statsmodels.stats import weightstats as stests


parser = argparse.ArgumentParser(description='t test script')
parser.add_argument('--paths', nargs='+', default=['genetic_algorithm/populations/Simple/Simple_Cellular-20-10_None_NStep/1000.pop.npy', 'genetic_algorithm/populations/Simple/Simple_Island-5-4-1-1-10_Blend_NStep/1000.pop.npy'], help='path/to/populations')
args = parser.parse_args()

players = ['random', 'smart', 'simple', 'advanced', 'full', 'fellow-student']
x = np.array([
    [0.80, 0.46, 0.516, 0.283, 0.269, 0.572]
])


def main():
    for i, data in enumerate(x):
        print(f'Player {players[i]} with data {data}')
        for j, d in enumerate(data):
            print(f'Checking against {players[j]} with data {d}')
            z_test, p_val = stests.ztest([], x2=None, value=0.5)



    print(float(p_val))

    if p_val < 0.05:
        print('Reject null hypothesis')
    else:
        print('accept null hypothesis')

if __name__ == '__main__':
    main()
