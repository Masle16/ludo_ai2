import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='python script to plot random search results')
parser.add_argument('--path', type=str, default='genetic_algorithm/tuning/Simple.txt', help='path/to/file/with/results')
parser.add_argument('--tight', action='store_const', const=True, default=False)
parser.add_argument('--legends', action='store_const', const=True, default=False)
args = parser.parse_args()


chosen = [0, 2, 4]


def load_data(path):
    bars, scores = [], []
    f = open(path, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        line = line.split()
        bars.append(line[1])
        scores.append(float(line[3]))
    f.close()
    return bars, scores

def main():
    bars, scores = load_data(args.path)

    if args.legends:
        plt.xticks(rotation='vertical')
        plt.bar(bars, scores)
    else:
        plt.bar([i for i in range(len(scores))], scores)
    
    plt.ylim([0, 1])

    if args.tight:
        plt.tight_layout(pad=0.25)
    
    plt.xlabel('EA combinations')
    plt.ylabel('Win rate [%]')

    plt.show()


if __name__ == '__main__':
    main()