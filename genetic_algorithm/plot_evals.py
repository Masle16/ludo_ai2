import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():
    players = [
        'Simple_Island-5-4-1-1-10_Whole_NStep-0.82-0.52',
        'Simple_Cellular-20-10_None_Normal-0.71',
        'Simple_Normal-20-10_None_None'
    ]
    random = [55.4, 56.44, 57.64]
    fast = [66.8, 63.96, 64.56]
    aggressive = [54.44, 56.4, 56.88]
    smart = [21.96, 20.96, 19.24]
    data = [
        [55.4, 66.8, 54.44, 21.96],
        [56.44, 63.96, 56.4, 20.96],
        [57.64, 64.56, 56.88, 19.24]
    ]
    opps = ['Random', 'Fast', 'Aggressive', 'Smart']

    x = np.arange(len(players))
    width = 0.2

    fig, ax = plt.subplots()
    rects0 = ax.bar(x - (width+width/2), random, width, label=opps[0])
    rects1 = ax.bar(x - width/2, fast, width, label=opps[1])
    rects2 = ax.bar(x + width/2, aggressive, width, label=opps[2])
    rects3 = ax.bar(x + (width+width/2), smart, width, label=opps[3])

    ax.set_ylabel('Win rate [%]')
    ax.set_xlabel('EA combination')
    ax.set_xticks(x)
    ax.set_xticklabels(players)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects0)
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # fig.tight_layout()
    # plt.xticks(rotation='vertical')

    plt.show()

if __name__ == '__main__':
    main()
