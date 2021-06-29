import numpy as np
import matplotlib.pyplot as plt


def draw_result(lst_iter, lst_loss, lst_ppl, title):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 15))
    fig.suptitle(title)

    ax1.plot(lst_iter, lst_loss, '-b', label='loss')
    ax2.plot(lst_iter, lst_ppl, '-r', label='perplexity')

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc='upper right')
    ax1.set_yticks(np.arange(min(lst_loss), max(lst_loss), 0.5))

    ax2.set_xlabel("epoch")
    ax2.set_ylabel("perplexity")
    ax2.legend(loc='upper right')
    ax2.set_yticks(np.arange(min(lst_ppl), max(lst_ppl), 50))

    last_value = float("{:.2f}".format(lst_ppl[-1]))
    ax2.annotate(last_value, (lst_iter[-1], lst_ppl[-1]))

    # save image
    fig.savefig('plots/' + title + '.png')

    # show
    plt.show()
