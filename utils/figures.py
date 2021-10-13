import os
from datetime import datetime as dt
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from hydrograph import hydrograph


def plot_hydrograph_years(csv, fig_dir, hline=400):
    df = hydrograph(csv)
    name = df.columns[0]
    df['date'] = df.index
    df['year'] = df.date.dt.year
    df['date'] = df.date.dt.strftime('%m-%d')
    df.index = [x for x in range(0, df.shape[0])]
    ydf = df.set_index(['year', 'date'])[name].unstack(-2)
    ydf.dropna(axis=1, how='all', inplace=True)
    ldf = ydf.copy()
    ldf.index = [x for x in range(0, ldf.shape[0])]
    mins = [(c, ldf[c][152: 304].idxmin(), ldf[c][152: 304].min()) for c in ldf.columns if ldf[c][152: 304].min() < 400.]
    ydf['hline'] = [400 for _ in ydf.index]
    colors = ['k' if _ != 'hline' else 'r' for _ in ydf.columns]
    ax = ydf.plot(logy=True, legend=False, alpha=0.2, color=colors, ylabel='discharge [cfs]',
             title='Clark Fork at Turah: {}\n'
                   '1985 - 2020'.format(name.split(':')[1]))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ann = [('{}-{} {:.0f} cfs'.format(ydf.index[m[1]], m[0], m[2])) for m in mins]
    txt = '\n'.join(ann)
    props = dict(boxstyle='round', facecolor='white', alpha=1)

    ax.text(1, 1, txt, transform=ax.transAxes, fontsize=10,
            horizontalalignment='right', verticalalignment='top', bbox=props)

    ax.annotate('400 cfs', xy=(100, 410), xycoords='data', color='r')
    plt.savefig(os.path.join(fig_dir, 'stacked_{}'.format(name.split(':')[1])))



if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/Montana/water_rights/hydrographs'
    h = os.path.join(root, '12334550.csv')
    plot_hydrograph_years(h, root, hline=400)
# ========================= EOF ====================================================================
