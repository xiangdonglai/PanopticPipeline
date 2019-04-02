#!/usr/bin/env python
import inspect
import os
import random
import sys
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show, save
from bokeh.palettes import Spectral11
from bokeh.models import LinearAxis, Range1d

def get_log_parsing_script():
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe())))
    return dirname + '/parse_log.py'


def get_data_file(chart_type, path_to_log):
    return (os.path.basename(path_to_log) + '.' +
            'train'.lower())

def get_data_label(path_to_log):
    label = path_to_log[path_to_log.rfind('/')+1 : path_to_log.rfind(
        get_log_file_suffix())]
    return label

def plot_chart(path_to_png, path_to_log_list):
    # print(path_to_png[0:-4]+".html")
    output_file(path_to_png[0:-4]+".html")
    TOOLS = 'pan,box_zoom,wheel_zoom,save,hover,reset,resize'
    p = figure(title="TrainLoss(smoothed) vs NumIters",
    x_axis_label='NumIters', tools=TOOLS, width=1000, height=800)
    fig, ax1 = plt.subplots()


    for path_to_log in path_to_log_list:
        os.system('%s %s ./ ' % (get_log_parsing_script(), path_to_log))
        data_file = get_data_file(6, path_to_log)
        pddata = pd.read_csv(data_file)
        # print >> sys.stderr, pddata.head()
        standard = ['NumIters', 'Seconds', 'LearningRate']
        labels = [label for label in pddata if label not in standard]
        sp = 500 if len(pddata['NumIters'])>2000 else 10
        skip = 5 if len(pddata['NumIters'])>2000 else 0
        ylims = [100,-100]
        for label,i in zip(labels, range(len(labels))):
            Y = pddata[label].ewm(span=sp,adjust=True).mean()
            plt.plot(pddata["NumIters"], Y, alpha=0.4)
            ylims[0] = min(ylims[0],Y[skip:].min())
            ylims[1] = max(ylims[1],Y[skip:].max())
            p.line(pddata["NumIters"], Y, legend=label,
                    color=Spectral11[i%11], line_width=2)
        print(int(pddata["NumIters"].iloc[-1]))
        plt.legend()
        ax1.set_ylim(ylims)
        p.extra_y_ranges = {"LearningRate": Range1d(start=0, end=pddata['LearningRate'].max()*1.1)}
        p.add_layout(LinearAxis(y_range_name="LearningRate"), 'left')
        p.line(pddata["NumIters"],
               pddata['LearningRate'],
               legend='LearningRate',
               color='navy', y_range_name="LearningRate")
        # p.add_layout(LinearAxis(y_range_name="LearningRate"), 'left')
    save(p)
        # ax2 = ax1.twinx()
        # ax2.plot(pddata["NumIters"], pddata['LearningRate'], alpha=0.4)
    plt.title('TrainLoss vs NumIters')
    # ax2.set_ylabel('Learning Rate', color='r')
    plt.xlabel('NumIters')
    plt.savefig(path_to_png)

    # plt.show()

def print_help():
    print """This script mainly serves as the basis of your customizations.
Customization is a must.
You can copy, paste, edit them in whatever way you want.
Be warned that the fields in the training log may change in the future.
You had better check the data files and change the mapping from field name to
 field index in create_field_index before designing your own plots.
Usage:
    ./plot_training_log.py /where/to/save.png /path/to/first.log ...
Notes:
    1. Supporting multiple logs."""
    sys.exit()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print_help()
    else:
        path_to_png = sys.argv[1]
        if not path_to_png.endswith('.png'):
            print 'Path must ends with png' % path_to_png
            sys.exit()
        path_to_logs = sys.argv[2:]
        for path_to_log in path_to_logs:
            if not os.path.exists(path_to_log):
                print 'Path does not exist: %s' % path_to_log
                sys.exit()
        plot_chart(path_to_png, path_to_logs)
