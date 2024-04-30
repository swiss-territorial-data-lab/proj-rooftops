import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


def plot_histo(dir_plots, df1, df2, attribute, xlabel):

    written_file = []
    
    df1 = df1.fillna(0)
    df2 = df2.fillna(0)

    for i in attribute:
        fig = plt.figure(figsize =(9,6), layout='constrained')
        bins = np.histogram(np.hstack((df1[i], df2[i])), bins=10)[1]
        df1[i].plot.hist(bins=bins, alpha=0.5, label='GT')
        df2[i].plot.hist(bins=bins, alpha=0.5, label='Detections')

        plt.xlabel(xlabel[i] , fontweight='bold', fontsize=14)
        plt.ylabel('Frequency' , fontweight='bold', fontsize=14)

        plt.legend(frameon=False, fontsize=14)  
        plt.title(f'Object distribution')

        plot_path = os.path.join(dir_plots, f'histogram_{i}.png')  
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        plt.clf()

        written_file.append(plot_path)
    
    return written_file


def plot_area(dir_plots, df, attribute, xlabel, broken_axis=False):


    color_list = ['limegreen', 'tomato']  
    width = 0.4

    df = df[df['attribute'] == attribute] 

    if not broken_axis:
        fig, ax = plt.subplots(1, 1, figsize=(9,6))

        df.plot(ax=ax, x='value', y=['free_area_labels', 'occup_area_labels',], kind='bar', stacked=True, rot=0, width=width, position=1.05, color=color_list)
        df.plot(ax=ax, x='value', y=['free_area_dets', 'occup_area_dets',], kind='bar', stacked=True, rot=0, width=width, position=-0.05, color=color_list)
        
        for b in ax.containers:
            labels = [f'{int(a)}' if a > 0 else "" for a in b.datavalues]
            ax.bar_label(b, label_type='center', color = "black", labels=labels, fontsize=8)
    
        ax.set_xlim((-0.5, len(df)-0.5)) 
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=14)
        ax.set_ylabel('Area ($m^2$)', fontweight='bold', fontsize=14)   

        locs = ax.get_xticks(minor=False) 
        major_ticks_labels = ax.get_xticklabels(minor=False) 
        minor_ticks = []  
        minor_ticks_labels = [] 
        for i in locs:
            minor_ticks.extend([i - width/2, i + width/2])
            minor_ticks_labels.extend(['labels', 'detections'])
        ax.set_xticks(ticks=minor_ticks, labels=minor_ticks_labels, minor=True)

        if 'All evaluated EGIDs' in df['value'].unique():
            ax.tick_params(axis='x', which='major', bottom=False, labelbottom=False)
        else:
            ax.set_xticks(ticks=locs, labels=major_ticks_labels, minor=False, fontsize=14)
            ax.tick_params(axis='x', which='major', pad=30, length=0)

        ax.ticklabel_format(style='sci', axis='y', useOffset=True, scilimits=(0,0))

        ax.legend(['Free surface', 'Occupied surface'], bbox_to_anchor=(1.02, 1.0), loc='upper left', frameon=False, fontsize=14)
        plot_path = os.path.join(dir_plots, f'area_{attribute}.png')  

    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9,6))

        ax1.spines['bottom'].set_visible(False)
        ax1.tick_params(axis='x', which='both',bottom=False)
        ax2.spines['top'].set_visible(False)
        bs = 1.0e4
        ts = 4.0e4
        ax2.set_ylim(0, bs)
        ax1.set_ylim(ts, 1.5e5)

        df.plot(ax=ax1, x='value', y=['free_area_labels', 'occup_area_labels'], kind='bar', stacked=True, rot=0, width=width, position=1.05, color=color_list)
        df.plot(ax=ax1, x='value', y=['free_area_dets', 'occup_area_dets'], kind='bar', stacked=True, rot=0, width=width, position=-0.05, color=color_list)
        df.plot(ax=ax2, x='value', y=['free_area_labels', 'occup_area_labels'], kind='bar', stacked=True, rot=0, width=width, position=1.05, color=color_list)
        df.plot(ax=ax2, x='value', y=['free_area_dets', 'occup_area_dets'], kind='bar', stacked=True, rot=0, width=width, position=-0.05, color=color_list)

        n = 0
        for (b,c) in zip(ax1.containers, ax2.containers):
            labels = [f'{int(a)}' if a > 0 else '' for a in c.datavalues]
            ax2.bar_label(c, label_type='center', color = 'black', labels=labels, fontsize=8)
            n += 1
            labels = [f'{int(a)}' if a > 0 else '' for a in b.datavalues]
            if n == 1 or n == 3:
                labels= [x if x < str(bs) else '' for x in labels]
            ax1.bar_label(b, label_type='center', color = 'black', labels=labels, fontsize=8)

        locs = ax2.get_xticks(minor=False) 
        major_ticks_labels = ax2.get_xticklabels(minor=False) 
        minor_ticks = []  
        minor_ticks_labels = [] 
        for i in locs:
            minor_ticks.extend([i - width/2, i + width/2])        
            minor_ticks_labels.extend(['labels', 'detections'])  
        ax2.set_xticks(ticks=minor_ticks, labels=minor_ticks_labels, minor=True)

        ax1.ticklabel_format(style='sci', axis='y', useOffset=True, scilimits=(0,0))
        ax2.ticklabel_format(style='sci', axis='y', useOffset=True, scilimits=(0,0))

        ax1.legend(['Free surface', 'Occupied surface'], bbox_to_anchor=(1.02, 1.0), loc='upper left', frameon=False, fontsize=14)    
        ax2.legend().remove()

        for tick in ax2.get_xticklabels():
            tick.set_rotation(0)
        d = .01  
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)      
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax2.transAxes)  
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax1.set_xlim((-0.5, len(df)-0.5)) 
        ax2.set_xlabel(xlabel, fontweight='bold', fontsize=14)
        ax1.set_ylabel('Area ($m^2$)', fontweight='bold', fontsize=14)   
        ax1.yaxis.set_label_coords(-0.05, -.1)

        if 'All evaluated EGIDs' in df['value'].unique():
            ax2.tick_params(axis='x', which='major', bottom=False, labelbottom=False)
        else:
            ax2.set_xticks(ticks=locs, labels=major_ticks_labels, minor=False, fontsize=14)
            ax2.tick_params(axis='x', which='major', pad=30, length=0)

        plot_path = os.path.join(dir_plots, f'area_{attribute}_broken.png')  

    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def plot_groups(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(figsize=(9,6), layout='constrained')

    df = df[df['attribute'] == attribute] 
    if df['FP'].isnull().values.any():
        color_list = ['#00D800', '#21D4DA']  
        counts_list = ['TP', 'FN']  
    else:
        color_list = ['#00D800', '#EF9015', '#21D4DA']  
        counts_list = ['TP', 'FP', 'FN']  

    df = df[df['attribute'] == attribute].copy()
    df = df[['value', 'TP', 'FP', 'FN']].set_index('value')

    df[counts_list].plot.bar(ax=ax, color=color_list, rot=0, width=0.8)
    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    elif attribute == 'all':
        plt.xticks([])
    plt.xlabel(xlabel, fontweight='bold', fontsize=14)

    for c in ax.containers:
        labels = [f'{int(a)}' if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color = "black", labels=labels, fontsize=8)

    plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', frameon=False, fontsize=14)     

    plot_path = os.path.join(dir_plots, f'counts_{attribute}.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def plot_histo_object(dir_plots, df, attribute, datasets):

    fig, ax = plt.subplots(figsize=(16,10), layout='constrained')
    
    df = df[['descr', 'dataset']]

    df['counts'] = 1
    df = pd.pivot_table(data=df, index=['descr'], columns=['dataset'], values='counts', aggfunc='count')
    df = df[['training', 'test']] 
    ax = df.plot.bar(rot=0, log=True, stacked=True, color=['turquoise', 'gold'] , width=0.8)

    for c in ax.containers:
        labels = [int(a) if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color="black", labels=labels, fontsize=8)

    plt.gca().set_yticks(plt.gca().get_yticks().tolist())

    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    plt.xlabel('', fontweight='bold')
    plt.ylim(1e0, 1e4)

    plt.legend(title='Dataset', loc='upper left', frameon=False)    

    plot_path = os.path.join(dir_plots, f'counts_{attribute}_GT.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def plot_stacked_grouped_object(dir_plots, df, param_ranges, param, attribute, label):

    fig, ax = plt.subplots(figsize=(16,10), layout='constrained')
    df_toplot = pd.DataFrame() 
    param_ranges = param_ranges[param] 
    label = label[param] 

    for lim_inf, lim_sup in param_ranges:
        filter_df = df[(df[param] >= lim_inf) & (df[param] <= lim_sup)]
        filter_df['category'] = f"{lim_inf} - {lim_sup}" 
        df_toplot = pd.concat([df_toplot, filter_df], ignore_index=True)

    df_toplot = df_toplot[['descr', 'category']] 
    ax = df_toplot.groupby(['descr', 'category']).value_counts().unstack().plot(kind='bar', stacked=True, log=True, width=0.8)

    for c in ax.containers:
        labels = [int(a) if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color="black", labels=labels, fontsize=8)

    plt.gca().set_yticks(plt.gca().get_yticks().tolist()) 
    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    plt.xlabel('')
    # plt.ylim(1e-1, 1e5)

    plt.legend(title=label.capitalize().replace("_", " "), loc='upper left', frameon=False, ncol=math.ceil(len(param_ranges)/2))    

    plot_path = os.path.join(dir_plots, f'{param}_class.png')   
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def plot_metrics(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(figsize=(8,6), layout='constrained')

    metrics_list = ['precision', 'recall', 'f1', 'IoU_mean', 'IoU_median']    

    df = df[df['attribute'] == attribute] 
    
    for metric in metrics_list:
        if not df[metric].isnull().values.any():
            plt.scatter(df['value'], df[metric], label=metric.replace("_", " "), s=150, zorder=2)

    plt.ylim(0, 1.05)
    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    elif attribute == 'all':
        plt.xticks([])
    plt.xlabel(xlabel, fontweight='bold', fontsize=14)
    plt.grid(color='lightgray', linewidth = 0.5, zorder=1)

    plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', frameon=False, fontsize=14)
 
    plot_path = os.path.join(dir_plots, f'metrics_{attribute}.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path