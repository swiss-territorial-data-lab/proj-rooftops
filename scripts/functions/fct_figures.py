import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


def plot_histo(dir_plots, df1, df2, attribute, xlabel):

    written_file = []
    fig = plt.figure(figsize =(12, 8))

    df1 = df1.fillna(0)
    df2 = df2.fillna(0)

    for i in attribute:
        bins = np.histogram(np.hstack((df1[i],df2[i])), bins=10)[1]
        df1[i].plot.hist(bins=bins, alpha=0.5, label='GT')
        df2[i].plot.hist(bins=bins, alpha=0.5, label='Detections')

        plt.xlabel(xlabel[i] , fontweight='bold')

        plt.legend(frameon=False)  
        plt.title(f'Object distribution')

        plt.tight_layout() 
        plot_path = os.path.join(dir_plots, f'histo_{i}.png')  
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)

        written_file.append(plot_path)
    
    return written_file


def plot_surface(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(1, 2, sharey= True, figsize=(16,8))

    color_list = ['limegreen', 'tomato']  

    df = df[df['attribute'] == attribute]  

    df.plot(ax=ax[0], x='value', y=['free_surface_label', 'occupied_surface_label',], kind='bar', stacked=True, rot=0, color = color_list)
    df.plot(ax=ax[1], x='value', y=['free_surface_det', 'occupied_surface_det',], kind='bar', stacked=True, rot=0, color = color_list)
    for b, c in zip(ax[0].containers, ax[1].containers):
        labels1 = [f'{"{0:.1f}".format(a)}' if a > 0 else "" for a in b.datavalues]
        labels2 = [f'{"{0:.1f}".format(a)}' if a > 0 else "" for a in c.datavalues]
        ax[0].bar_label(b, label_type='center', color = "black", labels=labels1, fontsize=10)
        ax[1].bar_label(c, label_type='center', color = "black", labels=labels2, fontsize=10)

    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')  
    ax[0].set_xlabel(xlabel, fontweight='bold')
    ax[0].set_ylabel('Surface ($m^2$)', fontweight='bold')
    ax[1].set_xlabel(xlabel, fontweight='bold')

    ax[0].legend('', frameon=False)  
    ax[1].legend(['Free', 'Occupied'], bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)    
    ax[0].set_title(f'GT surfaces by {attribute.replace("_", " ")}')
    ax[1].set_title(f'Detection surfaces by {attribute.replace("_", " ")}')

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'surface_{attribute}.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def plot_surface_bin(dir_plots, df, bins, attribute):

    fig, ax = plt.subplots(1, 1, figsize=(12,8))

    df = df[df['attribute'] == attribute] 
    bins = list(set(bins).intersection(df.keys()))

    values = df[bins].iloc[0]

    df = pd.DataFrame({'bins':bins, 'val':values * 100})
    df.plot.bar(x='bins', y='val', rot=0, color='limegreen')

    plt.xlabel('Free surface area (%)', fontweight='bold')
    plt.ylabel('Accurate detection (%)', fontweight='bold')
    plt.legend('', frameon=False)
    plt.title(attribute)

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'surface_accuracy_{attribute}.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def plot_groups(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(figsize=(12,8))

    color_list = ['limegreen', 'orange', 'tomato']  
    counts_list = ['TP', 'FP', 'FN']    

    df = df[df['attribute'] == attribute].copy()
    df = df[['value', 'TP', 'FP', 'FN']].set_index('value')


    if attribute == 'object_class':
        df[counts_list].plot.bar(ax=ax, color=color_list, rot=0, stacked=True)
        plt.xticks(rotation=40, ha='right')
    else:
        df[counts_list].plot.bar(ax=ax, color=color_list, rot=0, width=0.7)
    plt.xlabel(xlabel, fontweight='bold')

    for c in ax.containers:
        labels = [f'{int(a)}' if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color = "white", labels=labels, fontsize=10)

    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)    
    plt.title(f'Counts by {attribute.replace("_", " ")}')

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'counts_{attribute}.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    
    return plot_path


def plot_stacked_grouped(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(figsize=(12,8))
 
    df = df[df['attribute'] == attribute]  
    if df['FP'].isnull().values.any():
        color_list = ['limegreen', 'tomato']  
        counts_list = ['TP', 'FN']  
    else:
        color_list = ['limegreen', 'orange', 'tomato']  
        counts_list = ['TP', 'FP', 'FN']  

    df = df[['value', 'TP', 'FP', 'FN']].set_index('value')

    df[counts_list].plot(ax=ax, kind='bar', stacked=True, color=color_list, rot=0)

    for c in ax.containers:
        labels = [f'{"{0:.1f}".format(a)}' if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color = "white", labels=labels, fontsize=10)

    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    plt.xlabel(xlabel, fontweight='bold')

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)    
    plt.title(f'Counts by {attribute.replace("_", " ")}')

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'counts_{attribute}.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    
    return plot_path


def plot_stacked_grouped_percent(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(figsize=(12,8))

    df = df[df['attribute'] == attribute]  
    # if df['FP'].isnull().values.any():
    if attribute != 'EGID' or attribute!= 'object_class':

        color_list = ['limegreen', 'tomato']  
        counts_list = ['TP', 'FN']  
        df = df[['value', 'TP', 'FN']].set_index('value')
    else:
        color_list = ['limegreen', 'orange', 'tomato']  
        counts_list = ['TP', 'FP', 'FN'] 
        df = df[['value', 'TP', 'FP', 'FN']].set_index('value')

    df['sum'] = df.sum(axis=1)

    for count in counts_list:
        df[count] =  df[count] / df['sum']
    
    df[counts_list].plot(ax=ax, kind='bar', stacked=True, color=color_list, rot=0, width = 0.5)

    for c in ax.containers:
        labels = [f'{"{0:.1%}".format(a)}' if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color = "white", labels=labels, fontsize=10)

    plt.ylim(0, 1)
    plt.gca().set_yticks(plt.gca().get_yticks().tolist())
    plt.gca().set_yticklabels([f'{"{0:.0%}".format(x)}' for x in plt.gca().get_yticks()]) 
    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    plt.xlabel(xlabel, fontweight='bold')

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)    
    plt.title(f'Counts by {attribute.replace("_", " ")}')

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'counts_{attribute}.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def plot_histo_object(dir_plots, df, attribute):

    fig, ax = plt.subplots(figsize=(12,8))

    df['descr'].value_counts(sort=False).plot.bar(rot=0, log=True, width=0.8)

    for c in ax.containers:
        labels = [f'{"{0:.1f}".format(a)}' if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color="white", labels=labels, fontsize=10)

    plt.gca().set_yticks(plt.gca().get_yticks().tolist())

    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    plt.xlabel('', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.ylim(1e0,1e4)

    plt.legend(title='Roundness', bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)    
    # plt.title(f'Counts by {attribute.replace("_", " ")}')

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'counts_{attribute}_GT.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def plot_stacked_grouped_object(dir_plots, df, param_ranges, attribute):

    fig, ax = plt.subplots(figsize=(12,8))
    df_toplot = pd.DataFrame() 

    for lim_inf, lim_sup in param_ranges:
        filter_df = df[(df['roundness'] >= lim_inf) & (df['roundness'] <= lim_sup)]
        filter_df['round_cat'] = f"{lim_inf} - {lim_sup}" 
        df_toplot = pd.concat([df_toplot, filter_df], ignore_index=True)

    df_toplot = df_toplot[['descr', 'round_cat']] 
    df_toplot.groupby(['descr', 'round_cat']).value_counts().unstack().plot(kind='bar', stacked=True, log='True')

    # for c in ax.containers:
    #     labels = [f'{"{0:.1%}".format(a)}' if a > 0 else "" for a in c.datavalues]
    #     ax.bar_label(c, label_type='center', color="white", labels=labels, fontsize=10)

    plt.gca().set_yticks(plt.gca().get_yticks().tolist())
    # plt.gca().set_yticklabels([f'{"{0:.0%}".format(x)}' for x in plt.gca().get_yticks()]) 
    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    plt.xlabel('', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.ylim(1e-1,1e4)

    plt.legend(title='Roundness', bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)    
    # plt.title(f'Counts by {attribute.replace("_", " ")}')

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'roundness_class.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def plot_metrics(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(figsize=(8,6))

    metrics_list = ['precision', 'recall', 'f1', 'IoU_mean', 'IoU_median']    

    df = df[df['attribute'] == attribute] 
    
    for metric in metrics_list:
        if not df[metric].isnull().values.any():
            plt.scatter(df['value'], df[metric], label=metric.replace("_", " "), s=150)

    plt.ylim(-0.05, 1.05)
    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    plt.xlabel(xlabel, fontweight='bold')

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)
    plt.title(f'Metrics by {attribute.replace("_", " ")}')
 
    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'metrics_{attribute}.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path