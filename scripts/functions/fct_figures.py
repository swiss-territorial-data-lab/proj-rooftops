import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_histo(dir_plots, df1, df2, attribute, xlabel):

    written_file = []
    fig = plt.figure(figsize =(12, 8))

    df1 = df1.fillna(0)
    df2 = df2.fillna(0)

    for i in attribute:
        bins = np.histogram(np.hstack((df1[i], df2[i])), bins=10)[1]
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


def plot_area(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(1, 2, sharey= True, figsize=(16,8))

    color_list = ['limegreen', 'tomato']  

    df = df[df['attribute'] == attribute]  

    df.plot(ax=ax[0], x='value', y=['free_area_labels', 'occup_area_labels',], kind='bar', stacked=True, rot=0, color = color_list)
    df.plot(ax=ax[1], x='value', y=['free_area_dets', 'occup_area_dets',], kind='bar', stacked=True, rot=0, color = color_list)
    for b, c in zip(ax[0].containers, ax[1].containers):
        labels1 = [f'{"{0:.1f}".format(a)}' if a > 0 else "" for a in b.datavalues]
        labels2 = [f'{"{0:.1f}".format(a)}' if a > 0 else "" for a in c.datavalues]
        ax[0].bar_label(b, label_type='center', color = "black", labels=labels1, fontsize=10)
        ax[1].bar_label(c, label_type='center', color = "black", labels=labels2, fontsize=10)

    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')  
    ax[0].set_xlabel(xlabel, fontweight='bold')
    ax[0].set_ylabel('area ($m^2$)', fontweight='bold')
    ax[1].set_xlabel(xlabel, fontweight='bold')

    ax[0].legend('', frameon=False)  
    ax[1].legend(['Free', 'Occupied'], bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)    
    ax[0].set_title(f'GT areas by {attribute.replace("_", " ")}')
    ax[1].set_title(f'Detection areas by {attribute.replace("_", " ")}')

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'area_{attribute}.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def plot_area_bin(dir_plots, df, bins):

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
 
    bins = list(set(bins).intersection(df.keys()))
    bins.sort()

    values = df[bins].iloc[0]

    df = pd.DataFrame({'bins':bins, 'val':values * 100})
    ax = df.plot.bar(x='bins', y='val', rot=0, color='limegreen')

    labels = list(map(lambda x: x.replace('accuracy bin ', ''), bins))
    ax.set_xticklabels(labels)

    plt.xlabel('Free area (%)', fontweight='bold')
    plt.ylabel('Accurate detection (%)', fontweight='bold')
    plt.legend('', frameon=False)
    plt.title('EGID')

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'area_accuracy_EGID.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def plot_groups(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(figsize=(12,8))

    df = df[df['attribute'] == attribute]  
    if df['FP'].isnull().values.any():
        color_list = ['#00D800', '#21D4DA']  
        counts_list = ['TP', 'FN']  
    else:
        color_list = ['#00D800', '#EF9015', '#21D4DA']  
        counts_list = ['TP', 'FP', 'FN']  

    df = df[df['attribute'] == attribute].copy()
    df = df[['value', 'TP', 'FP', 'FN']].set_index('value')

    if attribute == 'object_class':
        df[counts_list].plot.bar(ax=ax, color=color_list, rot=0, width=0.7, stacked=True)
        plt.xticks(rotation=40, ha='right')
    else:
        df[counts_list].plot.bar(ax=ax, color=color_list, rot=0, width=0.7)
    plt.xlabel(xlabel, fontweight='bold')

    for c in ax.containers:
        labels = [f'{int(a)}' if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color = "black", labels=labels, fontsize=10)

    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)    
    plt.title(f'Counts by {attribute.replace("_", " ")}')

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'counts_{attribute}.png')  
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
        color_list = ['#00D800', '#21D4DA']  
        counts_list = ['TP', 'FN']  
    else:
        color_list = ['#00D800', '#EF9015', '#21D4DA']  
        counts_list = ['TP', 'FP', 'FN']  

    df = df[['value', 'TP', 'FP', 'FN']].set_index('value')

    df[counts_list].plot(ax=ax, kind='bar', stacked=True, color=color_list, rot=0)

    for c in ax.containers:
        labels = [f'{"{0:.1f}".format(a)}' if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color="black", labels=labels, fontsize=10)

    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    plt.xlabel(xlabel, fontweight='bold')

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)    
    plt.title(f'Counts by {attribute.replace("_", " ")}')

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'counts_{attribute}_stacked.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    
    return plot_path


def plot_stacked_grouped_percent(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(figsize=(12,8))

    format_plot = {'gt': {'color_list': ['#00D800', '#21D4DA'], 'count_list': ['TP', 'FN'], 'position': -0.05, 'legend': ['TP', 'FN']},
                   'dt': {'color_list': ['#00D800', '#EF9015'], 'count_list': ['TP', 'FP'], 'position': 1.05, 'legend': ['', 'FP']}}

    df = df[df['attribute'] == attribute] 
    for variables in format_plot.values():

        df_subset = df[variables['count_list'] + ['value']].set_index('value')
        df_subset['sum'] = df_subset.sum(axis=1)

        for count in variables['count_list']:
            df_subset[count] = df_subset[count] / df_subset['sum']

        if attribute == 'object_class':
            df_subset[variables['count_list']].plot.bar(ax=ax, stacked=True, color=variables['color_list'], rot=0, width=0.3, align='center')
            break
        else:
            df_subset[variables['count_list']].plot.bar(ax=ax, stacked=True, color=variables['color_list'], rot=0, width=0.3, position=variables['position'], label=variables['legend'])

    for c in ax.containers:
        labels = [f'{"{0:.1%}".format(a)}' if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color = "black", labels=labels, fontsize=10)

    plt.ylim(0, 1)
    plt.gca().set_yticks(plt.gca().get_yticks().tolist())
    plt.gca().set_yticklabels([f'{"{0:.0%}".format(x)}' for x in plt.gca().get_yticks()]) 
    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    else:
        old_xlim = plt.xlim()
        plt.xlim(old_xlim[0], old_xlim[1] + 0.3)
    plt.xlabel(xlabel, fontweight='bold')

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)  
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))  
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


def plot_histo_object(dir_plots, df, attribute, datasets):

    fig, ax = plt.subplots(figsize=(16,10))
    
    df = df[['descr', 'dataset']]
    df['counts'] = 1
    ax = pd.pivot_table(data=df, index=['descr'], columns=['dataset'], values='counts', aggfunc='count').plot.bar(rot=0, log=True, stacked=True, color=['turquoise', 'gold'] , width=0.8)

    for c in ax.containers:
        labels = [int(a) if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color="black", labels=labels, fontsize=8)

    plt.gca().set_yticks(plt.gca().get_yticks().tolist())

    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    plt.xlabel('', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.ylim(1e0, 1e4)

    plt.legend(title='Dataset', loc='upper left', frameon=False)    

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'counts_{attribute}_GT.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def plot_stacked_grouped_object(dir_plots, df, param_ranges, param, attribute, label):

    fig, ax = plt.subplots(figsize=(16,10))
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
        ax.bar_label(c, label_type='center', color="black", labels=labels, fontsize=6)

    plt.gca().set_yticks(plt.gca().get_yticks().tolist())
    # plt.gca().set_yticklabels([f'{"{0:.0%}".format(x)}' for x in plt.gca().get_yticks()]) 
    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    plt.xlabel('', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    # plt.ylim(1e-1, 1e5)

    plt.legend(title=label.capitalize().replace("_", " "), loc='upper left', frameon=False, ncol=math.ceil(len(param_ranges)/2))    

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'{param}_class.png')   
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