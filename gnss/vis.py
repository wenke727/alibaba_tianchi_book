import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_satellite_distribution_seaborn(df):
    df = df.copy()
    palette = sns.color_palette("hsv", len(df['satTye'].unique()))
    color_map = {sat_type: palette[i]
                 for i, sat_type in enumerate(df['satTye'].unique())}

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    df.loc[:, 'x'] = np.radians(df['azimuth'])
    df.loc[:, 'y'] = 90 - df['elevation']
    # df.loc[:, 'size'] = np.power(df.satDb, 1.5)
    
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='satTye',
        palette=color_map,
        size='satDb',
        sizes=(0, 300),
        # size_norm=True,
        alpha=0.8,
        ax=ax,
        legend=True,
        marker='.')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], title='Sat Type', loc='upper right')

    ax.set_ylim(0, 90)
    ax.set_yticks(range(0, 91, 30))
    ax.set_yticklabels([str(90 - x) for x in range(0, 91, 30)])
    ax.set_ylabel("")

    ax.set_xticks(np.radians(range(0, 360, 90)))
    ax.set_xticklabels(['N', 'E',  'S', 'W'])
    ax.set_xticks(np.radians(range(0, 360, 30)))
    ax.set_xlabel("Azimuth")

    ax.set_title("Satellite Distribution")
    ax.grid(True)
    plt.tight_layout()

    return ax


