from ast import literal_eval

import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
import os
import pandas as pd

plt.style.use('report_stylesheet_standard.mplstyle')

class Plotting:
    def __init__(self, mapping_basis=('orthogonal', 'original', 'sines')):

        self.mapping_basis = mapping_basis
        self.acc_overlap_vs_chimax()
        self.acc_overlap_vs_nb_images()

    def acc_overlap_vs_chimax(self):
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax_overlap = ax.twinx()

        for basis in self.mapping_basis:
            df_path = os.path.join('results', basis, 'results.csv')
            try:
                df = pd.read_csv(df_path, sep=',')
            except FileNotFoundError:
                continue

            if basis == 'original':
                label = 'non-orthonormal'
            else:
                label = 'orthonormal'

            max_sweeps_df = df[df['nb_sweeps'] == 3]
            new_df = max_sweeps_df['truncation_overlap'].apply(literal_eval)
            overlap_gmean = new_df.apply(gmean)
            ax.plot(max_sweeps_df['chi_max'], max_sweeps_df['accuracy'], '-o', label=label)
            ax.set_xlabel('$\chi_{max}$', fontsize=12)
            ax.set_ylabel('$Accuracy$', fontsize=12)
            ax_overlap.plot(max_sweeps_df['chi_max'], overlap_gmean, '--o', label=label)
            ax_overlap.set_ylabel(r'Mean of overlaps $| \langle \Sigma_l^\chi | \Sigma_l \rangle |$', fontsize=12)
            ax.legend(fontsize=12, bbox_to_anchor=(.3, .68))
            ax.grid(True)
            ax.text(2, 0.93, '(b)')

            ax.set_axisbelow(True)
            fig.savefig('figures/accuracy_and_overlap_vs_chimax.pdf', bbox_inches='tight')

    def acc_overlap_vs_nb_images(self):
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax_overlap = ax.twinx()

        for basis in self.mapping_basis:
            df_path = os.path.join('results', basis, 'acc_vs_nb_path.csv')
            df = pd.read_csv(df_path, sep=',')
            df = df.sort_values('nb_images_exact')
            new_df = df['truncation_overlap'].apply(literal_eval)
            overlap_gmean = new_df.apply(gmean)
            ax.plot(df['nb_images_exact'], df['accuracy'], '-o', label=basis)
            ax.set_xlabel('# of images', fontsize=12)
            ax.set_ylabel('$accuracy$', fontsize=12)
            ax.grid(True)
            ax_overlap.plot(df['nb_images_exact'], overlap_gmean, 'o:', label=basis)
            ax_overlap.set_ylabel('truncation overlap', fontsize=12)
            ax.legend(fontsize=12)
            fig.savefig('figures/accuracy_and_overlap_vs_nb_images.pdf', bbox_inches='tight')


if __name__ == '__main__':
    Plotting(mapping_basis=['orthogonal', 'original'])
    plt.show()