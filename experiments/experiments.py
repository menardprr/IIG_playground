import os
import importlib
from datetime import datetime
import time
import collections.abc
from copy import deepcopy

import yaml
import pickle

import random
import math
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from cycler import cycler

from tbparse import SummaryReader


import gc
import matplotlib.pyplot as plt
from cycler import cycler


def dict_deep_update(d, u):
    """deep dict update from 
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_class(class_name):
    '''Import class from its name
    '''
    module_names = class_name.split('.')
    module = ".".join(module_names[:-1])
    return getattr(importlib.import_module(module), module_names[-1])


class ExperimentGenerator(object):
    def __init__(
        self,
        description,
        game_names,
        agents,
        save_path,
        global_init_kwargs=None,
        seeds=[1, 2, 3],
    ):
        # Name of the exp
        self.description = description

        # Games
        self.game_names = game_names

        # List of seeds
        self.seeds = seeds

        # Global init kwargs
        self.global_init_kwargs = {}
        if global_init_kwargs:
            self.global_init_kwargs = global_init_kwargs

        # Path to save results
        self.save_path = os.path.join(save_path, description)
        self.log_path = os.path.join(self.save_path, "logs")

        # Build the agent constructors
        self.dict_agent_constructor = {}
        self.dict_agent_kwargs = {}
        self.agent_names = []
        for agent_config_path in agents:
            # Get agent config
            agent_config = yaml.load(
                open(agent_config_path, 'r'), Loader=yaml.FullLoader)
            # Get agent class
            agent_class_name = agent_config['agent_class']
            agent_class = get_class(agent_class_name)
            # Set agent parameter
            agent_kwargs = deepcopy(self.global_init_kwargs)
            dict_deep_update(agent_kwargs, agent_config['init_kwargs'])
            agent_name = agent_kwargs['config_kwargs']['name']
            # record name, kwargs, constructor
            self.agent_names.append(agent_name)
            self.dict_agent_kwargs[agent_name] = agent_kwargs
            self.dict_agent_constructor[agent_name] = agent_class

    def run(self):
        # Prepare tasks
        list_tasks = []
        agent_idx = 0
        for game_name in self.game_names:
            for agent_name in self.agent_names:
                for seed in self.seeds:
                    list_tasks.append([
                        self.dict_agent_constructor[agent_name],
                        self.dict_agent_kwargs[agent_name],
                        game_name,
                        seed,
                        agent_idx
                    ])
                    agent_idx += 1
        # Fit agents
        for task in list_tasks:
            self.fit_agent(*task)
        print()
        print('Finished!')

    def fit_agent(
        self,
        agent_contstructor,
        agent_kwargs,
        game_name,
        seed,
        agent_idx
    ):
        # Set game and seed
        agent_kwargs["game_name"] = game_name
        agent_kwargs["seed"] = seed
        # Set writer path
        agent_name = agent_kwargs['config_kwargs']['name']
        now = datetime.now().strftime("%d-%m__%H:%M")
        agent_kwargs["writer_path"] = os.path.join(
            self.log_path,
            game_name,
            agent_name,
            str(agent_idx),
            now
        )
        # Build agent
        agent = agent_contstructor(**agent_kwargs)
        # Run agent
        print(f'Train {agent_name} on {game_name} with seed {seed}')
        try:
            agent.fit()
        except KeyboardInterrupt:
            gc.collect()
            pass

    def load_results(self):
        """Load and return the last results"""
        dict_results = {}
        agent_idx = 0
        for game_name in self.game_names:
            dict_results[game_name] = {}
            for agent_name in self.agent_names:
                dict_results[game_name][agent_name] = {}
                for seed in self.seeds:
                    # Get writer folder
                    writer_path = os.path.join(
                        self.log_path,
                        game_name,
                        agent_name,
                        str(agent_idx)
                    )
                    list_res = os.listdir(writer_path)
                    # Get last logs
                    last_res = max(list_res, key=lambda x: time.mktime(time.strptime(
                        x,
                        '%d-%m__%H:%M'
                    )))
                    res_path = os.path.join(writer_path, last_res)
                    # Read logs
                    print(f"Load {agent_name} on {game_name} with seed {seed}")
                    reader = SummaryReader(res_path)
                    df = reader.scalars
                    # Load res
                    dict_results[game_name][agent_name][agent_idx] = df
                    agent_idx += 1
        return dict_results

    def plot_ax(self, ax, df, y_name, y_label=''):
        sns.lineplot(data=df, x='step', y=y_name,
                     hue='algorithm', style='algorithm', ax=ax)
        ax.grid()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(y_label)

    def make_df(self, y_name, y_key, game_dict):
        dfs = []
        for agent_name, agent_dict in game_dict.items():
            for df in agent_dict.values():
                df_masked = df[df["tag"] == y_key]  # "nash_{name}/exp"]
                dfs.append(pd.DataFrame({
                    "step": df_masked["step"].to_numpy(),
                    y_name: df_masked["value"].to_numpy(),
                    "algorithm":  [agent_name]*len(df_masked["step"])
                }))
        return pd.concat(dfs)

    def plot_game(self, axs, plot_rw, plot_col, y_name, y_label, title, game_dict):
        df = self.make_df(y_name, f"nash_{y_name}/exp", game_dict)
        self.plot_ax(axs[plot_rw, plot_col], df, y_name, y_label)
        axs[plot_rw, plot_col].set_title(title)

    def plot_results(self, show=False):
        # Load results:
        results = self.load_results()

        # figure size in inches
        plt.rcParams['figure.figsize'] = 20, 15
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['font.size'] = 14
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['axes.prop_cycle'] = (cycler('color', ['r', 'g', 'b', 'y']) +
                                           cycler('linestyle', ['-', '--', ':', '-.']))

        # Plot cur and prox exploitatbility
        fig, axs = plt.subplots(nrows=2, ncols=len(
            self.game_names), squeeze=False)

        y_label = 'Exploitability'
        i = 0
        for game_name, game_dict in results.items():
            self.plot_game(
                axs,
                0,
                i,
                "cur",
                y_label,
                f'Current exp. in {game_name}'.replace('_', ' ').replace(
                    'kuhn', 'Kuhn').replace('leduc', 'Leduc'),
                game_dict
            )
            self.plot_game(
                axs,
                1,
                i,
                "prox",
                y_label,
                f'Proximal exp. in {game_name}'.replace('_', ' ').replace(
                    'kuhn', 'Kuhn').replace('leduc', 'Leduc'),
                game_dict
            )
            i += 1
            y_label = ''

        # Save fig
        save_path = os.path.join(self.save_path, self.description+'_plot.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f'Saved plot at {save_path}')
        if show:
            fig.show()
