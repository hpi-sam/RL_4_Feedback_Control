import os
import random
import pandas as pd
from envs.environments import check_environment
import envs.data_utils as du
from data.transition_matrix.transition import Transition


class DataHandler:

    def __init__(self, data_generation: str = 'Linear', take_component_id: bool = True, transformation: str = 'raw', distinguishable: bool = False):
        '''
        Initializes the Data Handler by loading data into the environment and select between using the componenent id or name.

        :param data_generation:
            Choose between the version of the mRubis environment 'Linear' (default) | 'Saturating' | 'Combined' | 'Discontinuous'
            or shifted data 'LinearShifted'
            or non_stationary_data choose model like 'ARCH'
        :param take_component_id:
            Choose compononent id or name. When take_component_id is false, you will take name.
        :param transformation:
            Choose between 'raw' (Default), 'sqt', 'cube', 'log10', 'ln', 'log2'
        :param distinguishable:
            Choose only distinguishable <component,failure> pairs by setting this true.

        For all possible combinations of environment, id and type please have a look on the file 'environments.py'.
        '''

        self.environment, self.filename = check_environment(data_generation, take_component_id, transformation, distinguishable)
        self.transition = Transition()
        self.data: pd.DataFrame = pd.DataFrame()
        self.component_failure_pairs = []
        self.__load_data()
        self.__create_component_failure_pairs()

    def __load_transform_data(self) -> None:
        frames = []

        # searching for all csv files in the data directory and loading the data in multiple dataframes
        for root, dirs, files in os.walk(r'../data/original_data'):
            for f in files:
                if f.endswith(self.environment[0] + '.csv'):
                    file_path = os.path.join(root, f)
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.replace('\t', '')
                    frames.append(df)

        # combining all df to one df
        self.data = pd.concat(frames, sort=False, ignore_index=True)[
            ['Optimal_Affected_Component', 'Optimal_Affected_Component_Uid', 'Optimal_Failure', 'Optimal_Utility_Increase']].rename(columns={'Optimal_Utility_Increase': 'raw'})

        # removing empty rows
        nan_value = float("NaN")
        self.data.replace("", nan_value, inplace=True)
        self.data.dropna(subset=["Optimal_Failure"], inplace=True)

        # transform data
        self.data = du.transform_data(self.data)

        # save transformed data this to environment as a csv file for quick reload in future
        self.data.to_csv(self.filename)

    def __load_data(self):
        try:
            self.data = pd.read_csv(self.filename, index_col=0)[[self.environment[1], 'Optimal_Failure', self.environment[2]]]
        except FileNotFoundError:
            self.__load_transform_data()
            self.data = self.data[[self.environment[1], 'Optimal_Failure', self.environment[2]]]

    def __create_component_failure_pairs(self) -> None:
        combinations = self.data.groupby([self.data.columns[0], self.data.columns[1]]).size().reset_index().rename(
            columns={0: 'count'})
        del combinations['count']
        self.component_failure_pairs = [tuple(val) for val in combinations.values]

    def get_all_component_failure_pairs(self):
        return self.component_failure_pairs

    def get_sample_component_failure_pairs(self, sample_size: int, seed=None):
        if sample_size > len(self.component_failure_pairs):
            print('Error: Sample size exceeds number of (component, failure) pairs.')
        else:
            return list(set(pd.Series(self.get_all_component_failure_pairs()).sample(sample_size, random_state=seed).tolist()))

    def get_reward(self, component_failure_pair, list_of_failings) -> float:
        '''
        :param component_failure_pair: A tuple of component and failure to repair.
        :param list_of_failings: A list of tuples of component and failures which are still failing.
        :return: The reward for the component_failure_pair. A reward of 0 states that the fix fails.
        '''
        component = component_failure_pair[0]
        failure = component_failure_pair[1]

        # get the failing probability
        failing_components = list(set([i[0] for i in list_of_failings]))
        failing_probability = self.transition.return_failing_probability(component, failing_components)

        # will the fixing fail?
        if random.random() <= failing_probability:
            # yes, it fails
            return 0
        else:  # no, the fixing will not fail
            # reduce list of possible rewards to the selection
            filtered = self.data[
                (self.data[self.data.columns[0]] == component) & (self.data[self.data.columns[1]] == failure)]

            # select a reward
            sample_value = 0
            if self.environment[0] in ['ARol', 'GARCH']:
                # in the non-stationary enviornment a already used value is removed from the list of possible reward values
                sample_value = filtered.iloc[0][filtered.columns[2]]
                index = filtered.index[0]
                self.data = self.data.drop(index)
            else:
                sample_value = filtered.sample()[self.data.columns[2]].values[0]
            return sample_value
