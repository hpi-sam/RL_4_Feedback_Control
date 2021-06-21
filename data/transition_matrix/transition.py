import osimport pandas as pdclass Transition:    def __init__(self, path_to_transition_matrix: str = r'../data/transition_matrix/transition_matrix.csv'):        self.transition_matrix = self.__load_transition_matrix__(path_to_transition_matrix)        self.mapping = self.__create_mapping_list__()    def __load_transition_matrix__(self, filename):        return pd.read_csv(filename)    def __create_mapping_list__(self):        frames = []        # searching for all csv files in the data directory and loading the data in multiple dataframes        for root, dirs, files in os.walk(r'../data/original_data'):            for f in files:                if f.endswith('Linear.csv'):                    file_path = os.path.join(root, f)                    df = pd.read_csv(file_path)                    df.columns = df.columns.str.replace('\t', '')                    frames.append(df)        # combining all df to one df        mapping = pd.concat(frames, sort=False, ignore_index=True)[['Optimal_Affected_Component', 'Optimal_Affected_Component_Uid']]        # remove duplicate rows        mapping = mapping.drop_duplicates()        return mapping    def __get_name_for_id__(self, component_id: str) -> str:        '''        Returns the component name for a given id.        :param component_id: the component_id        :return: the belonging component name        '''        return self.mapping.loc[(self.mapping['Optimal_Affected_Component_Uid'] == component_id)]['Optimal_Affected_Component'].tolist()[0]    def return_failing_probability(self, component_to_be_fixed: str, failing_components: list) -> float:        '''        Returns for a component to be fixed the probability to fail.        :param component_to_be_fixed: The name of the component or its id starting with a underscore.        :param failing_components: A list of component names or ids which are still failing.        :return: A float describing the probability that the fix will failing.        '''        component = component_to_be_fixed        failings = []        # map id to name if necessary        if component_to_be_fixed.startswith('_'):            component = self.__get_name_for_id__(component_to_be_fixed)        for fail in failing_components:            if fail.startswith('_'):                failings.append(self.__get_name_for_id__(fail))            else:                failings.append(fail)        # reduce matrix to only the component and its failing components        reduced_matrix = self.transition_matrix.loc[(self.transition_matrix[self.transition_matrix.columns[0]] == component)][failings]        return sum(reduced_matrix.values.tolist()[0])