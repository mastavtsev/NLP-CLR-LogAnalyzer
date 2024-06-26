import pm4py
import pandas as pd

from tqdm import tqdm


class XESTracesProcessor:
    def __init__(self, input_filepath, output_filepath=None, num_needed_indexes=25):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.num_needed_indexes = num_needed_indexes

    def process_file(self):
        print('\033[93m' + "processing log..." + '\033[0m')

        log = self.__get_log(self.input_filepath)
        event_log = self.__get_needed_columns_log(log)

        print('\033[93m' + "extracting traces..." + '\033[0m')
        needed_indexes = self.__get_needed_indexes(event_log[event_log["ManagedThreadId"] != -1])

        outliers = self.__get_outliers(event_log, needed_indexes)

        event_log = event_log[event_log['ManagedThreadId'].isin(needed_indexes)]
        traces_log = self.__get_traces_log(event_log)
        final_trace_log = self.__get_final_log(traces_log, outliers)

        if self.output_filepath is not None:
            self.__write_traces_to_file(final_trace_log, self.output_filepath)

        return self.__get_list_of_traces(final_trace_log)

    def __get_needed_indexes(self, event_log):
        value_counts = event_log['ManagedThreadId'].value_counts()
        needed_indexes = list(value_counts[:self.num_needed_indexes].index)

        return needed_indexes

    def __get_log(self, filepath):
        return pm4py.read_xes(filepath)

    def __get_needed_columns_log(self, log):
        event_log = log[['ManagedThreadId', 'concept:name', 'time:timestamp']]
        event_log = event_log.astype({"ManagedThreadId": int, 'concept:name': str, 'time:timestamp': str})

        regex = r'(\d{2}:\d{2}:\d{2}.\d{6}\+\d{2}:\d{2})'
        event_log['time:timestamp'] = event_log['time:timestamp'].str.extract(regex)
        event_log['time:timestamp'] = pd.to_datetime(event_log['time:timestamp'])
        return event_log

    def __get_outliers(self, event_log, needed_indexes):
        outliers = event_log[~event_log['ManagedThreadId'].isin(needed_indexes)]
        return outliers

    def __get_traces_log(self, event_log):
        event_log = event_log.copy()
        event_log['time:timestamp'] = event_log['time:timestamp'].dt.tz_localize(None)
        traces = event_log.groupby('ManagedThreadId').apply(
            lambda x: [[row['concept:name'], row['time:timestamp']] for index, row in x.iterrows()]
        )
        start_times = event_log.groupby('ManagedThreadId')['time:timestamp'].min()
        end_times = event_log.groupby('ManagedThreadId')['time:timestamp'].max()
        traces_log = pd.DataFrame({
            'ManagedThreadId': traces.index,
            'Trace': traces.values,
            'Start Time': start_times.values,
            'End Time': end_times.values
        })
        return traces_log

    def __get_final_log(self, traces_log, outliers):
        traces_log = traces_log.copy()
        intervals = pd.arrays.IntervalArray.from_arrays(traces_log['Start Time'], traces_log['End Time'], closed='both')
        outliers = outliers.copy()
        outliers['time:timestamp'] = outliers['time:timestamp'].dt.tz_localize(None)

        for idx, row in tqdm(outliers.iterrows(), desc="creating dataframe", total=len(outliers)):
            timestamp = row['time:timestamp']
            concept_name = row['concept:name']
            pair = [concept_name, timestamp]
            mask = intervals.contains(timestamp)
            trace_indices = traces_log[mask].index
            for trace_idx in trace_indices:
                traces_log.at[trace_idx, 'Trace'].append(pair)

        def sort_trace(trace):
            filtered_trace = [event for event in trace if event[1] is not None]
            return [event for event, timestamp in sorted(filtered_trace, key=lambda elem_pair: elem_pair[1])]

        traces_log['Trace'] = traces_log['Trace'].apply(sort_trace)

        # traces_log['Trace'] = traces_log['Trace'].apply(
        #     lambda x: [event for event, timestamp in sorted(x, key=lambda elem_pair: elem_pair[1])])

        return traces_log

    def __write_traces_to_file(self, log, filepath):
        with open(filepath, "a") as file:
            for trace in log["Trace"]:
                file.write(' '.join(trace) + "\n")

    def __get_list_of_traces(self, log):
        traces = []
        for trace in log["Trace"]:
            traces.append(trace)
        return traces
