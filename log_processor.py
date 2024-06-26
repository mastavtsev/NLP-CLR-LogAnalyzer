from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pm4py

from processor import XESTracesProcessor
from traces_processor import TracesProcessor


class LogProcessor:
    max_LoA = 13

    def __init__(self,
                 xes_path,
                 case_id='ManagedThreadId',
                 activity_key='concept:name',
                 timestamp_key='time:timestamp'
                 ):

        self.xes_path = xes_path

        self.case_id = case_id
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key

        self.tokenized_log = None
        self.LoA = None

        self.traces_processor = TracesProcessor()
        self.traces = None

    def __extract_traces(self):
        processor = XESTracesProcessor(self.xes_path)
        self.traces = processor.process_file()

    def __process_trace(self, case_id, tokens, start_time):
        time = start_time
        prev_act = None
        events = []
        for activity in tokens:
            if activity != prev_act:
                events.append({
                    "case:concept:name": case_id,
                    "concept:name": activity,
                    "time:timestamp": time
                })
                time += timedelta(minutes=1)
                prev_act = activity
        return events

    def create_tokenized_event_log(self, LoA):
        if LoA <= 0 or LoA > self.max_LoA:
            raise ValueError(f"LOA must be between 1 and {self.max_LoA}")

        if self.tokenized_log is not None and self.LoA == LoA:
            return self.tokenized_log

        if not self.traces:
            self.__extract_traces()

        processed_traces = self.traces_processor.process_traces(self.traces, LoA)
        events = []
        start_time = datetime.now()

        print('\033[93m' + "finishing dataframe creation ..." + '\033[0m')
        with ThreadPoolExecutor() as executor:
            futures = []
            for case_id, tokens in enumerate(processed_traces, start=1):
                futures.append(executor.submit(self.__process_trace, case_id, tokens, start_time))

            for future in as_completed(futures):
                events.extend(future.result())

        log_df = pd.DataFrame(events)

        format_df = pm4py.format_dataframe(log_df, case_id='case:concept:name',
                                           activity_key='concept:name',
                                           timestamp_key='time:timestamp')

        self.tokenized_log = pm4py.convert_to_event_log(format_df)
        self.LoA = LoA

        return self.tokenized_log

    def get_traces(self):
        if not self.traces:
            self.__extract_traces()

        return self.traces


