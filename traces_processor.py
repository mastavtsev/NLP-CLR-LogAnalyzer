import json
import os

from tqdm import tqdm

from tokenizer_manager import TokenizerManager

class TracesProcessor:
    def __init__(self):

        base_dir = os.path.dirname(__file__)  # определяем базовый каталог
        accepted_events_path = os.path.join(base_dir, 'data', 'accepted_events.json')

        with open(accepted_events_path, 'r') as json_file:
            self.accepted_events = json.load(json_file)

        accepted_indexes = [i for i in range(33, 127)]

        total_events = len(self.accepted_events)
        to_add = total_events - len(accepted_indexes)
        addition = [i for i in range(256, 256 + to_add)]
        accepted_indexes += addition

        self.event_codes = {event: accepted_indexes[index] for index, event in
                            enumerate(self.accepted_events)}

    def __get_event_code(self, event_name):
        return self.event_codes.get(event_name, None)

    def __get_sequence(self, trace):
        sequence = ''
        for event in trace:
            id = self.__get_event_code(event)
            sequence += chr(id)
        return sequence

    def __traces2seqs(self, traces):
        return [self.__get_sequence(trace) for trace in traces]

    def process_traces(self, traces, LoA):
        sequences = self.__traces2seqs(traces)

        tokenizer = TokenizerManager.get_tokenizer(LoA)
        mapper = TokenizerManager.get_mapping(LoA)

        processed_traces = []

        for sequence in tqdm(sequences, desc="tokenizing traces"):
            trace = []

            tokens = tokenizer.tokenize(sequence)

            for token in tokens:
                trace.append(mapper[token])

            processed_traces.append(trace)

        return processed_traces
