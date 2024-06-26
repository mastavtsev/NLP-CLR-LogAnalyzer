import json
import os


class TraceProcessor:
    base_dir = os.path.dirname(__file__)
    file_path_e_v = os.path.join(base_dir, 'data', 'event_codes.json')
    event_codes = None

    @classmethod
    def initialize_event_codes(cls):
        with open(cls.file_path_e_v, 'r', encoding='utf-8') as file:
            cls.event_codes = json.load(file)

    @staticmethod
    def get_event_code(event_name):
        return TraceProcessor.event_codes.get(event_name, None)

    @classmethod
    def get_chars(cls, trace):
        if cls.event_codes is None:
            cls.initialize_event_codes()
        sequence = ''
        for event in trace:
            event_code = cls.get_event_code(event)
            if event_code is not None:
                sequence += chr(event_code)
        return sequence
    