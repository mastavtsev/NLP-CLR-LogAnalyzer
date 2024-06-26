import os

import pm4py

from log_processor import LogProcessor
from trace_evaluator import TraceEvaluatorDB
from transformers import SqueezeBertForMaskedLM, PreTrainedTokenizerFast
from visualize import visualize_event_traces
from file_manager import FileManager
from pm4py.objects.conversion.log import converter as log_converter

import warnings

warnings.filterwarnings("ignore")


class UserInteractionHandler:
    def __init__(self):
        self.model, self.tokenizer = self.__load_model_and_tokenizer()
        self.log_processor = None

    def __load_model_and_tokenizer(self):
        base_dir = os.path.dirname(__file__)
        model_dir = os.path.join(base_dir, 'model', 'squeezebert')
        tokenizer_dir = os.path.join(base_dir, 'model', 'tokenizer')
        model = SqueezeBertForMaskedLM.from_pretrained(model_dir)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        return model, tokenizer

    def __find_patterns(self, LoA):
        if not self.log_processor:
            self.log_processor = LogProcessor(self.xes_file)

        tokenized_log = self.log_processor.create_tokenized_event_log(LoA=LoA)
        dataframe = log_converter.apply(tokenized_log, variant=log_converter.Variants.TO_DATA_FRAME)
        visualize_event_traces(dataframe, self.xes_file, LoA)
        self.save_tokenized_event_log(dataframe, LoA)

    def __find_anomalies(self):
        if not self.log_processor:
            self.log_processor = LogProcessor(self.xes_file)

        traces = self.log_processor.get_traces()

        base_dir = os.path.dirname(__file__)
        db_dir = os.path.join(base_dir, 'db')

        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

        dp_path = os.path.join(db_dir, "trace_evaluator.db")

        te = TraceEvaluatorDB(self.model, self.tokenizer, db_path=dp_path)
        eval_results = te.evaluate_traces(traces)
        self.display_anomalies(eval_results)

    def save_tokenized_event_log(self, dataframe, LoA):
        print("Do you want to save tokenized event log? ")
        print("1. Yes")
        print("2. No")
        action = int(input("Enter 1, 2: "))

        if action == 1:
            filename = FileManager.get_filename(self.xes_file)
            default_filename = f"{filename}_LoA_{LoA}.xes"
            output_path = FileManager.get_save_path(default_filename, ".xes")
            pm4py.write_xes(dataframe, output_path)

    @staticmethod
    def display_anomalies(results):
        for trace_idx, trace_result in enumerate(results, 1):
            print(f"Trace {trace_idx}:")
            rate = 0
            for method_idx, (status, details) in enumerate(trace_result, 1):
                if status == 'abnormal':
                    rate += 1
                print(f"  Method {method_idx} - Status: {status}")

            if rate >= 2:
                print("\n  Final trace status: abnormal")
            else:
                print("\n  Final trace status: normal")
            print()

    @staticmethod
    def request_level_of_abstraction():
        while True:
            try:
                LoA = int(input("Please enter the level of abstraction (1-13): "))
                if LoA < 1 or LoA > 13:
                    raise ValueError
                return LoA
            except ValueError:
                print("Invalid level of abstraction. Please enter a number between 1 and 13.")

    @staticmethod
    def validate_file_path(xes_file):
        if not os.path.exists(xes_file):
            print(f"File '{xes_file}' does not exist. Please try again.")
            return False
        return True

    def __get_log(self):
        print("Please select XES log file: ")
        xes_file = FileManager.get_in_path("xes-anomaly-detector", "xes")
        if self.validate_file_path(xes_file):
            self.xes_file = xes_file
        else:
            print("File is not valid")

    def process_action(self, action):
        if action == 1:
            LoA = self.request_level_of_abstraction()
            self.__find_patterns(LoA)
        elif action == 2:
            self.__find_anomalies()
        elif action == 3:
            self.__get_log()
        elif action == 4:
            print("Exiting the program.")
            exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

    def run(self):
        print("Welcome to the XES Anomaly Detector!")
        self.__get_log()

        if not self.xes_file:
            return

        while True:
            print("Choose an action: ")
            print("1. Find patterns")
            print("2. Find anomalies")
            print("3. Change log")
            print("4. Exit")
            action = int(input("Enter 1, 2, 3, or 4: "))

            self.process_action(action)


def main():
    handler = UserInteractionHandler()
    handler.run()


if __name__ == '__main__':
    main()
