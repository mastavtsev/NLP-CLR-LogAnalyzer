import torch
import numpy as np
import torch.nn.functional as F
from transformers import BatchEncoding
from trace_database import TraceDatabase
from trace_processor import TraceProcessor
from tqdm import tqdm
import copy


class TraceEvaluatorDB:
    def __init__(self, model, tokenizer, AET=0.05, APT=0.85, BS=0.5, mask_share=0.2, db_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.abnormal_error_threshold = AET
        self.abnormal_prob_threshold = APT
        self.brier_score_threshold = BS
        self.mask_share = mask_share
        self.db = TraceDatabase(db_path) if db_path else None

    def evaluate_traces(self, traces: list[list[str]]):
        if not all(isinstance(trace, list) and all(isinstance(elem, str) for elem in trace) for trace in traces):
            raise ValueError("traces must be list of str lists")

        str_abnormal = "abnormal"
        eval_results = []

        for trace in tqdm(traces, desc='analyzing anomalies by model'):
            trace_results = [["normal", []] for _ in range(3)]

            if self.db:
                trace_id, db_results = self.db.get_trace(trace)
            else:
                trace_id, db_results = None, None

            if db_results:
                probs_anomalies, error_anomalies, brier_scores = db_results
            else:
                tokenized_trace_init = self.__preprocess_trace(trace)
                tokens = self.tokenizer.convert_ids_to_tokens(tokenized_trace_init['input_ids'][0])
                brier_scores = self.evaluate_trace_brier(tokenized_trace_init, tokens)
                probs_anomalies, error_anomalies = self.evaluate_trace_by_tokens(tokenized_trace_init, tokens)

                if self.db:
                    self.db.save_trace(trace, probs_anomalies, error_anomalies, brier_scores)

            if len(probs_anomalies) > 0:
                trace_results[0] = [str_abnormal, probs_anomalies]
            if len(error_anomalies) > 0:
                trace_results[1] = [str_abnormal, error_anomalies]
            if len(brier_scores) > 0:
                trace_results[2] = [str_abnormal, brier_scores]

            eval_results.append(trace_results)

        return eval_results

    def __preprocess_trace(self, trace):
        sequence = TraceProcessor.get_chars(trace)
        tokenized_trace_init = self.tokenizer(sequence, return_tensors="pt").to(self.device)

        return tokenized_trace_init

    def brier_multi(self, targets, probs):
        return np.mean(np.sum((probs - targets) ** 2))

    def mask_tokens_and_evaluate(self, tokenized_trace_init, num_tokens, mask_indices):
        tokenized_trace = copy.deepcopy(tokenized_trace_init)
        true_indices_20pct = [tokenized_trace['input_ids'][0][idx].item() for idx in mask_indices]

        for idx in mask_indices:
            tokenized_trace['input_ids'][0][idx] = self.mask_token_id

        with torch.no_grad():
            logits = self.model(**tokenized_trace).logits

        predicted_probs = []
        true_labels = []
        for idx, true_idx in zip(mask_indices, true_indices_20pct):
            mask_token_logits = logits[0, idx, :]
            mask_token_probs = F.softmax(mask_token_logits, dim=-1).cpu().numpy()
            predicted_probs.append(mask_token_probs)

            true_label = np.zeros_like(mask_token_probs)
            true_label[true_idx] = 1
            true_labels.append(true_label)

        predicted_probs_array = np.array(predicted_probs)
        true_labels_array = np.array(true_labels)

        return self.brier_multi(true_labels_array, predicted_probs_array)

    def evaluate_trace_brier(self, tokenized_trace_init, tokens):
        brier_scores = []
        batch_encodings = [tokenized_trace_init] if len(
            tokenized_trace_init['input_ids'][0]) <= 510 else self.split_trace_to_batch_encoding(tokenized_trace_init)

        for tokenized_trace in batch_encodings:
            num_tokens = len(tokenized_trace['input_ids'][0])
            for _ in range(10):
                mask_indices_20pct = np.random.choice(num_tokens, int(num_tokens * self.mask_share), replace=False)
                brier_score = self.mask_tokens_and_evaluate(tokenized_trace, num_tokens, mask_indices_20pct)
                if brier_score > self.brier_score_threshold:
                    masked_tokens = [tokens[idx] for idx in mask_indices_20pct]
                    brier_scores.append((brier_score, masked_tokens))
        return brier_scores

    def evaluate_token(self, tokenized_trace, idx):
        true_idx = tokenized_trace['input_ids'][0][idx].item()
        tokenized_trace['input_ids'][0][idx] = self.mask_token_id
        with torch.no_grad():
            logits = self.model(**tokenized_trace).logits
        mask_token_logits = logits[0, idx, :]
        abnormal_error = F.cross_entropy(mask_token_logits.view(1, -1).to(self.device),
                                         torch.tensor([true_idx]).to(self.device))
        abnormal_prob = F.softmax(mask_token_logits, dim=-1)[true_idx].item()
        token_value = self.tokenizer.convert_ids_to_tokens(true_idx)
        error_anomaly = (abnormal_error.item(), token_value) if abnormal_error > self.abnormal_error_threshold else None
        prob_anomaly = (abnormal_prob, token_value) if abnormal_prob < self.abnormal_prob_threshold else None
        return prob_anomaly, error_anomaly

    def split_trace_to_batch_encoding(self, tokenized_trace):
        max_length = 510
        input_ids = tokenized_trace['input_ids'][0]
        token_type_ids = tokenized_trace['token_type_ids'][0]
        attention_mask = tokenized_trace['attention_mask'][0]

        def split_component(component):
            return [component[i:i + max_length] for i in range(0, len(component), max_length)]

        input_ids_chunks = split_component(input_ids)
        token_type_ids_chunks = split_component(token_type_ids)
        attention_mask_chunks = split_component(attention_mask)

        batch_encodings = []
        for i in range(len(input_ids_chunks)):
            batch_encoding = BatchEncoding({
                'input_ids': input_ids_chunks[i].unsqueeze(0),
                'token_type_ids': token_type_ids_chunks[i].unsqueeze(0),
                'attention_mask': attention_mask_chunks[i].unsqueeze(0)
            }, tensor_type='pt')
            batch_encodings.append(batch_encoding)
        return batch_encodings

    def evaluate_trace_by_tokens(self, tokenized_trace_init, tokens):
        error_anomalies = []
        probs_anomalies = []
        batch_encodings = [tokenized_trace_init] if len(
            tokenized_trace_init['input_ids'][0]) <= 510 else self.split_trace_to_batch_encoding(tokenized_trace_init)

        for tokenized_trace in batch_encodings:
            num_tokens = len(tokenized_trace['input_ids'][0])
            for idx in range(num_tokens):
                tokenized_trace_copy = copy.deepcopy(tokenized_trace)
                if tokenized_trace['input_ids'][0][idx] == self.pad_token_id:
                    break
                prob_anomaly, error_anomaly = self.evaluate_token(tokenized_trace_copy, idx)
                if error_anomaly:
                    error_anomalies.append(error_anomaly)
                if prob_anomaly:
                    probs_anomalies.append(prob_anomaly)
        return probs_anomalies, error_anomalies
