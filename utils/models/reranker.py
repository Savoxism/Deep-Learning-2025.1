import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Union

class RerankingModel:
    def __init__(self, model_name: str = 'BAAI/bge-reranker-base', device = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Reranker loading: {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, pairs: Union[List[Tuple[str, str]], List[List[str]]]) -> List[float]:
        """
        Input: list of [query, document]
        Output: list of float scores [0, 1]
        """
        if len(pairs) == 0:
            return []
        
        # Validate input
        pairs = [list(p) for p in pairs]
        
        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)

        # Forward pass
        outputs = self.model(**inputs, return_dict=True)
        logits = outputs.logits

        # BGE-Reranker output is a single logit per pair
        if logits.shape[1] == 1:
            scores = logits.view(-1)
        else:
            scores = logits[:, 1]

        scores = torch.sigmoid(scores)        
        return scores.cpu().tolist()

if __name__ == "__main__":
    reranker = RerankingModel()
    pairs = [
        ['what does a python eat?', 'Python is a high-level, general-purpose programming language.'],
        ['what does a python eat?', 'Pythons are non-venomous constricting snakes that eat small mammals.'],
    ]
    scores = reranker.predict(pairs)
    print("Scores:", scores)