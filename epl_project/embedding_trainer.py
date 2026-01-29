import json
import os

class EmbeddingTrainer:
    """
    [Unsloth-Powered Domain Adaptation]
    Prepares and orchestrates the fine-tuning of domain-specific embedding models.
    Focuses on tactical terminology grounding (e.g., Half Space, Inverted Fullback).
    """
    def __init__(self, trace_path="data/distilled_reasoning_traces.jsonl", train_path="data/embedding_train_data.jsonl"):
        self.trace_path = trace_path
        self.train_path = train_path
        os.makedirs(os.path.dirname(train_path), exist_ok=True)

    def prepare_training_data(self):
        """
        [Contrastive Learning Prep]
        Converts Gold Traces into (Anchor, Positive) pairs for Embedding Fine-tuning.
        Anchor: User Query containing tactical terms.
        Positive: AI-Generated high-quality tactical analysis.
        """
        if not os.path.exists(self.trace_path):
            return 0
        
        train_pairs = []
        with open(self.trace_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Create a pair: (query, output)
                # This aligns the vector space between user intent and specialist knowledge.
                pair = {
                    "anchor": data["input"],
                    "positive": data["output"]
                }
                train_pairs.append(pair)
        
        with open(self.train_path, "w", encoding="utf-8") as f:
            for pair in train_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        
        return len(train_pairs)

    def get_status_report(self):
        """Returns training readiness status."""
        count = self.prepare_training_data()
        if count < 50:
            return f"⌛ 준비 중 ({count}/50건): 데이터가 충분히 모이면 정밀 훈련이 시작됩니다."
        return f"✅ 훈련 준비 완료 ({count}건): Unsloth 기반 도메인 최적화 엔진 가동 가능."

# Singleton instance
embedding_trainer = EmbeddingTrainer()
