from transformers import AutoTokenizer, AutoModel
import torch

class CodeBERTEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def embed_documents(self, texts):
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)  # Use self.device
        with torch.no_grad():
            embeddings = self.model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[0].mean(dim=1)
        return embeddings.cpu().detach().numpy()

    def embed_query(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)  # Use self.device
        with torch.no_grad():
            embeddings = self.model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[0].mean(dim=1)
        return embeddings.cpu().detach().numpy().flatten()  # 1D array for compatibility



