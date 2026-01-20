import torch
from app.bpetokenizer.bpetokenizer import BPETokenizer
from app.embeddings.embeddings import TransformerEmbedding

VOCAB_SIZE = 5000
D_MODEL = 256
MAX_SEQ_LEN = 256

tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)

tokenizer.train(["test.txt"])
tokenizer.save("assets/model_tokens.json")

texto_prueba = "La navidad es satanica porque yisus no nacio el 25 de diciembre"
input_ids = tokenizer.encode(texto_prueba)
print(input_ids)

embedder = TransformerEmbedding(vocab_size=VOCAB_SIZE, d_model= D_MODEL, max_seq_len=MAX_SEQ_LEN)

input_tensor = torch.tensor(input_ids).unsqueeze(0)

with torch.no_grad():
    vectors = embedder(input_tensor)

print(f"Forma del tensor: {vectors.shape}")
print(f"Primeros 100 valores del tensor: \n{vectors[0, 0: 100]}")