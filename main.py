import torch
from app.bpetokenizer.bpetokenizer import BPETokenizer
from app.embeddings.embeddings import TransformerEmbedding
from app.settings.app_settings import Settings

tokenizer = BPETokenizer(vocab_size=Settings.VOCAB_SIZE)

tokenizer.train(["test.txt"])
tokenizer.save(str(Settings.ASSETS_DIR / "model_tokens.json"))

texto_prueba = "La navidad es satanica porque yisus no nacio el 25 de diciembre"
input_ids = tokenizer.encode(texto_prueba)
print(input_ids)

embedder = TransformerEmbedding(vocab_size=Settings.VOCAB_SIZE, d_model= Settings.D_MODEL, max_seq_len=Settings.MAX_SEQ_LEN)

input_tensor = torch.tensor(input_ids).unsqueeze(0)

with torch.no_grad():
    vectors = embedder(input_tensor)

print(f"Forma del tensor: {vectors.shape}")
print(f"Primeros 100 valores del tensor: \n{vectors[0, 0: 100]}")