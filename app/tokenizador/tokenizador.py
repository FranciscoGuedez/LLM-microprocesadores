from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import json
from typing import Dict

class Tokenizador:
    def __init__(self):

        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

        # Variables solicitadas
        self.size: int = 1000
        self.index: Dict = {}
        self.merge: Dict = {}

        #Entrenar un vocabulario mínimo.
        trainer = trainers.BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
            show_progress=False
        )

        corpus = ["ejemplo de texto para inicializar", "otro texto de prueba"]
        total = len(corpus)

        for i, texto in enumerate(corpus, start=1):
            self.tokenizer.train_from_iterator([texto], trainer)
            porcentaje = int((i / total) * 100)
            print(f"Entrenamiento: {porcentaje}% completado")

    def analizar(self, texto: str, archivo: str = "tokens.json"):
        """
        Analiza el texto, lo tokeniza y guarda los resultados en un archivo JSON.
        """
        output = self.tokenizer.encode(texto)
        tokens = output.tokens

        # Guardamos información en las variables
        self.size = len(tokens)
        self.index = {i: tok for i, tok in enumerate(tokens)}
        self.merge = {tok: i for i, tok in enumerate(tokens)}

        # Estructura ordenada para guardar en JSON
        data = {
            "texto_original": texto,
            "cantidad_tokens": self.size,
            "index": self.index,
            "merge": self.merge
        }

        # Guardamos en archivo JSON
        with open(archivo, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return data


if __name__ == "__main__":
    print("Bienvenido al Tokenizador del equipo DC")
    texto = input("Escribe el texto que quieres analizar: ")

    tok = Tokenizador()
    resultado = tok.analizar(texto)

    print("\n Análisis completado. Resultados:")
    print(f"- Texto original: {resultado['texto_original']}")
    print(f"- Cantidad de tokens: {resultado['cantidad_tokens']}")
    print(f"- Tokens por índice: {resultado['index']}")
    print(f"- Índices por token: {resultado['merge']}")
    print("\n Los resultados también se guardaron en 'tokens.json'")