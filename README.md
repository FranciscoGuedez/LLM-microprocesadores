# LLM
## BPETokenizer

Se ubica en el módulo bpetokenizer, se trata de una clase con las siguientes características:

- GENERAL: **Encapsula la lógica de entrenamiento empleando codificación y decodificación.**
- vocab_size: Tamaño máximo del vocabulario (50257 es el estándar de GPT-2); sin embargo, solo implementamos un vocab_size de 5000.
- Entrenar: Entrena el tokenizer a partir de una lista de archivos de texto.
- Codificar: Codifica texto a una secuencia de IDs enteros (Devuelve una lista).
- Decodificar: Decodifica IDs a cadenas de texto (Devuelve una lista).
- Guarda en archivos .json.
- Puede cargar un tokenizador previamente entrenado.

### Especificaciones adicionales sobre el código (Ocupación de cada función).

Contamos con una serie de funciones que permiten el funcionamiento del tokenizador, cada una de ellas tiene un trabajo relevante dentro de "class BPETokenizer:"  

 - *Inicializa el tokenizer con un modelo BPE vacío.*
 ```python 
 def __init__(self, vocab_size: int = 50257) -> None:
 ```

 - *Entrenar. Recibe una lista de str (texto).*
 ```python
 def train(self, files: List[str]) -> None:
 ```

 - *Codificar. Recibe str (texto) y devuelve una lista de int (números).*
 ```python
 def encode(self, text: str) -> List[int]:
 ```

 - *Decodificar. Recibe una lista de int (números) y devuelve str (texto).*
 ```python
 def decode(self, ids: List[int]) -> str:
 ```

 - *Guardar. Guarda el tokenizer entrenado en un archivo JSON.*
 ```python
 def save(self, path: str) -> None:
 ```

 - *Cargar. Carga un tokenizer previamente entrenado desde un archivo.*
 ```python
 def load(cls, path: str) -> "BPETokenizer":
 ```

## TransformerEmbedding

Se ubica en el módulo embeddings, se trata de una clase con las siguientes características:

- GENERAL: **Esta clase transforma IDs de tokens en vectores densos y añade información sobre la posición de cada token en la secuencia.** Es decir, convierte tokens en vectores numericos que el modelo puede procesar.
- vocab_size: Tamaño del vocabulario del tokenizer.
- d_model: Dimensión de los vectores de embedding (espacio latente).
- max_seq_len: Longitud máxima de las secuencias de entrada.

### Especificaciones adicionales sobre el código (Ocupación de cada función).

- *Inicializar las capas de embedding, para ello necesita vocab_size: int (tamaño del vacabulario); d_model: int (Dimnsión de los vectores); max_seq_len: int (Longitud máxima de las secuencias de entrada, podemos decir que es el número máximo de tokens que el modelo puede procesar en una sola entrada). No devuenve nada solo inicializa el Embedding*
```python
def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 512) -> None:
```

- *Transforma los IDs de tokens en vectores con información "self.position_embedding" (posición) + "self.token_embedding(input_ids) * math.sqrt(self.d_model)" es decir, los vectores poseen información Semántico + Posicional. Recibe IDs de tokens y los devuelve como vectores con sentido y memoria de ubicación*
```python
def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
```
## Orquestador

Lo llamamos: main. Dentro del mismo llamamos a las clases dentro de los módulos explicados anteriormente, estas clases son:

```python
class BPETokenizer:
```
```python
class TransformerEmbedding(nn.Module):
```
Dentro de main se especifican los valores con los que vamos a estar trabajando (VOCAB_SIZE = 5000, _MODEL = 256, MAX_SEQ_LEN = 256), también el texto de prueba que vamos a usar para el entrenamiento y la ubicación de guardado del archivo .json.


