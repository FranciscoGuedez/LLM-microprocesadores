import torch

# Creamos una lista con el tamaño '1' para cada una de las 256 dimensiones
dimensiones = [1] * 256

# Crear el tensor
tensor_256d = torch.randn(*dimensiones)

print(f"Número de dimensiones: {tensor_256d.ndim}")
print(f"Forma del tensor (shape): {tensor_256d.shape}")