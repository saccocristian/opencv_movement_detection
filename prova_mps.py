import torch
if torch.backends.mps.is_available():
    print("Capo Supremo, la potenza di Metal è pronta al tuo comando!")
else:
    print("C'è un problema. Stiamo ancora usando la CPU.")