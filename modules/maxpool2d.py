from modules.layer import Layer
#from cython_modules.maxpool2d import maxpool_forward_cython
import numpy as np

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    # Implementación original del profesor con 4 bucles anidados en Python puro.
    # Problema: recorre cada ventana una por una → muy lento.
    def _forward_naive(self, input):
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
        output = np.zeros((B, C, out_h, out_w), dtype=input.dtype)

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * SH
                        h_end = h_start + KH
                        w_start = j * SW
                        w_end = w_start + KW

                        window = input[b, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        max_val = window[max_idx]

                        output[b, c, i, j] = max_val
                        self.max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])

        return output

    def forward(self, input, training=True):
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        # --- INICIO BLOQUE GENERADO CON IA ---
        # Mejora: en vez de recorrer ventana por ventana con 4 bucles,
        # usamos reshape para organizar todas las ventanas como dimensiones separables.
        # Luego max sobre esas dimensiones de golpe → NumPy lo hace en C internamente.
        # Sin bucles Python → mucho más rápido.

        # Recortamos el input para que encaje exactamente con stride y kernel
        input_cropped = input[:, :, :out_h * SH, :out_w * SW]

        # Reshape: (B, C, out_h, SH, out_w, SW)
        # out_h y out_w = índice de ventana
        # SH y SW = posición dentro de la ventana
        windows = input_cropped.reshape(B, C, out_h, SH, out_w, SW)

        # max sobre las dimensiones del kernel (3 y 5) = máximo de cada ventana de golpe
        output = windows[:, :, :, :KH, :, :KW].max(axis=(3, 5))
        # --- FIN BLOQUE GENERADO CON IA ---

        return output

    def backward(self, grad_output, learning_rate=None):
        # ESTO NO ES NECESARIO YA QUE NO VAIS A HACER BACKPROPAGATION
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        r, s = self.max_indices[b, c, i, j]
                        grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input