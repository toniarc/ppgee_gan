"""
================================================================================
GERADOR DE DADOS ARTIFICIAIS DE IMAGENS
--------------------------------------------------------------------------------
Autor: Prof. Dr. Bruno Duarte Gomes
Disciplina: IA Generativa (PPGEE0248)
Programa de Pós-Graduação em Engenharia Elétrica - UFPA
Data: Janeiro de 2026
================================================================================

Este código gera dados artificiais de imagens usando matrizes pequenas para
minimizar o custo computacional. As imagens sintéticas serão usadas para
treinar e comparar GANs Vanilla e Wasserstein.

OBJETIVO:
---------
Criar um dataset sintético de "imagens" representadas por matrizes pequenas
com diferentes padrões geométricos (círculos, quadrados, triângulos).

ESTRUTURA DO CÓDIGO:
-------------------
1. Geração de formas geométricas simples
2. Adição de ruído controlado
3. Normalização dos dados
4. Salvamento em formato adequado para treinamento
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Tuple, List
import os

# ============================================================================
# HIPERPARÂMETROS CONFIGURÁVEIS
# ============================================================================
# MODIFIQUE ESTES VALORES PARA ALTERAR O DATASET

IMG_SIZE = 16           # Tamanho das imagens (16x16 pixels) - BAIXO CUSTO COMPUTACIONAL
NUM_SAMPLES = 999      # Número total de amostras a serem geradas
NOISE_LEVEL = 0.1       # Nível de ruído a ser adicionado (0.0 a 1.0)
RANDOM_SEED = 42        # Seed para reprodutibilidade

# ============================================================================


class SyntheticImageGenerator:
    """
    Classe para gerar imagens sintéticas com diferentes padrões geométricos.

    Attributes:
        img_size (int): Dimensão das imagens quadradas (img_size x img_size)
        noise_level (float): Quantidade de ruído gaussiano a ser adicionado
    """

    def __init__(self, img_size: int = 16, noise_level: float = 0.1):
        """
        Inicializa o gerador de imagens sintéticas.

        Args:
            img_size: Tamanho da imagem (altura = largura)
            noise_level: Desvio padrão do ruído gaussiano
        """
        self.img_size = img_size
        self.noise_level = noise_level

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Adiciona ruído gaussiano à imagem.

        Args:
            image: Imagem sem ruído

        Returns:
            Imagem com ruído adicionado e normalizada no intervalo [0, 1]
        """
        noise = np.random.normal(0, self.noise_level, image.shape)
        noisy_image = image + noise
        # Clipa valores para manter no intervalo [0, 1]
        return np.clip(noisy_image, 0, 1)

    def generate_circle(self) -> np.ndarray:
        """
        Gera uma imagem com um círculo centralizado.

        Returns:
            Matriz (img_size x img_size) representando um círculo
        """
        image = np.zeros((self.img_size, self.img_size))
        center = self.img_size // 2
        radius = self.img_size // 3

        # Cria grade de coordenadas
        y, x = np.ogrid[:self.img_size, :self.img_size]

        # Máscara circular usando equação do círculo: (x-cx)² + (y-cy)² <= r²
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        image[mask] = 1.0

        return self._add_noise(image)

    def generate_square(self) -> np.ndarray:
        """
        Gera uma imagem com um quadrado centralizado.

        Returns:
            Matriz (img_size x img_size) representando um quadrado
        """
        image = np.zeros((self.img_size, self.img_size))
        margin = self.img_size // 4

        # Define os limites do quadrado
        image[margin:-margin, margin:-margin] = 1.0

        return self._add_noise(image)

    def generate_triangle(self) -> np.ndarray:
        """
        Gera uma imagem com um triângulo centralizado.

        Returns:
            Matriz (img_size x img_size) representando um triângulo
        """
        image = np.zeros((self.img_size, self.img_size))

        # Define vértices do triângulo equilátero
        for i in range(self.img_size):
            for j in range(self.img_size):
                # Normaliza coordenadas para [-1, 1]
                x = 2 * j / self.img_size - 1
                y = 2 * i / self.img_size - 1

                # Condições para estar dentro do triângulo
                if y > -0.5 and y < 0.5 + 0.866 * x and y < 0.5 - 0.866 * x:
                    image[i, j] = 1.0

        return self._add_noise(image)

    def generate_dataset(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera um dataset completo com diferentes formas geométricas.

        Args:
            num_samples: Número total de amostras a serem geradas

        Returns:
            Tuple contendo:
                - images: Array de forma (num_samples, img_size, img_size, 1)
                - labels: Array de forma (num_samples,) com rótulos das classes
                         (0: círculo, 1: quadrado, 2: triângulo)
        """
        images = []
        labels = []

        # Distribui igualmente entre as três formas
        samples_per_class = num_samples // 3

        print(f"Gerando {samples_per_class} círculos...")
        for _ in range(samples_per_class):
            images.append(self.generate_circle())
            labels.append(0)

        print(f"Gerando {samples_per_class} quadrados...")
        for _ in range(samples_per_class):
            images.append(self.generate_square())
            labels.append(1)

        print(f"Gerando {samples_per_class} triângulos...")
        for _ in range(samples_per_class):
            images.append(self.generate_triangle())
            labels.append(2)

        # Converte para numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Adiciona dimensão de canal: (N, H, W) -> (N, H, W, 1)
        images = images[..., np.newaxis]

        # Embaralha os dados
        indices = np.random.permutation(num_samples)
        images = images[indices]
        labels = labels[indices]

        return images, labels


def visualize_samples(images: np.ndarray, labels: np.ndarray, num_samples: int = 9):
    """
    Visualiza amostras do dataset gerado.

    Args:
        images: Array de imagens
        labels: Array de rótulos
        num_samples: Número de amostras a visualizar
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle('Amostras do Dataset Sintético', fontsize=16)

    class_names = ['Círculo', 'Quadrado', 'Triângulo']

    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(images[i].squeeze(), cmap='gray')
            ax.set_title(f'Classe: {class_names[labels[i]]}')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    print("\nVisualizações salvas em 'dataset_samples.png'")
    plt.close()


def save_dataset(images: np.ndarray, labels: np.ndarray, filename: str = 'synthetic_dataset.pkl'):
    """
    Salva o dataset em arquivo pickle para uso posterior.

    Args:
        images: Array de imagens
        labels: Array de rótulos
        filename: Nome do arquivo de saída
    """
    dataset = {
        'images': images,
        'labels': labels,
        'img_size': images.shape[1],
        'num_samples': len(images)
    }

    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset salvo em '{filename}'")
    print(f"Forma das imagens: {images.shape}")
    print(f"Forma dos rótulos: {labels.shape}")
    print(f"Intervalo de valores: [{images.min():.3f}, {images.max():.3f}]")


def main():
    """
    Função principal que executa todo o pipeline de geração de dados.
    """
    print("="*80)
    print("GERADOR DE DADOS SINTÉTICOS PARA TREINAMENTO DE GANs")
    print("Prof. Dr. Bruno Duarte Gomes - PPGEE/UFPA")
    print("="*80)

    # Define seed para reprodutibilidade
    np.random.seed(RANDOM_SEED)

    # Cria o gerador
    generator = SyntheticImageGenerator(img_size=IMG_SIZE, noise_level=NOISE_LEVEL)

    # Gera o dataset
    print(f"\nGerando dataset com {NUM_SAMPLES} amostras...")
    print(f"Tamanho das imagens: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Nível de ruído: {NOISE_LEVEL}\n")

    images, labels = generator.generate_dataset(NUM_SAMPLES)

    # Visualiza amostras
    print("\nVisualizando amostras...")
    visualize_samples(images, labels)

    # Salva o dataset
    save_dataset(images, labels)

    # Estatísticas do dataset
    print("\n" + "="*80)
    print("ESTATÍSTICAS DO DATASET")
    print("="*80)
    print(f"Total de amostras: {len(images)}")
    print(f"Amostras por classe:")
    for i, class_name in enumerate(['Círculo', 'Quadrado', 'Triângulo']):
        count = np.sum(labels == i)
        print(f"  {class_name}: {count}")
    print(f"\nMemória utilizada: {images.nbytes / 1024:.2f} KB")
    print("="*80)


if __name__ == "__main__":
    main()
