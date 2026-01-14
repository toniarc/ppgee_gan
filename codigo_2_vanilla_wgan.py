"""
================================================================================
IMPLEMENTAÇÃO DE VANILLA GAN E WASSERSTEIN GAN
--------------------------------------------------------------------------------
Autor: Prof. Dr. Bruno Duarte Gomes
Disciplina: IA Generativa (PPGEE0248)
Programa de Pós-Graduação em Engenharia Elétrica - UFPA
Data: Janeiro de 2026
================================================================================

Este código implementa duas arquiteturas de GAN:
1. Vanilla GAN (GAN Original de Goodfellow et al., 2014)
2. Wasserstein GAN (WGAN com weight clipping)

OBJETIVO DA TAREFA:
------------------
Comparar o tempo de convergência e estabilidade de treinamento entre
Vanilla GAN e Wasserstein GAN usando o mesmo dataset sintético.

ATENÇÃO PARA OS ALUNOS:
-----------------------
- Os hiperparâmetros que DEVEM ser modificados estão claramente marcados
- Cada método e classe possui documentação detalhada
- O código salva métricas de treinamento para análise posterior
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from typing import Tuple, Dict, List
import os

# Verifica se GPU está disponível e compatível
def get_device():
    if torch.cuda.is_available():
        try:
            # Testa se o CUDA é compatível criando um tensor pequeno
            test_tensor = torch.zeros(1, device='cuda')
            del test_tensor
            return torch.device('cuda')
        except Exception as e:
            print(f"CUDA disponível mas incompatível: {e}")
            print("Usando CPU em vez da GPU incompatível.")
            return torch.device('cpu')
    else:
        return torch.device('cpu')

device = get_device()
print(f"Dispositivo utilizado: {device}")

# ============================================================================
# HIPERPARÂMETROS CONFIGURÁVEIS - ALUNOS DEVEM MODIFICAR ESTES VALORES
# ============================================================================

# --- PARÂMETROS DO DATASET ---
IMG_SIZE = 16                    # Tamanho das imagens (deve corresponder ao dataset)
IMG_CHANNELS = 1                 # Número de canais (1 para grayscale)

# --- PARÂMETROS DO ESPAÇO LATENTE ---
LATENT_DIM = 32                  # Dimensão do vetor latente z ~ N(0,I)
                                 # MODIFIQUE: Teste valores como 16, 32, 64, 100

# --- PARÂMETROS DE TREINAMENTO ---
BATCH_SIZE = 64                  # Tamanho do mini-batch
                                 # MODIFIQUE: Teste 32, 64, 128
LEARNING_RATE_G = 0.0002         # Taxa de aprendizado do Gerador
                                 # MODIFIQUE: Teste 0.0001, 0.0002, 0.0005
LEARNING_RATE_D = 0.0002         # Taxa de aprendizado do Discriminador/Critic
                                 # MODIFIQUE: Teste 0.0001, 0.0002, 0.0005
BETA1 = 0.5                      # Beta1 para Adam optimizer
BETA2 = 0.999                    # Beta2 para Adam optimizer

NUM_EPOCHS = 100                 # Número de épocas de treinamento
                                 # MODIFIQUE: Ajuste conforme necessário (50, 100, 200)

# --- PARÂMETROS ESPECÍFICOS DA WGAN ---
N_CRITIC = 5                     # Número de atualizações do critic por atualização do generator
                                 # MODIFIQUE: Teste 1, 5, 10 (WGAN geralmente usa 5)
WEIGHT_CLIP = 0.01               # Limite para weight clipping no WGAN
                                 # MODIFIQUE: Teste 0.01, 0.05, 0.1

# --- PARÂMETROS DE MONITORAMENTO ---
SAVE_INTERVAL = 10               # Salvar amostras a cada N épocas
RANDOM_SEED = 42                 # Seed para reprodutibilidade

# ============================================================================


class Generator(nn.Module):
    """
    GERADOR (Generator Network)

    Arquitetura do Gerador usando Transposed Convolutions para upsampling.

    Fluxo da Arquitetura:
    --------------------
    Input: z ~ N(0,I) com dimensão (batch_size, latent_dim)

    1. Linear Layer: latent_dim -> 128 * (img_size//4) * (img_size//4)
       Transforma vetor latente em feature map inicial

    2. Reshape: (batch_size, 128, img_size//4, img_size//4)
       Prepara para camadas convolucionais

    3. TransposedConv2d + BatchNorm + ReLU: 128 -> 64 channels
       Dobra resolução espacial (upsampling)

    4. TransposedConv2d + BatchNorm + ReLU: 64 -> 32 channels
       Novamente dobra resolução espacial

    5. TransposedConv2d + Tanh: 32 -> img_channels
       Gera imagem final com valores em [-1, 1]

    Output: Imagem sintética de dimensão (batch_size, img_channels, img_size, img_size)

    IMPORTANTE:
    ----------
    - BatchNorm estabiliza o treinamento
    - ReLU adiciona não-linearidade
    - Tanh na saída mapeia para [-1, 1] (compatível com normalização dos dados)
    - Transposed Convolution realiza upsampling aprendível
    """

    def __init__(self, latent_dim: int, img_channels: int, img_size: int):
        """
        Inicializa o Gerador.

        Args:
            latent_dim: Dimensão do espaço latente
            img_channels: Número de canais da imagem de saída
            img_size: Tamanho da imagem (altura = largura)
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.init_size = img_size // 4  # Tamanho inicial da feature map

        # Camada Linear para expandir vetor latente
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size * self.init_size)
        )

        # Camadas Convolucionais Transpostas (Deconvolution) para upsampling
        self.conv_blocks = nn.Sequential(
            # Bloco 1: 128 -> 64 channels, dobra resolução
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # Upsampling por fator 2
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Bloco 2: 64 -> 32 channels, dobra resolução
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Camada final: 32 -> img_channels
            nn.Conv2d(32, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Saída em [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do Gerador.

        Args:
            z: Vetor latente de forma (batch_size, latent_dim)

        Returns:
            Imagem gerada de forma (batch_size, img_channels, img_size, img_size)
        """
        # Expande vetor latente
        out = self.fc(z)
        # Reshape para formato de imagem
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # Aplica convoluções transpostas
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    """
    DISCRIMINADOR (Discriminator Network) - Para Vanilla GAN

    Arquitetura do Discriminador usando Convolutions para classificação binária.

    Fluxo da Arquitetura:
    --------------------
    Input: Imagem de dimensão (batch_size, img_channels, img_size, img_size)

    1. Conv2d + LeakyReLU: img_channels -> 32 channels
       Extração inicial de features

    2. Conv2d + BatchNorm + LeakyReLU: 32 -> 64 channels
       Downsampling e extração de features de nível médio

    3. Conv2d + BatchNorm + LeakyReLU: 64 -> 128 channels
       Downsampling e extração de features de alto nível

    4. Flatten + Linear + Sigmoid: 128*spatial -> 1
       Classificação binária: Real (1) ou Fake (0)

    Output: Probabilidade de a imagem ser real, valor em [0, 1]

    IMPORTANTE:
    ----------
    - LeakyReLU previne "dying ReLU" (neurônios mortos)
    - BatchNorm estabiliza treinamento
    - Sigmoid na saída fornece probabilidade
    - Não usa pooling (usa stride para downsampling)
    """

    def __init__(self, img_channels: int, img_size: int):
        """
        Inicializa o Discriminador.

        Args:
            img_channels: Número de canais da imagem de entrada
            img_size: Tamanho da imagem (altura = largura)
        """
        super(Discriminator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size

        # Camadas Convolucionais para extração de features
        self.conv_blocks = nn.Sequential(
            # Bloco 1: img_channels -> 32
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # Bloco 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # Bloco 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        # Calcula tamanho após convoluções
        ds_size = img_size // 8

        # Camada de classificação final
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size * ds_size, 1),
            nn.Sigmoid()  # Saída em [0, 1]
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do Discriminador.

        Args:
            img: Imagem de forma (batch_size, img_channels, img_size, img_size)

        Returns:
            Probabilidade de a imagem ser real, forma (batch_size, 1)
        """
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)  # Flatten
        validity = self.adv_layer(out)
        return validity


class Critic(nn.Module):
    """
    CRITIC (Critic Network) - Para Wasserstein GAN

    Arquitetura idêntica ao Discriminador, mas SEM sigmoid na saída.

    DIFERENÇA FUNDAMENTAL:
    ---------------------
    - Discriminador (Vanilla GAN): Saída com Sigmoid -> probabilidade [0,1]
    - Critic (WGAN): Saída LINEAR -> score real (pode ser qualquer valor)

    O Critic não classifica como Real/Fake, mas atribui um SCORE que
    representa "quão real" a imagem parece. Esse score é usado para
    calcular a Wasserstein distance.

    IMPORTANTE:
    ----------
    - A ausência do Sigmoid permite gradientes mais estáveis
    - Os pesos do Critic devem ser clipados (weight clipping)
    - O Critic deve ser treinado mais vezes que o Generator (tipicamente 5x)
    """

    def __init__(self, img_channels: int, img_size: int):
        """
        Inicializa o Critic.

        Args:
            img_channels: Número de canais da imagem de entrada
            img_size: Tamanho da imagem (altura = largura)
        """
        super(Critic, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size

        # Camadas Convolucionais (idênticas ao Discriminador)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        ds_size = img_size // 8

        # Camada de scoring final (SEM Sigmoid!)
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size * ds_size, 1)
            # Note: Sem Sigmoid! Saída pode ser qualquer valor real
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do Critic.

        Args:
            img: Imagem de forma (batch_size, img_channels, img_size, img_size)

        Returns:
            Score de realidade, forma (batch_size, 1)
        """
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        score = self.adv_layer(out)
        return score


def weights_init(m):
    """
    Inicialização de pesos usando distribuição normal.

    Esta função é aplicada recursivamente a todas as camadas da rede.
    Inicialização adequada é CRUCIAL para convergência do GAN.

    Args:
        m: Módulo (camada) da rede neural
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Inicializa convoluções com média 0 e desvio padrão 0.02
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Inicializa BatchNorm com média 1 e desvio padrão 0.02
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def load_dataset(filename: str = 'synthetic_dataset.pkl') -> DataLoader:
    """
    Carrega o dataset sintético e cria DataLoader.

    Args:
        filename: Nome do arquivo pickle contendo o dataset

    Returns:
        DataLoader pronto para treinamento
    """
    print(f"Carregando dataset de '{filename}'...")

    with open(filename, 'rb') as f:
        dataset = pickle.load(f)

    images = dataset['images']

    # Normaliza imagens para [-1, 1] (compatível com Tanh)
    images = (images - 0.5) / 0.5

    # Reorganiza dimensões: (N, H, W, C) -> (N, C, H, W)
    images = np.transpose(images, (0, 3, 1, 2))

    # Converte para tensor
    images_tensor = torch.FloatTensor(images)

    # Cria dataset e dataloader
    tensor_dataset = TensorDataset(images_tensor)
    dataloader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Dataset carregado: {len(images)} imagens")
    print(f"Formato das imagens: {images_tensor.shape}")

    return dataloader


class VanillaGAN:
    """
    VANILLA GAN - Implementação da GAN Original

    FUNÇÃO DE PERDA (Binary Cross-Entropy):
    ---------------------------------------
    L_D = -E[log(D(x))] - E[log(1 - D(G(z)))]    (Discriminador maximiza)
    L_G = -E[log(D(G(z)))]                         (Gerador maximiza)

    PROCESSO DE TREINAMENTO:
    -----------------------
    Para cada mini-batch:
        1. Atualiza Discriminador:
           - Calcula perda em imagens reais
           - Calcula perda em imagens fake
           - Backpropagation e otimização

        2. Atualiza Gerador:
           - Gera imagens fake
           - Calcula perda (quão bem enganou o discriminador)
           - Backpropagation e otimização
    """

    def __init__(self):
        """Inicializa Vanilla GAN com Generator e Discriminator."""
        # Inicializa redes
        self.generator = Generator(LATENT_DIM, IMG_CHANNELS, IMG_SIZE).to(device)
        self.discriminator = Discriminator(IMG_CHANNELS, IMG_SIZE).to(device)

        # Inicializa pesos
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        # Loss function: Binary Cross Entropy
        self.adversarial_loss = nn.BCELoss()

        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), 
                                      lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), 
                                      lr=LEARNING_RATE_D, betas=(BETA1, BETA2))

        # Para armazenar métricas
        self.metrics = {
            'd_loss': [],
            'g_loss': [],
            'd_real_acc': [],
            'd_fake_acc': [],
            'epoch_times': []
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Treina uma época da Vanilla GAN.

        Args:
            dataloader: DataLoader com imagens reais
            epoch: Número da época atual

        Returns:
            Dicionário com métricas da época
        """
        epoch_start = time.time()

        d_losses = []
        g_losses = []
        d_real_accs = []
        d_fake_accs = []

        for i, (real_imgs,) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # Labels para BCE loss
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # =====================================
            # (1) ATUALIZA DISCRIMINADOR
            # =====================================
            self.optimizer_D.zero_grad()

            # Perda em imagens reais
            real_pred = self.discriminator(real_imgs)
            d_real_loss = self.adversarial_loss(real_pred, valid)

            # Gera imagens fake
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_imgs = self.generator(z).detach()  # .detach() evita backprop no generator

            # Perda em imagens fake
            fake_pred = self.discriminator(fake_imgs)
            d_fake_loss = self.adversarial_loss(fake_pred, fake)

            # Perda total do discriminador
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            self.optimizer_D.step()

            # =====================================
            # (2) ATUALIZA GERADOR
            # =====================================
            self.optimizer_G.zero_grad()

            # Gera novas imagens fake
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            gen_imgs = self.generator(z)

            # Perda do gerador (quer que discriminador classifique como real)
            g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)

            g_loss.backward()
            self.optimizer_G.step()

            # Calcula acurácias
            d_real_acc = (real_pred > 0.5).float().mean().item()
            d_fake_acc = (fake_pred < 0.5).float().mean().item()

            # Armazena métricas
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            d_real_accs.append(d_real_acc)
            d_fake_accs.append(d_fake_acc)

        epoch_time = time.time() - epoch_start

        # Médias da época
        metrics = {
            'd_loss': np.mean(d_losses),
            'g_loss': np.mean(g_losses),
            'd_real_acc': np.mean(d_real_accs),
            'd_fake_acc': np.mean(d_fake_accs),
            'epoch_time': epoch_time
        }

        return metrics

    def train(self, dataloader: DataLoader, num_epochs: int):
        """
        Treina Vanilla GAN por múltiplas épocas.

        Args:
            dataloader: DataLoader com dados de treinamento
            num_epochs: Número de épocas
        """
        print("\n" + "="*80)
        print("TREINANDO VANILLA GAN")
        print("="*80)

        for epoch in range(num_epochs):
            metrics = self.train_epoch(dataloader, epoch)

            # Armazena métricas
            self.metrics['d_loss'].append(metrics['d_loss'])
            self.metrics['g_loss'].append(metrics['g_loss'])
            self.metrics['d_real_acc'].append(metrics['d_real_acc'])
            self.metrics['d_fake_acc'].append(metrics['d_fake_acc'])
            self.metrics['epoch_times'].append(metrics['epoch_time'])

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Época [{epoch+1}/{num_epochs}] | "
                      f"D Loss: {metrics['d_loss']:.4f} | "
                      f"G Loss: {metrics['g_loss']:.4f} | "
                      f"D Real Acc: {metrics['d_real_acc']:.2%} | "
                      f"D Fake Acc: {metrics['d_fake_acc']:.2%} | "
                      f"Time: {metrics['epoch_time']:.2f}s")

            # Salva amostras
            if (epoch + 1) % SAVE_INTERVAL == 0:
                self.save_samples(epoch + 1)

        print("\nTreinamento concluído!")
        self.save_metrics('vanilla_gan')

    def save_samples(self, epoch: int, num_samples: int = 16):
        """Gera e salva amostras do gerador."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, LATENT_DIM, device=device)
            samples = self.generator(z).cpu()
        self.generator.train()

        # Desnormaliza
        samples = samples * 0.5 + 0.5
        samples = torch.clamp(samples, 0, 1)

        # Visualiza
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i].squeeze(), cmap='gray')
            ax.axis('off')
        plt.suptitle(f'Vanilla GAN - Época {epoch}')
        plt.tight_layout()

        os.makedirs('vanilla_gan_samples', exist_ok=True)
        plt.savefig(f'vanilla_gan_samples/epoch_{epoch}.png')
        plt.close()

    def save_metrics(self, prefix: str):
        """Salva métricas de treinamento."""
        os.makedirs('metrics', exist_ok=True)
        with open(f'metrics/{prefix}_metrics.pkl', 'wb') as f:
            pickle.dump(self.metrics, f)


class WassersteinGAN:
    """
    WASSERSTEIN GAN - Implementação com Weight Clipping

    FUNÇÃO DE PERDA (Wasserstein Distance):
    ---------------------------------------
    L_C = -E[C(x)] + E[C(G(z))]              (Critic minimiza)
    L_G = -E[C(G(z))]                        (Gerador minimiza)

    DIFERENÇAS FUNDAMENTAIS DA VANILLA GAN:
    --------------------------------------
    1. Usa Critic ao invés de Discriminator (sem sigmoid)
    2. Usa Wasserstein distance ao invés de BCE loss
    3. Aplica weight clipping nos pesos do Critic
    4. Treina Critic múltiplas vezes por atualização do Generator

    VANTAGENS:
    ---------
    - Gradientes mais estáveis (não satura)
    - Melhor convergência
    - Menos mode collapse
    - Wasserstein distance é métrica mais significativa
    """

    def __init__(self):
        """Inicializa WGAN com Generator e Critic."""
        # Inicializa redes
        self.generator = Generator(LATENT_DIM, IMG_CHANNELS, IMG_SIZE).to(device)
        self.critic = Critic(IMG_CHANNELS, IMG_SIZE).to(device)

        # Inicializa pesos
        self.generator.apply(weights_init)
        self.critic.apply(weights_init)

        # Optimizers (geralmente RMSprop é recomendado para WGAN, mas Adam também funciona)
        self.optimizer_G = optim.Adam(self.generator.parameters(), 
                                      lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
        self.optimizer_C = optim.Adam(self.critic.parameters(), 
                                      lr=LEARNING_RATE_D, betas=(BETA1, BETA2))

        # Para armazenar métricas
        self.metrics = {
            'c_loss': [],
            'g_loss': [],
            'wasserstein_dist': [],
            'epoch_times': []
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Treina uma época da WGAN.

        Args:
            dataloader: DataLoader com imagens reais
            epoch: Número da época atual

        Returns:
            Dicionário com métricas da época
        """
        epoch_start = time.time()

        c_losses = []
        g_losses = []
        w_dists = []

        for i, (real_imgs,) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # =====================================
            # (1) ATUALIZA CRITIC (N_CRITIC vezes)
            # =====================================
            for _ in range(N_CRITIC):
                self.optimizer_C.zero_grad()

                # Score para imagens reais
                real_score = self.critic(real_imgs)

                # Gera imagens fake
                z = torch.randn(batch_size, LATENT_DIM, device=device)
                fake_imgs = self.generator(z).detach()

                # Score para imagens fake
                fake_score = self.critic(fake_imgs)

                # Wasserstein loss para o Critic
                # Critic quer maximizar: E[C(x)] - E[C(G(z))]
                # Equivalente a minimizar: -E[C(x)] + E[C(G(z))]
                c_loss = -torch.mean(real_score) + torch.mean(fake_score)

                c_loss.backward()
                self.optimizer_C.step()

                # *** WEIGHT CLIPPING ***
                # Clipa os pesos do Critic para satisfazer restrição de Lipschitz
                for p in self.critic.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            # =====================================
            # (2) ATUALIZA GERADOR
            # =====================================
            self.optimizer_G.zero_grad()

            # Gera novas imagens
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            gen_imgs = self.generator(z)

            # Score das imagens geradas
            gen_score = self.critic(gen_imgs)

            # Gerador quer maximizar E[C(G(z))]
            # Equivalente a minimizar -E[C(G(z))]
            g_loss = -torch.mean(gen_score)

            g_loss.backward()
            self.optimizer_G.step()

            # Calcula Wasserstein distance aproximada
            w_dist = torch.mean(real_score).item() - torch.mean(fake_score).item()

            # Armazena métricas
            c_losses.append(c_loss.item())
            g_losses.append(g_loss.item())
            w_dists.append(w_dist)

        epoch_time = time.time() - epoch_start

        # Médias da época
        metrics = {
            'c_loss': np.mean(c_losses),
            'g_loss': np.mean(g_losses),
            'wasserstein_dist': np.mean(w_dists),
            'epoch_time': epoch_time
        }

        return metrics

    def train(self, dataloader: DataLoader, num_epochs: int):
        """
        Treina WGAN por múltiplas épocas.

        Args:
            dataloader: DataLoader com dados de treinamento
            num_epochs: Número de épocas
        """
        print("\n" + "="*80)
        print("TREINANDO WASSERSTEIN GAN")
        print("="*80)

        for epoch in range(num_epochs):
            metrics = self.train_epoch(dataloader, epoch)

            # Armazena métricas
            self.metrics['c_loss'].append(metrics['c_loss'])
            self.metrics['g_loss'].append(metrics['g_loss'])
            self.metrics['wasserstein_dist'].append(metrics['wasserstein_dist'])
            self.metrics['epoch_times'].append(metrics['epoch_time'])

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Época [{epoch+1}/{num_epochs}] | "
                      f"C Loss: {metrics['c_loss']:.4f} | "
                      f"G Loss: {metrics['g_loss']:.4f} | "
                      f"W Dist: {metrics['wasserstein_dist']:.4f} | "
                      f"Time: {metrics['epoch_time']:.2f}s")

            # Salva amostras
            if (epoch + 1) % SAVE_INTERVAL == 0:
                self.save_samples(epoch + 1)

        print("\nTreinamento concluído!")
        self.save_metrics('wasserstein_gan')

    def save_samples(self, epoch: int, num_samples: int = 16):
        """Gera e salva amostras do gerador."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, LATENT_DIM, device=device)
            samples = self.generator(z).cpu()
        self.generator.train()

        # Desnormaliza
        samples = samples * 0.5 + 0.5
        samples = torch.clamp(samples, 0, 1)

        # Visualiza
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i].squeeze(), cmap='gray')
            ax.axis('off')
        plt.suptitle(f'Wasserstein GAN - Época {epoch}')
        plt.tight_layout()

        os.makedirs('wgan_samples', exist_ok=True)
        plt.savefig(f'wgan_samples/epoch_{epoch}.png')
        plt.close()

    def save_metrics(self, prefix: str):
        """Salva métricas de treinamento."""
        os.makedirs('metrics', exist_ok=True)
        with open(f'metrics/{prefix}_metrics.pkl', 'wb') as f:
            pickle.dump(self.metrics, f)


def main():
    """
    Função principal que executa o treinamento comparativo.
    """
    # Define seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Carrega dados
    dataloader = load_dataset()

    print("\n" + "="*80)
    print("CONFIGURAÇÃO DO EXPERIMENTO")
    print("="*80)
    print(f"Imagem: {IMG_SIZE}x{IMG_SIZE} com {IMG_CHANNELS} canal(is)")
    print(f"Dimensão latente: {LATENT_DIM}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rates: G={LEARNING_RATE_G}, D/C={LEARNING_RATE_D}")
    print(f"Número de épocas: {NUM_EPOCHS}")
    print(f"N_critic (WGAN): {N_CRITIC}")
    print(f"Weight clip (WGAN): {WEIGHT_CLIP}")
    print("="*80)

    # Treina Vanilla GAN
    print("\n### INICIANDO TREINAMENTO DA VANILLA GAN ###")
    vanilla_gan = VanillaGAN()
    vanilla_start = time.time()
    vanilla_gan.train(dataloader, NUM_EPOCHS)
    vanilla_total_time = time.time() - vanilla_start

    # Treina Wasserstein GAN
    print("\n### INICIANDO TREINAMENTO DA WASSERSTEIN GAN ###")
    wgan = WassersteinGAN()
    wgan_start = time.time()
    wgan.train(dataloader, NUM_EPOCHS)
    wgan_total_time = time.time() - wgan_start

    # Comparação final
    print("\n" + "="*80)
    print("COMPARAÇÃO DE RESULTADOS")
    print("="*80)
    print(f"Vanilla GAN - Tempo total: {vanilla_total_time/60:.2f} minutos")
    print(f"WGAN - Tempo total: {wgan_total_time/60:.2f} minutos")
    print(f"Diferença: {abs(vanilla_total_time - wgan_total_time)/60:.2f} minutos")
    print("="*80)

    print("\nTodos os resultados foram salvos em:")
    print("  - vanilla_gan_samples/ (amostras da Vanilla GAN)")
    print("  - wgan_samples/ (amostras da WGAN)")
    print("  - metrics/ (métricas de treinamento)")


if __name__ == "__main__":
    main()
