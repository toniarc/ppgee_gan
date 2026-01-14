"""
================================================================================
ANÁLISE COMPARATIVA: VANILLA GAN vs WASSERSTEIN GAN
--------------------------------------------------------------------------------
Autor: Prof. Dr. Bruno Duarte Gomes
Disciplina: IA Generativa (PPGEE0248)
Programa de Pós-Graduação em Engenharia Elétrica - UFPA
Data: Janeiro de 2026
================================================================================

Este código realiza análise comparativa completa entre Vanilla GAN e WGAN.

FUNCIONALIDADES:
---------------
1. Carrega métricas salvas durante o treinamento
2. Gera gráficos comparativos de convergência
3. Calcula estatísticas de desempenho
4. Cria relatório visual completo

USO:
----
Execute após treinar ambas as GANs usando o código_2_vanilla_wgan.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import os

# Configuração de estilo para gráficos
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_metrics(filename: str) -> Dict:
    """
    Carrega métricas salvas de um arquivo pickle.

    Args:
        filename: Caminho para o arquivo de métricas

    Returns:
        Dicionário contendo as métricas
    """
    with open(filename, 'rb') as f:
        metrics = pickle.load(f)
    return metrics


def plot_loss_comparison(vanilla_metrics: Dict, wgan_metrics: Dict, save_path: str):
    """
    Plota comparação das funções de perda.

    Args:
        vanilla_metrics: Métricas da Vanilla GAN
        wgan_metrics: Métricas da WGAN
        save_path: Caminho para salvar a figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparação de Perdas: Vanilla GAN vs Wasserstein GAN', 
                 fontsize=16, fontweight='bold')

    epochs_vanilla = range(1, len(vanilla_metrics['d_loss']) + 1)
    epochs_wgan = range(1, len(wgan_metrics['c_loss']) + 1)

    # Plot 1: Perda do Discriminador/Critic
    axes[0, 0].plot(epochs_vanilla, vanilla_metrics['d_loss'], 
                    label='Vanilla GAN - Discriminador', color='blue', linewidth=2)
    axes[0, 0].plot(epochs_wgan, wgan_metrics['c_loss'], 
                    label='WGAN - Critic', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Perda')
    axes[0, 0].set_title('Perda do Discriminador/Critic')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Perda do Gerador
    axes[0, 1].plot(epochs_vanilla, vanilla_metrics['g_loss'], 
                    label='Vanilla GAN', color='blue', linewidth=2)
    axes[0, 1].plot(epochs_wgan, wgan_metrics['g_loss'], 
                    label='WGAN', color='red', linewidth=2)
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Perda')
    axes[0, 1].set_title('Perda do Gerador')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Acurácia do Discriminador (apenas Vanilla GAN)
    axes[1, 0].plot(epochs_vanilla, vanilla_metrics['d_real_acc'], 
                    label='Acurácia em Reais', color='green', linewidth=2)
    axes[1, 0].plot(epochs_vanilla, vanilla_metrics['d_fake_acc'], 
                    label='Acurácia em Fakes', color='orange', linewidth=2)
    axes[1, 0].axhline(y=0.5, color='black', linestyle='--', label='Linha de Base (50%)')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Acurácia')
    axes[1, 0].set_title('Acurácias do Discriminador (Vanilla GAN)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # Plot 4: Wasserstein Distance (apenas WGAN)
    axes[1, 1].plot(epochs_wgan, wgan_metrics['wasserstein_dist'], 
                    label='Distância de Wasserstein', color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Distância')
    axes[1, 1].set_title('Distância de Wasserstein (WGAN)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de perdas salvo em: {save_path}")
    plt.close()


def plot_convergence_speed(vanilla_metrics: Dict, wgan_metrics: Dict, save_path: str):
    """
    Plota análise da velocidade de convergência.

    Args:
        vanilla_metrics: Métricas da Vanilla GAN
        wgan_metrics: Métricas da WGAN
        save_path: Caminho para salvar a figura
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Análise de Velocidade de Convergência', 
                 fontsize=16, fontweight='bold')

    epochs_vanilla = range(1, len(vanilla_metrics['epoch_times']) + 1)
    epochs_wgan = range(1, len(wgan_metrics['epoch_times']) + 1)

    # Plot 1: Tempo por época
    axes[0].plot(epochs_vanilla, vanilla_metrics['epoch_times'], 
                 label='Vanilla GAN', color='blue', linewidth=2, marker='o', markersize=3)
    axes[0].plot(epochs_wgan, wgan_metrics['epoch_times'], 
                 label='WGAN', color='red', linewidth=2, marker='s', markersize=3)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Tempo (segundos)')
    axes[0].set_title('Tempo de Execução por Época')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Tempo acumulado
    cumulative_vanilla = np.cumsum(vanilla_metrics['epoch_times'])
    cumulative_wgan = np.cumsum(wgan_metrics['epoch_times'])

    axes[1].plot(epochs_vanilla, cumulative_vanilla / 60, 
                 label='Vanilla GAN', color='blue', linewidth=2)
    axes[1].plot(epochs_wgan, cumulative_wgan / 60, 
                 label='WGAN', color='red', linewidth=2)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Tempo Acumulado (minutos)')
    axes[1].set_title('Tempo Total de Treinamento Acumulado')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de convergência salvo em: {save_path}")
    plt.close()


def calculate_convergence_metrics(metrics: Dict, model_name: str) -> Dict:
    """
    Calcula métricas de convergência.

    Args:
        metrics: Dicionário de métricas
        model_name: Nome do modelo

    Returns:
        Dicionário com métricas calculadas
    """
    g_loss = np.array(metrics['g_loss'])

    # Calcula variação da perda do gerador
    g_loss_variance = np.var(g_loss)
    g_loss_std = np.std(g_loss)

    # Detecta convergência (quando variação se estabiliza)
    # Usa janela deslizante para calcular desvio padrão local
    window_size = 10
    local_stds = []
    for i in range(len(g_loss) - window_size):
        local_stds.append(np.std(g_loss[i:i+window_size]))

    # Época de convergência: quando desvio padrão local fica consistentemente baixo
    threshold = g_loss_std * 0.3  # 30% do desvio padrão total
    convergence_epoch = None
    for i in range(len(local_stds) - 5):
        if all(std < threshold for std in local_stds[i:i+5]):
            convergence_epoch = i + window_size
            break

    # Tempo total
    total_time = sum(metrics['epoch_times'])
    avg_time_per_epoch = np.mean(metrics['epoch_times'])

    results = {
        'model': model_name,
        'final_g_loss': g_loss[-1],
        'g_loss_variance': g_loss_variance,
        'g_loss_std': g_loss_std,
        'convergence_epoch': convergence_epoch if convergence_epoch else 'Não convergiu',
        'total_time_minutes': total_time / 60,
        'avg_time_per_epoch': avg_time_per_epoch,
        'total_epochs': len(g_loss)
    }

    return results


def generate_comparison_report(vanilla_metrics: Dict, wgan_metrics: Dict, save_path: str):
    """
    Gera relatório comparativo em texto.

    Args:
        vanilla_metrics: Métricas da Vanilla GAN
        wgan_metrics: Métricas da WGAN
        save_path: Caminho para salvar o relatório
    """
    vanilla_stats = calculate_convergence_metrics(vanilla_metrics, 'Vanilla GAN')
    wgan_stats = calculate_convergence_metrics(wgan_metrics, 'Wasserstein GAN')

    report = []
    report.append("="*80)
    report.append("RELATÓRIO COMPARATIVO: VANILLA GAN vs WASSERSTEIN GAN")
    report.append("="*80)
    report.append("")
    report.append("Autor: Prof. Dr. Bruno Duarte Gomes")
    report.append("Disciplina: IA Generativa (PPGEE0248)")
    report.append("Programa de Pós-Graduação em Engenharia Elétrica - UFPA")
    report.append("")
    report.append("="*80)
    report.append("1. MÉTRICAS DE CONVERGÊNCIA")
    report.append("="*80)
    report.append("")

    # Vanilla GAN
    report.append("VANILLA GAN:")
    report.append("-" * 40)
    report.append(f"  Perda final do gerador: {vanilla_stats['final_g_loss']:.4f}")
    report.append(f"  Variância da perda do gerador: {vanilla_stats['g_loss_variance']:.4f}")
    report.append(f"  Desvio padrão da perda: {vanilla_stats['g_loss_std']:.4f}")
    report.append(f"  Época de convergência: {vanilla_stats['convergence_epoch']}")
    report.append(f"  Acurácia final em reais: {vanilla_metrics['d_real_acc'][-1]:.2%}")
    report.append(f"  Acurácia final em fakes: {vanilla_metrics['d_fake_acc'][-1]:.2%}")
    report.append("")

    # WGAN
    report.append("WASSERSTEIN GAN:")
    report.append("-" * 40)
    report.append(f"  Perda final do gerador: {wgan_stats['final_g_loss']:.4f}")
    report.append(f"  Variância da perda do gerador: {wgan_stats['g_loss_variance']:.4f}")
    report.append(f"  Desvio padrão da perda: {wgan_stats['g_loss_std']:.4f}")
    report.append(f"  Época de convergência: {wgan_stats['convergence_epoch']}")
    report.append(f"  Distância de Wasserstein final: {wgan_metrics['wasserstein_dist'][-1]:.4f}")
    report.append("")

    # Comparação
    report.append("="*80)
    report.append("2. COMPARAÇÃO DE DESEMPENHO")
    report.append("="*80)
    report.append("")
    report.append("ESTABILIDADE DE TREINAMENTO:")
    report.append("-" * 40)

    if wgan_stats['g_loss_variance'] < vanilla_stats['g_loss_variance']:
        report.append(f"  ✓ WGAN apresentou MENOR variância na perda do gerador")
        report.append(f"    Vanilla GAN: {vanilla_stats['g_loss_variance']:.4f}")
        report.append(f"    WGAN: {wgan_stats['g_loss_variance']:.4f}")
        report.append(f"    Diferença: {vanilla_stats['g_loss_variance'] - wgan_stats['g_loss_variance']:.4f}")
        more_stable = "WGAN"
    else:
        report.append(f"  ✓ Vanilla GAN apresentou MENOR variância na perda do gerador")
        report.append(f"    Vanilla GAN: {vanilla_stats['g_loss_variance']:.4f}")
        report.append(f"    WGAN: {wgan_stats['g_loss_variance']:.4f}")
        report.append(f"    Diferença: {wgan_stats['g_loss_variance'] - vanilla_stats['g_loss_variance']:.4f}")
        more_stable = "Vanilla GAN"
    report.append("")

    report.append("VELOCIDADE DE CONVERGÊNCIA:")
    report.append("-" * 40)
    report.append(f"  Vanilla GAN convergiu na época: {vanilla_stats['convergence_epoch']}")
    report.append(f"  WGAN convergiu na época: {wgan_stats['convergence_epoch']}")

    if isinstance(vanilla_stats['convergence_epoch'], int) and isinstance(wgan_stats['convergence_epoch'], int):
        if wgan_stats['convergence_epoch'] < vanilla_stats['convergence_epoch']:
            report.append(f"  ✓ WGAN convergiu MAIS RÁPIDO")
            report.append(f"    Diferença: {vanilla_stats['convergence_epoch'] - wgan_stats['convergence_epoch']} épocas")
            faster = "WGAN"
        else:
            report.append(f"  ✓ Vanilla GAN convergiu MAIS RÁPIDO")
            report.append(f"    Diferença: {wgan_stats['convergence_epoch'] - vanilla_stats['convergence_epoch']} épocas")
            faster = "Vanilla GAN"
    else:
        report.append("  ⚠ Convergência não detectada para um ou ambos os modelos")
        faster = "Indeterminado"
    report.append("")

    report.append("TEMPO COMPUTACIONAL:")
    report.append("-" * 40)
    report.append(f"  Vanilla GAN:")
    report.append(f"    Tempo total: {vanilla_stats['total_time_minutes']:.2f} minutos")
    report.append(f"    Tempo médio por época: {vanilla_stats['avg_time_per_epoch']:.2f} segundos")
    report.append(f"  WGAN:")
    report.append(f"    Tempo total: {wgan_stats['total_time_minutes']:.2f} minutos")
    report.append(f"    Tempo médio por época: {wgan_stats['avg_time_per_epoch']:.2f} segundos")
    report.append(f"  Diferença: {abs(vanilla_stats['total_time_minutes'] - wgan_stats['total_time_minutes']):.2f} minutos")

    if wgan_stats['total_time_minutes'] > vanilla_stats['total_time_minutes']:
        overhead = ((wgan_stats['total_time_minutes'] / vanilla_stats['total_time_minutes']) - 1) * 100
        report.append(f"  ⚠ WGAN levou {overhead:.1f}% mais tempo que Vanilla GAN")
    else:
        overhead = ((vanilla_stats['total_time_minutes'] / wgan_stats['total_time_minutes']) - 1) * 100
        report.append(f"  ⚠ Vanilla GAN levou {overhead:.1f}% mais tempo que WGAN")
    report.append("")

    report.append("="*80)
    report.append("3. CONCLUSÕES")
    report.append("="*80)
    report.append("")
    report.append(f"Modelo mais estável: {more_stable}")
    report.append(f"Modelo com convergência mais rápida: {faster}")
    report.append("")
    report.append("OBSERVAÇÕES:")
    report.append("-" * 40)
    report.append("")
    report.append("• WGAN geralmente apresenta treinamento mais estável devido à")
    report.append("  Wasserstein distance, que não satura como BCE loss.")
    report.append("")
    report.append("• WGAN requer mais atualizações do critic (N_CRITIC=5), o que")
    report.append("  pode aumentar o tempo de treinamento por época.")
    report.append("")
    report.append("• A acurácia do discriminador na Vanilla GAN próxima de 50%")
    report.append("  indica equilíbrio de Nash (discriminador confuso).")
    report.append("")
    report.append("• Wasserstein distance fornece uma métrica mais interpretável")
    report.append("  de qualidade do gerador que BCE loss.")
    report.append("")
    report.append("="*80)
    report.append("FIM DO RELATÓRIO")
    report.append("="*80)

    # Salva relatório
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    # Também imprime no console
    print('\n'.join(report))


def main():
    """
    Função principal que executa toda a análise.
    """
    print("="*80)
    print("ANÁLISE COMPARATIVA: VANILLA GAN vs WASSERSTEIN GAN")
    print("="*80)
    print("")

    # Verifica se os arquivos de métricas existem
    vanilla_path = 'metrics/vanilla_gan_metrics.pkl'
    wgan_path = 'metrics/wasserstein_gan_metrics.pkl'

    if not os.path.exists(vanilla_path):
        print(f"ERRO: Arquivo não encontrado: {vanilla_path}")
        print("Execute primeiro o código_2_vanilla_wgan.py para gerar as métricas.")
        return

    if not os.path.exists(wgan_path):
        print(f"ERRO: Arquivo não encontrado: {wgan_path}")
        print("Execute primeiro o código_2_vanilla_wgan.py para gerar as métricas.")
        return

    # Carrega métricas
    print("Carregando métricas...")
    vanilla_metrics = load_metrics(vanilla_path)
    wgan_metrics = load_metrics(wgan_path)
    print("Métricas carregadas com sucesso!")
    print("")

    # Cria diretório para análises
    os.makedirs('analises', exist_ok=True)

    # Gera gráficos
    print("Gerando gráficos comparativos...")
    plot_loss_comparison(vanilla_metrics, wgan_metrics, 
                        'analises/comparacao_perdas.png')
    plot_convergence_speed(vanilla_metrics, wgan_metrics, 
                          'analises/velocidade_convergencia.png')
    print("")

    # Gera relatório
    print("Gerando relatório comparativo...")
    generate_comparison_report(vanilla_metrics, wgan_metrics, 
                             'analises/relatorio_comparativo.txt')
    print("")

    print("="*80)
    print("ANÁLISE CONCLUÍDA!")
    print("="*80)
    print("")
    print("Arquivos gerados:")
    print("  - analises/comparacao_perdas.png")
    print("  - analises/velocidade_convergencia.png")
    print("  - analises/relatorio_comparativo.txt")
    print("")
    print("="*80)


if __name__ == "__main__":
    main()
