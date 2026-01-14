install:
 - python3 -m venv venv
 - /venv/bin/pip install -r requirements.txt

Rodar:
 - /venv/bin/python codigo_1_geracao_dados_sinteticos.py



Esta atividade tem como objetivo comparar empiricamente o desempenho de duas arquiteturas de GANs: Vanilla GAN e Wasserstein GAN. Voc�s receber�o tr�s c�digos Python que devem ser executados na sequ�ncia indicada abaixo. Todos os c�digos cont�m documenta��o extensa e coment�rios did�ticos para auxili�-los.
�Primeiro, execute o arquivo�. Este c�digo criar� um dataset artificial de 1000 imagens 16x16 representando tr�s formas geom�tricas (c�rculos, quadrados e tri�ngulos). O c�digo gera o arquivo��(que ser� usado no treinamento) e uma visualiza��o��para voc� verificar os dados gerados. Tempo estimado: 1 minuto.
�Em seguida, execute o arquivo�. Este c�digo implementa e treina tanto a Vanilla GAN quanto a Wasserstein GAN usando o dataset gerado anteriormente. O treinamento roda 100 �pocas para cada GAN e salva automaticamente amostras de imagens geradas ao longo do processo nas pastas��e�. As m�tricas de treinamento (perdas, acur�cias, tempo) s�o salvas na pasta�. Observe no terminal o progresso de cada �poca. Tempo estimado: 30-60 minutos dependendo do seu hardware (mais r�pido com GPU).
�Ap�s o treinamento completo, execute o arquivo�. Este c�digo carrega as m�tricas salvas e gera automaticamente gr�ficos comparativos e um relat�rio textual detalhado na pasta�. Voc� encontrar� os arquivos��(com 4 gr�ficos comparativos),��(an�lise temporal) e��(relat�rio com conclus�es). Leia atentamente todos os arquivos gerados. Tempo estimado: 10 segundos.
�Ap�s completar a execu��o baseline (etapas 1-3), voc� DEVE realizar pelo menos 3 experimentos adicionais modificando hiperpar�metros no arquivo�. Os hiperpar�metros est�o claramente marcados no in�cio do c�digo (LATENT_DIM, BATCH_SIZE, LEARNING_RATE_G, LEARNING_RATE_D, NUM_EPOCHS, N_CRITIC, WEIGHT_CLIP). Para cada experimento: modifique um ou mais hiperpar�metros, execute novamente o c�digo 2, renomeie as pastas de sa�da para n�o sobrescrever os resultados anteriores, execute o c�digo 3 para gerar nova an�lise, e documente todas as modifica��es e resultados.
�Ap�s realizar todos os experimentos, elabore um relat�rio t�cnico em PDF contendo: introdu��o com contextualiza��o te�rica, fundamenta��o matem�tica das duas GANs, metodologia detalhando dataset e hiperpar�metros, resultados e discuss�o de TODOS os experimentos com gr�ficos e tabelas, e conclus�es comparando o desempenho das duas arquiteturas. A estrutura completa do relat�rio est� detalhada no arquivo��que acompanha os c�digos.
�Python 3.7+, PyTorch, NumPy e Matplotlib. Instale com: pip install torch numpy matplotlib
