# padel-hit-detection

Replicação da etapa de **atribuição golpe → jogador** de Decorte et al., *Multi-Modal Hit Detection and Positional
Analysis in Padel Competitions* (CVPR Workshops 2024): dado o instante de um golpe numa partida de padel, identificar
qual dos 4 jogadores bateu na bola, combinando pose dos jogadores (YOLO11-pose + ByteTrack), trajetória da bola
(TrackNet) e um voto ponderado pela distância punho–bola.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/henrique-gerhardt/padel-hit-detection/blob/main/notebooks/replicacao_colab.ipynb)

## Resultados

Avaliação nos 319 golpes com jogador anotado do dataset do artigo (o mesmo conjunto usado por ele).
IC de 95% por bootstrap. Nenhum hiperparâmetro foi ajustado nesses golpes.

| Variante | Acurácia jogador (%) | Acurácia dupla (%) |
|---|---|---|
| Artigo (Decorte et al., 2024) | 83,70 | 86,83 |
| **Esta replicação (E1)** | **82,4** [78,4; 86,2] | **87,8** [84,3; 91,2] |
| Centro da bbox em vez do punho | 73,0 | 81,2 |
| Peso linear da eq. 1 do artigo | 74,3 | 80,6 |
| Sem bola (só velocidade do punho) | 77,1 | 85,0 |
| Sem fallback de velocidade | 78,4 | 83,1 |
| σ = 0,25 / 1,0 | 82,8 / 81,5 | 87,8 / 87,5 |
| Janela de 6 / 24 frames | 83,7 / 75,2 | 90,0 / 80,3 |

## Como rodar

O experimento inteiro está em [`notebooks/replicacao_colab.ipynb`](notebooks/replicacao_colab.ipynb), feito para o
Google Colab com GPU:

1. Abra pelo botão acima.
2. Escolha *Ambiente de execução → Alterar tipo → T4 GPU*.
3. Use *Executar tudo*.

O notebook clona este repositório, baixa os vídeos da fonte oficial e roda as etapas em ordem. Cada etapa é testada
num exemplo antes de rodar em todos os golpes, e o resultado final é comparado com a tabela acima.

| Seção | O que faz |
|---|---|
| 0–1 | Ambiente e parâmetros do experimento, todos numa célula |
| 2 | Download dos 16 rallies rotulados; anotações e homografias lidas de [`sample/`](sample/) |
| 3 | Pesos do YOLO11m-pose e do TrackNet |
| 4 | Bola: TrackNet nos 16 rallies, com remoção de outliers e interpolação |
| 5 | Pose: YOLO11m-pose (1920 px) + ByteTrack nas janelas de ±12 frames de cada golpe |
| 6 | Identidade: slot de cada jogador (fundo/frente × esquerda/direita) pela homografia |
| 7–8 | Atribuição (E1) e ablação (método, σ, janela, sem bola, sem fallback) |
| 9–10 | Análise de erros, exemplos desenhados e comparação com a referência |

Os números podem variar alguns décimos em relação à tabela, porque a pose e a bola são recalculadas noutra GPU (a
referência foi gerada num Apple M4, via MPS).

## Principais decisões da replicação

- **Instante do golpe:** é o anotado no `hit_assignments.xlsx` (o contato visível), como no artigo. O detector de
  golpes por áudio do artigo (CRNN) não foi replicado.
- **Identidade:** cada jogador recebe um slot pela posição na quadra (homografia), com voto por trilha do ByteTrack
  na janela. É uma simplificação do re-ID do artigo.
- **Atribuição:** voto `exp(-d²/2σ²)` por frame, em que `d` é a menor distância punho–bola normalizada pela altura da
  bbox. Sem bola na janela, vence a maior velocidade de punho. O peso linear da eq. 1 do artigo aparece na ablação.
- **Bola:** TrackNet pré-treinado em tênis, sem fine-tuning, com o pós-processamento do repositório usado.
- **Calibração:** uma homografia por câmera (são duas), com as retas da quadra ajustadas automaticamente numa
  imagem mediana sem jogadores. A única entrada manual são as faixas de busca de cada reta. O resultado e o modo de
  refazer estão em [`sample/README.md`](sample/README.md).

## Estrutura

```
notebooks/replicacao_colab.ipynb  experimento completo (Colab)
sample/annotations/hits.csv       319 golpes rotulados, convertidos do dataset (ver sample/README.md)
sample/calibration/               homografias da quadra por rally + parâmetros das 2 câmeras
src/padelrep/                     identidade, atribuição, métricas, calibração, anotações, vídeo
```

## Dados, créditos e licenças

- **Dataset:** vídeos e anotações são de Decorte et al. (imec / Universidade de Gent). Os vídeos **não são
  redistribuídos aqui**. A pasta `sample/` traz só derivados: as 319 anotações de jogador, convertidas para o
  formato do código, e as homografias que calculamos, com a origem descrita em `sample/README.md`. O notebook baixa
  os vídeos do
  [compartilhamento público indicado pelos autores](https://cloud.ilabt.imec.be/index.php/s/TFimLDWno6W9ED3)
  (descrição em [robbedec/datasets](https://github.com/robbedec/datasets/tree/master/CVsports/Padel)). As imagens
  são transmissões do World Padel Tour; os direitos são dos respectivos titulares.
- **TrackNet:** Huang et al. (2019), na reimplementação PyTorch
  [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) (commit `730ea17`), com os pesos de tênis do
  repositório. O código é clonado e os pesos baixados durante a execução, sem cópia aqui.
- **YOLO11-pose:** [Ultralytics](https://github.com/ultralytics/ultralytics), licenciado sob AGPL-3.0. É instalado
  como dependência, e o seu uso segue essa licença.
- **ByteTrack:** Zhang et al. (2022), pela implementação incluída na Ultralytics.

O código deste repositório está sob a licença Apache-2.0 (ver [`LICENSE`](LICENSE)). As dependências mantêm as
suas próprias licenças.

## Referências

- R. Decorte, M. Paré, J. Vanhaeverbeke, J. Taelman, M. Slembrouck e S. Verstockt. Multi-Modal Hit Detection and
  Positional Analysis in Padel Competitions. *CVPR Workshops*, 2024, pp. 3306–3314.
  DOI [10.1109/CVPRW63382.2024.00335](https://doi.org/10.1109/CVPRW63382.2024.00335).
- Y.-C. Huang, I-N. Liao, C.-H. Chen, T.-U. İk e W.-C. Peng. TrackNet: A Deep Learning Network for Tracking
  High-speed and Tiny Objects in Sports Applications. *AVSS*, 2019. DOI
  [10.1109/AVSS.2019.8909871](https://doi.org/10.1109/AVSS.2019.8909871).
- Y. Zhang et al. ByteTrack: Multi-object Tracking by Associating Every Detection Box. *ECCV*, 2022. DOI
  [10.1007/978-3-031-20047-2_1](https://doi.org/10.1007/978-3-031-20047-2_1).
- G. Jocher, J. Qiu e A. Chaurasia. Ultralytics YOLO. Software, 2023.
  <https://github.com/ultralytics/ultralytics>.

```bibtex
@inproceedings{decorte2024padel,
  author    = {Robbe Decorte and Martin Par{\'e} and Jelle Vanhaeverbeke and Joachim Taelman and Maarten Slembrouck and Steven Verstockt},
  title     = {Multi-Modal Hit Detection and Positional Analysis in Padel Competitions},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2024},
  pages     = {3306--3314},
  doi       = {10.1109/CVPRW63382.2024.00335}
}
```

Autor da replicação: Henrique Gerhardt (UFSM).
