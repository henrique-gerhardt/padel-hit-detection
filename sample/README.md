# sample/

Arquivos derivados usados pelo notebook, para que ele não precise converter as anotações nem recalibrar a quadra a
cada execução. Os vídeos não estão aqui: o notebook os baixa da fonte oficial do dataset.

## `annotations/hits.csv`

Os 319 golpes com jogador anotado do dataset de Decorte et al. (CVPRW 2024), convertidos para o formato usado pelo
código:

| Coluna | Conteúdo |
|---|---|
| `hit_id` | `<rally>_hNNN`: posição do golpe no rally, pela ordem dos onsets de áudio de `metadata/hits.csv`. `<rally>_aNNN_x`: os 9 rótulos sem onset de áudio correspondente |
| `video_id` | rally (`rallies/<video_id>.mp4` no dataset) |
| `frame` | frame do contato com a bola, a 25 fps |
| `gt_slot` | jogador que bateu, como posição vista da câmera: `far_left`, `far_right`, `near_left` ou `near_right` |
| `rally_id` | igual a `video_id` (um vídeo por rally) |

**Origem:** `metadata/hits.csv` (onsets de áudio dos 2.377 golpes) e `metadata/hit_assignments.xlsx` (319 golpes
com jogador), do [compartilhamento público do dataset](https://cloud.ilabt.imec.be/index.php/s/TFimLDWno6W9ED3).

**Conversão** (`padelrep.annotations.convert_decorte_to_canonical(..., fps=25)`):

- O frame é `round(timestamp × 25)`, com o `timestamp` do `hit_assignments.xlsx`: o frame visível do contato, que é
  o instante que o artigo usa para a atribuição. O onset de áudio vem, em mediana, 0,12 s antes.
- Os dois arquivos não têm chave comum. Eles são casados por alinhamento temporal que preserva a ordem (programação
  dinâmica, tolerância de 0,5 s). O casamento só define o `hit_id`. 310 rótulos casam com um onset, e os 9 restantes
  ficam com o sufixo `_x`.
- O rótulo segue o esquema do artigo, visto da câmera: `t1p1 → far_left`, `t1p2 → far_right`,
  `t2p1 → near_left` e `t2p2 → near_right`.

| Slot | Golpes |
|---|---|
| `far_left` | 89 |
| `far_right` | 69 |
| `near_left` | 67 |
| `near_right` | 94 |

## `calibration/`

Homografia imagem → quadra (em metros) de cada um dos 16 rallies rotulados. Ela é usada só para dar a cada jogador
o seu slot (lado da rede e metade esquerda/direita) e para descartar quem está fora da quadra.

Sistema de coordenadas da quadra, visto de cima a partir da câmera: `x` vai de 0 (lateral esquerda) a 10 m, e `y`
vai de 0 (fundo longe da câmera) a 20 m, com a rede em `y = 10`.

Os rallies vêm de duas câmeras fixas, e cada câmera tem uma única homografia, copiada para os seus vídeos:

| Câmera | Torneio | Vídeos | Erro de reprojeção médio / máximo |
|---|---|---|---|
| `vigo` | WPT Vigo Open | `20230528_VIGO_00` a `_08` | 0,044 m / 0,082 m |
| `denmark` | Cupra Danish Padel Open 2023 | `20230528_VIGO_09` a `_12` e `_14` a `_16` | 0,042 m / 0,057 m |

Cada `<video_id>.json` traz a matriz `H_img2court`, os 10 pontos de referência na imagem (`points`, em px), o
vídeo de origem da calibração e o erro de reprojeção.

**Como foram obtidas** (`padelrep.calibration`, sem pontos clicados):

1. A imagem de fundo é a mediana de 8 frames de cada rally da câmera (`background_videos` em `cameras.json`). Os
   jogadores somem e ficam o piso e as linhas. A câmera `denmark` usa também o `VIGO_13`, que não tem golpes
   rotulados.
2. Retas são ajustadas às linhas de saque e à linha central (pixels brancos) e às bordas do piso azul. Cada reta é
   procurada numa faixa da imagem (`bands` em `cameras.json`), definida à mão por câmera e conferida visualmente.
3. As interseções das retas dão 10 pontos: 4 cantos, 4 pontas das linhas de saque e os 2 "T". A homografia é
   ajustada com RANSAC.

Para refazer uma câmera a partir dos vídeos:

```python
import json
from padelrep.calibration import median_background, suggest
from padelrep.court import CourtModel

cam = json.load(open("sample/calibration/cameras.json"))["vigo"]
bg = median_background([f"data/rallies/{v}.mp4" for v in cam["background_videos"]])
points, _, _ = suggest(bg, cam["bands"])
assert points == cam["points_px"]
court = CourtModel.from_named_points(points)
```

## Créditos

Os vídeos e as anotações originais são de Decorte et al. (imec / Universidade de Gent). Os arquivos desta pasta são
derivados deles, e qualquer uso deve citar o artigo:

> R. Decorte, M. Paré, J. Vanhaeverbeke, J. Taelman, M. Slembrouck e S. Verstockt. Multi-Modal Hit Detection and
> Positional Analysis in Padel Competitions. *CVPR Workshops*, 2024, pp. 3306–3314.
> DOI [10.1109/CVPRW63382.2024.00335](https://doi.org/10.1109/CVPRW63382.2024.00335).
