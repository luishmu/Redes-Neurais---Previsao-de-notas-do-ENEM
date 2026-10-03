"""
Etapa 1 - Preparação dos dados (ENEM 2023)
Lê o CSV direto do zip em blocos (pouca RAM), mantém só quem fez as 4 provas
objetivas e teve redação válida, e salva uma amostra aleatória de 15%.
Por que 2023? A partir de 2024 o INEP separou questionário e notas em arquivos
sem chave em comum (LGPD), então não dá para ligar perfil -> nota.
"""
import zipfile, pandas as pd, numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
ZIP = BASE.parent / "microdados_enem_2023.zip"
CSV_NO_ZIP = "microdados_enem_2023/DADOS/MICRODADOS_ENEM_2023.csv"
SAIDA = BASE / "dados" / "enem2023_amostra.csv.gz"
FRACAO = 0.15
SEED = 42

PERFIL = ["TP_FAIXA_ETARIA", "TP_SEXO", "TP_ESTADO_CIVIL", "TP_COR_RACA",
          "TP_NACIONALIDADE", "TP_ST_CONCLUSAO", "TP_ESCOLA", "IN_TREINEIRO",
          "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC", "SG_UF_PROVA", "TP_LINGUA",
          "CO_MUNICIPIO_PROVA", "TP_ANO_CONCLUIU"]
QUEST = [f"Q{i:03d}" for i in range(1, 26)]
NOTAS = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
FILTRO = ["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT", "TP_STATUS_REDACAO"]

SAIDA.parent.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)
partes, total, validos = [], 0, 0
with zipfile.ZipFile(ZIP) as z, z.open(CSV_NO_ZIP) as f:
    for bloco in pd.read_csv(f, sep=";", encoding="latin1", usecols=PERFIL + QUEST + NOTAS + FILTRO,
                             dtype=str, chunksize=300_000):
        total += len(bloco)
        ok = (bloco[FILTRO] == "1").all(axis=1)
        bloco = bloco[ok].drop(columns=FILTRO)
        validos += len(bloco)
        partes.append(bloco[rng.random(len(bloco)) < FRACAO])
        print(f"lidas {total:,} | válidas {validos:,}", flush=True)

df = pd.concat(partes, ignore_index=True)
for c in NOTAS:
    df[c] = pd.to_numeric(df[c])
df["NOTA_MEDIA"] = df[NOTAS].mean(axis=1).round(2)
tmp = SAIDA.with_suffix(".tmp")
df.to_csv(tmp, index=False, compression="gzip")
tmp.replace(SAIDA)  # grava em arquivo temporário e renomeia: evita arquivo corrompido se interromper
print(f"FIM: {total:,} inscritos, {validos:,} com as 5 notas, amostra = {len(df):,} -> {SAIDA}")
