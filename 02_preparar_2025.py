"""
Etapa 2 - Dados de 2025 (questionário e notas SEM chave em comum)
- resultados    : nota média real por UF (quem fez as 4 provas + redação válida)
- participantes : amostra de 15% dos perfis/questionários, para aplicar o modelo
                  treinado em 2023 e comparar média PREVISTA x REAL por UF.
Uso: python 02_preparar_2025.py resultados   |   python 02_preparar_2025.py participantes
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
D25 = BASE.parent / "microdados_enem_2025" / "microdados_enem_2025" / "DADOS"
OUT = BASE / "dados"
OUT.mkdir(exist_ok=True)
ETAPAS = sys.argv[1:] or ["resultados", "participantes"]
rng = np.random.default_rng(42)

NOTAS = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
PRES = ["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT", "TP_STATUS_REDACAO"]
PERFIL = ["TP_FAIXA_ETARIA", "TP_SEXO", "TP_ESTADO_CIVIL", "TP_COR_RACA", "TP_NACIONALIDADE",
          "TP_ST_CONCLUSAO", "IN_TREINEIRO", "SG_UF_PROVA"] + [f"Q{i:03d}" for i in range(1, 24)]


def resultados():
    somas, amostra = [], []
    for b in pd.read_csv(D25 / "RESULTADOS_2025.csv", sep=";", encoding="latin1",
                         usecols=["SG_UF_PROVA"] + NOTAS + PRES, dtype=str, chunksize=400_000):
        b = b[(b[PRES] == "1").all(axis=1)].copy()
        b["NOTA_MEDIA"] = b[NOTAS].apply(pd.to_numeric).mean(axis=1)
        somas.append(b.groupby("SG_UF_PROVA")["NOTA_MEDIA"].agg(["sum", "count"]))
        amostra.append(b.loc[rng.random(len(b)) < 0.15, ["SG_UF_PROVA", "NOTA_MEDIA"]])
        print("resultados: bloco ok", flush=True)
    uf = pd.concat(somas).groupby(level=0).sum()
    uf["media_real"] = uf["sum"] / uf["count"]
    uf[["count", "media_real"]].to_csv(OUT / "enem2025_media_real_por_uf.csv")
    pd.concat(amostra).round(2).to_csv(OUT / "enem2025_notas_amostra.csv.gz", index=False, compression="gzip")


def participantes():
    am = []
    for b in pd.read_csv(D25 / "PARTICIPANTES_2025.csv", sep=";", encoding="latin1",
                         usecols=PERFIL, dtype=str, chunksize=400_000):
        am.append(b[rng.random(len(b)) < 0.15])
        print("participantes: bloco ok", flush=True)
    pd.concat(am).to_csv(OUT / "enem2025_participantes_amostra.csv.gz", index=False, compression="gzip")


for etapa in ETAPAS:
    {"resultados": resultados, "participantes": participantes}[etapa]()
print("FIM")
