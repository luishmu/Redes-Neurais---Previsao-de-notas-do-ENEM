"""
Etapa 2b - Contexto municipal (dados externos, ligados pelo código IBGE do município)

Fontes (repositórios públicos no GitHub, baixados via raw.githubusercontent.com):
- Atlas do Desenvolvimento Humano no Brasil (PNUD/IPEA/FJP), Censo 2010
  https://github.com/mauriciocramos/IDHM  -> municipal.csv
- Tabela de municípios (coordenadas, capital, população estimada 2021)
  https://github.com/mapaslivres/municipios-br -> tabelas/municipios.csv

Saída: dados/contexto_municipal.csv (uma linha por município, chave CO_MUNICIPIO = código IBGE 7 dígitos)
"""
import io
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
SAIDA = BASE / "dados" / "contexto_municipal.csv"
URL_ATLAS = "https://raw.githubusercontent.com/mauriciocramos/IDHM/master/municipal.csv"
URL_MUN = "https://raw.githubusercontent.com/mapaslivres/municipios-br/main/tabelas/municipios.csv"


SAIDA.parent.mkdir(exist_ok=True)


def baixar(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return io.BytesIO(r.read())


atlas = pd.read_csv(baixar(URL_ATLAS), sep=";", decimal=",")
atlas = atlas[atlas.ANO == 2010]
ctx = pd.DataFrame({
    "CO_MUNICIPIO": atlas.Codmun7.astype(int),
    "IDHM": atlas.IDHM, "IDHM_EDUC": atlas.IDHM_E, "IDHM_RENDA": atlas.IDHM_R, "IDHM_LONGEV": atlas.IDHM_L,
    "GINI": atlas.GINI,
    "RENDA_PER_CAPITA": atlas.RDPC,
    "PCT_EXTREM_POBRES": atlas.PIND,
    "PCT_ANALFAB_15M": atlas.T_ANALF15M,
    "PCT_MEDIO_18M": atlas.T_MED18M,
    "PCT_SUPERIOR_25M": atlas.T_SUPER25M,
    "TAXA_URBANIZACAO": 100 * atlas.pesourb / atlas.pesotot,
})

mun = pd.read_csv(baixar(URL_MUN))
mun = mun.rename(columns={"municipio": "CO_MUNICIPIO"})
mun["CAPITAL"] = mun.is_capital.fillna(False).astype(bool).astype(int)
cap = mun[mun.CAPITAL == 1][["uf_code", "lat", "lon"]].rename(columns={"lat": "lat_c", "lon": "lon_c"})
mun = mun.merge(cap, on="uf_code", how="left")
# distância (km) até a capital do estado — fórmula de haversine
la1, lo1, la2, lo2 = map(np.radians, [mun.lat, mun.lon, mun.lat_c, mun.lon_c])
h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
mun["DIST_CAPITAL_KM"] = 6371 * 2 * np.arcsin(np.sqrt(h))
mun["LOG_POPULACAO"] = np.log10(mun.pop_21)

mun = mun.rename(columns={"name": "NOME_MUNICIPIO", "uf_code": "UF"})
ctx = mun[["CO_MUNICIPIO", "NOME_MUNICIPIO", "UF", "CAPITAL", "DIST_CAPITAL_KM", "LOG_POPULACAO"]].merge(ctx, on="CO_MUNICIPIO", how="left")
ctx.round(4).to_csv(SAIDA, index=False)
print(f"{len(ctx):,} municípios | sem dado do Atlas 2010 (criados depois): {ctx.IDHM.isna().sum()} -> {SAIDA}")
