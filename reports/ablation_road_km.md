# Road km feature (P7b U1)

## Parsing

Regex built from `config.MAIN_ROADS` + `ROAD_ALIASES` ("Jakal" = Kaliurang, "Paris" = Parangtritis). The road name must sit right before "km" ("Jl."/"jalan" optional; "Barat", "Timur", "Raya" allowed in between), so distances like "5 km dari UGM" never match. Comma or dot decimals; values outside 0.5–40 km are ignored. **109 of 1,180 listings (9.2%)** have a marker; km 2–22, median 9.

| Road | Listings |
|---|---|
| kaliurang | 66 |
| palagan | 10 |
| wates | 10 |
| godean | 7 |
| imogiri | 5 |
| parangtritis | 4 |
| bantul | 4 |
| wonosari | 2 |
| solo | 1 |

| Kabupaten | With marker | Listings |
|---|---|---|
| Bantul | 20 | 172 |
| Gunungkidul | 0 | 6 |
| Kota Yogyakarta | 1 | 122 |
| Kulon Progo | 0 | 26 |
| Sleman | 88 | 854 |

### 10 random examples (seed 42)

| Description snippet | Parsed |
|---|---|
| ...Kasongan di Jl Bantul Km 7 Sewon Ban... | bantul km 7 |
| ...UGM PERUMAHAN JAKAL KM 6 | kaliurang km 6 |
| ...alam Perumahan Jakal km 13 | kaliurang km 13 |
| Rumah di Jl Imogiri Barat Km 11 Dekat Kam... | imogiri km 11 |
| ...s Bangun Di Jl Godean Km 12 Harga 450... | godean km 12 |
| ...iap Huni di Jl Wates Km 9 Sedayu Ba... | wates km 9 |
| ...N UII DI JALAN KALIURANG KM 10 | kaliurang km 10 |
| ...mpus Uii Jalan Kaliurang Km 12 Cuma 800... | kaliurang km 12 |
| ...Perum Elite Jl.Wates Km9 | wates km 9 |
| Rumah Baru Jakal Km 8 Dekat UGM | kaliurang km 8 |

## Ablation

### Best model per family

| Variant | Model | Spatial MAPE | Spatial MdAPE | Spatial MAE | Random MAPE |
|---|---|---|---|---|---|
| without road km | Ridge alpha=10 | 28.91% ± 2.7 | 21.37% | Rp 490 jt | 27.24% |
| without road km | LightGBM leaves=7 min_child=10 | 25.62% ± 3.2 | 19.46% | Rp 487 jt | 24.23% |
| with road km | Ridge alpha=10 | 28.83% ± 2.6 | 20.95% | Rp 490 jt | 27.20% |
| with road km | LightGBM leaves=7 min_child=10 | 25.62% ± 3.2 | 18.97% | Rp 487 jt | 24.31% |

### Same LightGBM params in both variants

| Variant | Model | Spatial MAPE | Spatial MdAPE | Spatial MAE | Random MAPE |
|---|---|---|---|---|---|
| without road km | LightGBM leaves=7 min_child=10 | 25.62% | 19.46% | Rp 487 jt | 24.23% |
| with road km | LightGBM leaves=7 min_child=10 | 25.62% | 18.97% | Rp 487 jt | 24.31% |

### LightGBM error by subset (spatial CV, out-of-fold)

| Variant | Spatial MAPE, rows with marker | Spatial MAPE, other rows |
|---|---|---|
| without road km | 22.3% | 26.0% |
| with road km | 21.8% | 26.0% |

Rule (guide P7b U1): keep only if spatial-CV MAPE is equal or better. LightGBM: 25.62% without vs 25.62% with (fold std ± 3.2); MdAPE 19.46% → 18.97%; random MAPE 24.23% → 24.31%; on the listings with a marker 22.3% → 21.8%. Every difference is far inside the fold-to-fold spread.

**Decision: not kept** (`config.USE_ROAD_KM_FEATURES = False`). No measurable gain, and keeping it would add a road + km input to the app that helps under 10% of houses. The parser stays; `road_name`, `road_km`, `has_road_km` remain in `model_table.csv` for EDA.
