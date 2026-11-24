# Dataset AIMD BGDY 3x3 (LUMI + MN5) — decisiones y cifras

Actualizado: 2026-10-20 (v2: posiciones envueltas, últimos 50 pasos de metastable fuera, menos peso a 2750/3000 K).
Pipeline: notebook `BGDY_3x3_dataset_AIMD.ipynb` (parámetros en la sección 1).

## Datos de partida
- 31 outputs pw.x (QE 7.5), NVT Nosé–Hoover, C108B18 (126 átomos), celda hexagonal a = 35.54 Å, vacío 20 Å, PBE PAW, spin-polarizado.
- 18 trayectorias (output.N = reinicios de una misma trayectoria; case_a/b/c = independientes). Todas parten de la misma geometría. 11 en carpetas stable, 7 en metastable.
- dt = 40 a.u. = 1.9351 fs (500–2500 K); dt = 20 a.u. = 0.9676 fs (2750 y 3000 K, todas metastable). Comprobado con el tiempo impreso en los 31 outputs.

## Decisiones (v2)
- Reinicios unidos por número de paso. La cabecera de output.2+ imprime la geometría del input (t = 0); el primer SCF de cada reinicio usa la última ATOMIC_POSITIONS del output anterior → corregido (ase.io.read lo asigna mal). 1 duplicado eliminado (MN5 500 K).
- Descartados los primeros 0.5 ps de cada trayectoria (259 pasos con dt 1.935 fs, 517 con 0.968 fs). Mismo criterio LUMI/MN5; etiquetas por T fusionando HPC.
- Trayectorias metastable: se quitan además los últimos 50 pasos MD (7 × 50 = 350 configuraciones).
- Filtro |F| > 100 eV/Å: no elimina nada (máx 26.9 eV/Å).
- Posiciones: QE las da sin envolver (en el plano llegan a 1.32 en coordenadas fraccionarias; en z quedan entre 0.19 y 0.72). Se analizan sin envolver y se envuelven en la celda al exportar (365 de las 1000 configuraciones seleccionadas tenían átomos fuera).
- Descriptor: SOAP (r_cut 5 Å, n_max 6, l_max 4) promediado, estandarizado, PCA 36 comp. (99 %).
- FPS por T, N = 1000: cuota = mín + parte libre del reparto del FPS global × peso; mínimo 50 × peso. Peso 0.25 en 2750 y 3000 K (configuraciones muy desordenadas, menos relevantes).

## Cifras (v2)
- 28 983 SCF únicos → 22 939 configuraciones.
- n_eff total ≈ 819 (τ 20–95 fs).
- Selección FPS: 500 K 50 · 1000 K 50 · 1500 K 67 · 2000 K 197 · 2250 K 307 · 2500 K 295 · 2750 K 22 · 3000 K 12.

## Hallazgos
- LUMI y MN5 indistinguibles a igual T (2000 K: misma nube, E/átomo −236.26 eV).
- Las trayectorias de carpetas metastable (MN5 2250 K desde ≈0.7 ps, LUMI 2500 K, 2750 K) forman una región aparte en UMAP; 3000 K, otra. MN5 metastable 2500 K se parece a las estables. Guardar la etiqueta `regime`.
- 5 dinámicas metastable terminaron por SCF no convergido (magnetización total hasta ±2 μB en ese régimen).
- 2750 y 3000 K: muy pocos datos independientes (n_eff ≈ 25 y 8).
