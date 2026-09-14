# Correction campaign

Common sensor rows: 908. Exact calendar t+3, threshold 0.5.

Standard deviation across seeds measures training variability, not sampling uncertainty.

```
model      variant  seed  sensor_available  excluded_from_common   N  WetFrac  Accuracy  ROC-AUC    WetF1    DryF1  DryRecall  DryPrecision
 lstm      control    42               956                    48 908  0.79185  0.903084 0.949099 0.939643 0.754190   0.714286      0.798817
 rgcn      control    42               908                     0 908  0.79185  0.947137 0.984017 0.966667 0.872340   0.867725      0.877005
 lstm availability    42               956                    48 908  0.79185  0.951542 0.981367 0.968883 0.890547   0.947090      0.840376
 rgcn availability    42               908                     0 908  0.79185  0.961454 0.985636 0.975089 0.914842   0.994709      0.846847
 lstm      control    43               956                    48 908  0.79185  0.940529 0.988741 0.962963 0.849162   0.804233      0.899408
 rgcn      control    43               908                     0 908  0.79185  0.969163 0.987711 0.980198 0.930348   0.989418      0.877934
 lstm availability    43               956                    48 908  0.79185  0.958150 0.985900 0.973464 0.901042   0.915344      0.887179
 rgcn availability    43               908                     0 908  0.79185  0.962555 0.986342 0.975852 0.916667   0.989418      0.853881
 lstm      control    44               956                    48 908  0.79185  0.947137 0.983818 0.967033 0.866667   0.825397      0.912281
 rgcn      control    44               908                     0 908  0.79185  0.970264 0.986092 0.980892 0.933002   0.994709      0.878505
 lstm availability    44               956                    48 908  0.79185  0.943833 0.975532 0.964261 0.868895   0.894180      0.845000
 rgcn availability    44               908                     0 908  0.79185  0.966960 0.985091 0.978784 0.925373   0.984127      0.873239
```

```
                    Accuracy             ROC-AUC               WetF1           DryPrecision           DryRecall               DryF1          
                        mean       std      mean       std      mean       std         mean       std      mean       std      mean       std
model variant                                                                                                                                
lstm  availability  0.951175  0.007166  0.980933  0.005198  0.968869  0.004602     0.857518  0.025791  0.918871  0.026631  0.886828  0.016393
      control       0.930250  0.023757  0.973886  0.021607  0.956546  0.014779     0.870169  0.062127  0.781305  0.058997  0.823340  0.060522
rgcn  availability  0.963656  0.002914  0.985690  0.000627  0.976575  0.001950     0.857989  0.013667  0.989418  0.005291  0.918961  0.005628
      control       0.962188  0.013047  0.985940  0.001852  0.975919  0.008020     0.877815  0.000757  0.950617  0.071836  0.911897  0.034283
```
