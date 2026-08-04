# Summary — logs/ts/cmapss_explain

306 result rows.

### stage: explain

```
method                           dataset          variant                           loc_auroc          prec_at_k       deletion_auc  max_residual_nats mean_residual_nats loc_auroc[decouple]  loc_auroc[desync]   loc_auroc[drift]  seeds
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
PC Shapley (exact conditionals)  cmapss:FD001     subset-FD001                  0.8804±0.0063      0.6897±0.0255      0.2542±0.0154                  —                  —      0.4846±0.0339      0.6754±0.0384      0.9899±0.0052      3
PC conditional (exact)           cmapss:FD001     subset-FD001                  0.8774±0.0073      0.6844±0.0168      0.2490±0.0147                  —                  —      0.4884±0.0313      0.6851±0.0362      0.9770±0.0097      3
Gaussian conditional (exact)     cmapss:FD001     subset-FD001                  0.8593±0.0080      0.6258±0.0075      0.4576±0.0943                  —                  —      0.4715±0.0378      0.7187±0.0228      0.9442±0.0025      3
PC marginal (exact)              cmapss:FD001     subset-FD001                  0.8576±0.0034      0.6067±0.0150      0.2590±0.0169                  —                  —      0.4797±0.0416      0.5099±0.0318      0.9800±0.0064      3
z-score (per channel)            cmapss:FD001     subset-FD001                  0.8219±0.0085      0.5336±0.0073      0.5096±0.0433                  —                  —      0.4838±0.0409      0.4990±0.0231      0.9249±0.0005      3
PC structural (exact)            cmapss:FD001     subset-FD001                  0.7468±0.0139      0.4938±0.0116      0.7005±0.0153                  —                  —      0.5434±0.0359      0.7561±0.0451      0.8086±0.0283      3
AE reconstruction (per channel)  cmapss:FD001     subset-FD001                  0.7326±0.0119      0.3703±0.0285      0.3081±0.0774                  —                  —      0.4860±0.0037      0.5526±0.0315      0.8244±0.0042      3
AE sampling-SHAP (32/ch)         cmapss:FD001     subset-FD001                  0.6658±0.0137      0.3033±0.0267      0.2857±0.0888                  —                  —      0.4925±0.0109      0.4748±0.0037      0.7468±0.0070      3
Gaussian conditional (exact)     cmapss:FD004     subset-FD004                  0.6266±0.1954      0.3304±0.2843      0.1536±0.1701                  —                  —      0.6292±0.2315      0.6135±0.2080      0.5997±0.1413      3
AE reconstruction (per channel)  cmapss:FD002     subset-FD002                  0.6198±0.1957      0.3121±0.2803      0.4168±0.3541                  —                  —      0.6340±0.2041      0.6135±0.1886      0.6080±0.1779      3
PC conditional (exact)           cmapss:FD004     subset-FD004                  0.6182±0.1922      0.2938±0.2525      0.0733±0.0441                  —                  —      0.5967±0.1507      0.5493±0.1347      0.6375±0.2029      3
AE reconstruction (per channel)  cmapss:FD004     subset-FD004                  0.6124±0.1968      0.3187±0.2843      0.1580±0.1762                  —                  —      0.6070±0.2207      0.5862±0.2047      0.5945±0.1711      3
PC conditional (exact)           cmapss:FD002     subset-FD002                  0.6123±0.2058      0.2920±0.2407      0.0621±0.0236                  —                  —      0.6150±0.1440      0.4465±0.2434      0.6372±0.2099      3
PC structural (exact)            cmapss:FD002     subset-FD002                  0.6049±0.1118      0.2716±0.1272      0.2180±0.2264                  —                  —      0.6492±0.1901      0.7139±0.2582      0.5497±0.1180      3
Gaussian conditional (exact)     cmapss:FD002     subset-FD002                  0.5949±0.1747      0.3087±0.2828      0.1359±0.0722                  —                  —      0.6585±0.1995      0.5967±0.1996      0.5553±0.1208      3
PC Shapley (exact conditionals)  cmapss:FD002     subset-FD002                  0.5853±0.2306      0.2809±0.2375      0.0630±0.0247                  —                  —      0.6174±0.1350      0.3106±0.3390      0.6395±0.2098      3
PC Shapley (exact conditionals)  cmapss:FD004     subset-FD004                  0.5841±0.2228      0.2869±0.2569      0.0734±0.0440                  —                  —      0.6059±0.1408      0.2916±0.3342      0.6384±0.2165      3
PC structural (exact)            cmapss:FD004     subset-FD004                  0.5809±0.1029      0.2441±0.1308      0.1413±0.1430                  —                  —      0.5836±0.1769      0.7870±0.0620      0.5190±0.1452      3
z-score (per channel)            cmapss:FD004     subset-FD004                  0.5735±0.2089      0.2735±0.2342      0.0566±0.0174                  —                  —      0.5297±0.0018      0.1695±0.2518      0.6652±0.2463      3
PC marginal (exact)              cmapss:FD002     subset-FD002                  0.5724±0.2291      0.2767±0.2417      0.1057±0.0878                  —                  —      0.5193±0.0391      0.1907±0.2393      0.6659±0.2520      3
z-score (per channel)            cmapss:FD002     subset-FD002                  0.5702±0.2288      0.2713±0.2459      0.0768±0.0200                  —                  —      0.4776±0.0489      0.1738±0.2547      0.6667±0.2456      3
PC marginal (exact)              cmapss:FD004     subset-FD004                  0.5658±0.1977      0.2664±0.2263      0.1234±0.1245                  —                  —      0.5204±0.0166      0.2064±0.2502      0.6536±0.2443      3
AE sampling-SHAP (32/ch)         cmapss:FD004     subset-FD004                  0.4490±0.0382      0.0960±0.0422      0.1777±0.2049                  —                  —      0.4561±0.0934      0.2906±0.0199      0.4876±0.0361      3
AE sampling-SHAP (32/ch)         cmapss:FD002     subset-FD002                  0.4444±0.0077      0.1056±0.0089      0.1935±0.1940                  —                  —      0.3623±0.1186      0.2381±0.1106      0.4936±0.0237      3
PC chain-rule completeness       cmapss:FD004     subset-FD004                              —                  —                  —   2.566e+12±2e+12    2.400e+11±2e+11                   —                  —                  —      3
PC chain-rule completeness       cmapss:FD002     subset-FD002                              —                  —                  —   1.466e+12±2e+12    1.208e+11±2e+11                   —                  —                  —      3
PC chain-rule completeness       cmapss:FD001     subset-FD001                              —                  —                  —   1.068e-04±1e-04    5.682e-06±3e-06                   —                  —                  —      3
```
