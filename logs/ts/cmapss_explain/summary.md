# Summary — logs/ts/cmapss_explain

153 result rows.

### stage: explain

```
method                           dataset          variant                           loc_auroc          prec_at_k       deletion_auc  max_residual_nats mean_residual_nats loc_auroc[decouple]  loc_auroc[desync]   loc_auroc[drift]  seeds
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Gaussian conditional (exact)     cmapss:FD001     subset-FD001                  0.8807±0.0088      0.6510±0.0064      0.1160±0.0111                  —                  —      0.4732±0.0361      0.7217±0.0205      0.9741±0.0017      3
PC Shapley (exact conditionals)  cmapss:FD001     subset-FD001                  0.8664±0.0086      0.6322±0.0096      0.0672±0.0008                  —                  —      0.4831±0.0678      0.5495±0.0341      0.9761±0.0051      3
PC marginal (exact)              cmapss:FD002     subset-FD002                  0.8635±0.0106      0.6093±0.0158      0.0756±0.0026                  —                  —      0.4812±0.0434      0.4975±0.0109      0.9896±0.0007      3
PC marginal (exact)              cmapss:FD001     subset-FD001                  0.8630±0.0084      0.6205±0.0083      0.0672±0.0008                  —                  —      0.4822±0.0655      0.5298±0.0378      0.9713±0.0044      3
PC conditional (exact)           cmapss:FD001     subset-FD001                  0.8625±0.0084      0.6288±0.0119      0.0670±0.0009                  —                  —      0.4791±0.0675      0.5491±0.0359      0.9717±0.0052      3
z-score (per channel)            cmapss:FD002     subset-FD002                  0.8556±0.0022      0.5851±0.0228      0.0922±0.0049                  —                  —      0.4744±0.0645      0.5210±0.0213      0.9739±0.0014      3
PC conditional (exact)           cmapss:FD002     subset-FD002                  0.8533±0.0016      0.5178±0.0144      0.0647±0.0027                  —                  —      0.6168±0.0430      0.6217±0.0278      0.8926±0.0155      3
PC marginal (exact)              cmapss:FD004     subset-FD004                  0.8518±0.0171      0.5862±0.0144      0.0611±0.0108                  —                  —      0.4927±0.0311      0.4959±0.0187      0.9713±0.0023      3
PC Shapley (exact conditionals)  cmapss:FD002     subset-FD002                  0.8454±0.0019      0.4945±0.0223      0.0654±0.0027                  —                  —      0.6007±0.0419      0.6105±0.0298      0.8830±0.0128      3
z-score (per channel)            cmapss:FD004     subset-FD004                  0.8443±0.0201      0.5754±0.0175      0.0615±0.0070                  —                  —      0.5061±0.0417      0.4594±0.0063      0.9685±0.0014      3
PC conditional (exact)           cmapss:FD004     subset-FD004                  0.8434±0.0165      0.5359±0.0159      0.0588±0.0067                  —                  —      0.5755±0.0657      0.5569±0.0065      0.9112±0.0117      3
z-score (per channel)            cmapss:FD001     subset-FD001                  0.8425±0.0049      0.5570±0.0149      0.0694±0.0037                  —                  —      0.4866±0.0417      0.5047±0.0263      0.9490±0.0013      3
PC Shapley (exact conditionals)  cmapss:FD004     subset-FD004                  0.8345±0.0160      0.5082±0.0137      0.0571±0.0060                  —                  —      0.5659±0.0668      0.5487±0.0107      0.8992±0.0114      3
Gaussian conditional (exact)     cmapss:FD004     subset-FD004                  0.8279±0.0149      0.5012±0.0114      0.0663±0.0176                  —                  —      0.5919±0.0282      0.7338±0.0184      0.8408±0.0273      3
Gaussian conditional (exact)     cmapss:FD002     subset-FD002                  0.8143±0.0062      0.4861±0.0164      0.1284±0.0153                  —                  —      0.5981±0.0243      0.6812±0.0529      0.8340±0.0096      3
AE reconstruction (per channel)  cmapss:FD004     subset-FD004                  0.7660±0.0098      0.4478±0.0367      0.0993±0.0170                  —                  —      0.5197±0.0601      0.5976±0.0366      0.8242±0.0396      3
AE sampling-SHAP (32/ch)         cmapss:FD002     subset-FD002                  0.7583±0.0102      0.4489±0.0294      0.0984±0.0074                  —                  —      0.4601±0.0212      0.5115±0.0374      0.8721±0.0050      3
AE reconstruction (per channel)  cmapss:FD002     subset-FD002                  0.7575±0.0101      0.4197±0.0092      0.1871±0.0418                  —                  —      0.4814±0.0101      0.5690±0.0445      0.8160±0.0152      3
AE sampling-SHAP (32/ch)         cmapss:FD004     subset-FD004                  0.7548±0.0178      0.4121±0.0114      0.0699±0.0101                  —                  —      0.4962±0.0303      0.5048±0.0229      0.8448±0.0146      3
AE reconstruction (per channel)  cmapss:FD001     subset-FD001                  0.7310±0.0237      0.3817±0.0451      0.1029±0.0118                  —                  —      0.4927±0.0206      0.5595±0.0349      0.8260±0.0216      3
PC structural (exact)            cmapss:FD002     subset-FD002                  0.6987±0.0216      0.3684±0.0302      0.0538±0.0016                  —                  —      0.6925±0.0455      0.6477±0.0406      0.6759±0.0321      3
AE sampling-SHAP (32/ch)         cmapss:FD001     subset-FD001                  0.6907±0.0196      0.3389±0.0312      0.0893±0.0072                  —                  —      0.4871±0.0311      0.4870±0.0129      0.7983±0.0331      3
PC structural (exact)            cmapss:FD004     subset-FD004                  0.6674±0.0261      0.3561±0.0384      0.0638±0.0099                  —                  —      0.6150±0.0463      0.6169±0.0466      0.6257±0.0189      3
PC structural (exact)            cmapss:FD001     subset-FD001                  0.6019±0.0181      0.3041±0.0271      0.0728±0.0028                  —                  —      0.4798±0.0297      0.6108±0.0132      0.6329±0.0096      3
PC chain-rule completeness       cmapss:FD004     subset-FD004                              —                  —                  —   8.240e-04±1e-03    2.935e-05±4e-05                   —                  —                  —      3
PC chain-rule completeness       cmapss:FD002     subset-FD002                              —                  —                  —      0.0208±0.0090   5.433e-04±2e-04                   —                  —                  —      3
PC chain-rule completeness       cmapss:FD001     subset-FD001                              —                  —                  —   1.831e-04±3e-04    6.795e-06±5e-06                   —                  —                  —      3
```
