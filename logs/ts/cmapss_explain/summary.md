# Summary — logs/ts/cmapss_explain

153 result rows.

### stage: explain

```
method                           dataset          variant                           loc_auroc          prec_at_k       deletion_auc  max_residual_nats mean_residual_nats loc_auroc[decouple]  loc_auroc[desync]   loc_auroc[drift]  seeds
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Gaussian conditional (exact)     cmapss:FD001     subset-FD001                  0.8807±0.0088      0.6510±0.0064      0.1158±0.0123                  —                  —      0.4732±0.0361      0.7217±0.0205      0.9741±0.0017      3
PC Shapley (exact conditionals)  cmapss:FD001     subset-FD001                  0.8668±0.0096      0.6320±0.0065      0.0671±0.0011                  —                  —      0.4823±0.0666      0.5520±0.0445      0.9765±0.0064      3
PC marginal (exact)              cmapss:FD001     subset-FD001                  0.8628±0.0085      0.6208±0.0113      0.0671±0.0011                  —                  —      0.4809±0.0653      0.5294±0.0358      0.9716±0.0050      3
PC marginal (exact)              cmapss:FD002     subset-FD002                  0.8625±0.0104      0.6113±0.0063      0.0752±0.0056                  —                  —      0.4825±0.0437      0.4946±0.0127      0.9897±0.0002      3
PC conditional (exact)           cmapss:FD001     subset-FD001                  0.8624±0.0082      0.6258±0.0105      0.0670±0.0011                  —                  —      0.4779±0.0660      0.5498±0.0479      0.9717±0.0047      3
PC conditional (exact)           cmapss:FD002     subset-FD002                  0.8566±0.0019      0.5380±0.0135      0.0641±0.0028                  —                  —      0.6282±0.0419      0.6132±0.0203      0.8998±0.0098      3
z-score (per channel)            cmapss:FD002     subset-FD002                  0.8556±0.0022      0.5851±0.0228      0.0919±0.0032                  —                  —      0.4744±0.0645      0.5210±0.0213      0.9739±0.0014      3
PC marginal (exact)              cmapss:FD004     subset-FD004                  0.8509±0.0171      0.5843±0.0130      0.0620±0.0054                  —                  —      0.4926±0.0320      0.4957±0.0195      0.9700±0.0027      3
PC Shapley (exact conditionals)  cmapss:FD002     subset-FD002                  0.8458±0.0010      0.4997±0.0204      0.0652±0.0035                  —                  —      0.6023±0.0430      0.5950±0.0234      0.8912±0.0094      3
z-score (per channel)            cmapss:FD004     subset-FD004                  0.8443±0.0201      0.5754±0.0175      0.0604±0.0022                  —                  —      0.5061±0.0417      0.4594±0.0063      0.9685±0.0014      3
PC conditional (exact)           cmapss:FD004     subset-FD004                  0.8434±0.0122      0.5365±0.0089      0.0606±0.0053                  —                  —      0.5642±0.0471      0.5379±0.0298      0.9204±0.0211      3
z-score (per channel)            cmapss:FD001     subset-FD001                  0.8425±0.0049      0.5570±0.0149      0.0693±0.0036                  —                  —      0.4866±0.0417      0.5047±0.0263      0.9490±0.0013      3
PC Shapley (exact conditionals)  cmapss:FD004     subset-FD004                  0.8340±0.0100      0.5086±0.0110      0.0612±0.0059                  —                  —      0.5589±0.0504      0.5313±0.0270      0.9067±0.0217      3
Gaussian conditional (exact)     cmapss:FD004     subset-FD004                  0.8279±0.0149      0.5012±0.0114      0.0639±0.0048                  —                  —      0.5919±0.0282      0.7338±0.0184      0.8408±0.0273      3
Gaussian conditional (exact)     cmapss:FD002     subset-FD002                  0.8143±0.0062      0.4861±0.0164      0.1239±0.0207                  —                  —      0.5981±0.0243      0.6812±0.0529      0.8340±0.0096      3
AE sampling-SHAP (32/ch)         cmapss:FD002     subset-FD002                  0.7576±0.0130      0.4516±0.0321      0.1051±0.0295                  —                  —      0.4690±0.0297      0.5126±0.0391      0.8661±0.0192      3
AE reconstruction (per channel)  cmapss:FD004     subset-FD004                  0.7558±0.0041      0.4205±0.0233      0.1158±0.0211                  —                  —      0.4999±0.0572      0.5978±0.0090      0.7862±0.0399      3
AE reconstruction (per channel)  cmapss:FD002     subset-FD002                  0.7553±0.0103      0.4096±0.0058      0.1769±0.0581                  —                  —      0.4864±0.0214      0.5663±0.0489      0.8085±0.0230      3
AE sampling-SHAP (32/ch)         cmapss:FD004     subset-FD004                  0.7537±0.0125      0.4142±0.0144      0.0725±0.0147                  —                  —      0.4930±0.0374      0.4984±0.0373      0.8326±0.0176      3
AE reconstruction (per channel)  cmapss:FD001     subset-FD001                  0.7418±0.0233      0.3861±0.0467      0.1085±0.0245                  —                  —      0.4947±0.0266      0.5753±0.0399      0.8513±0.0028      3
PC structural (exact)            cmapss:FD002     subset-FD002                  0.6893±0.0143      0.3623±0.0183      0.0525±0.0013                  —                  —      0.6980±0.0545      0.6316±0.0111      0.6645±0.0229      3
AE sampling-SHAP (32/ch)         cmapss:FD001     subset-FD001                  0.6864±0.0054      0.3231±0.0209      0.0924±0.0049                  —                  —      0.4732±0.0139      0.4931±0.0021      0.7929±0.0221      3
PC structural (exact)            cmapss:FD004     subset-FD004                  0.6857±0.0164      0.3507±0.0195      0.0937±0.0247                  —                  —      0.5905±0.0539      0.6092±0.0355      0.7012±0.0167      3
PC structural (exact)            cmapss:FD001     subset-FD001                  0.6087±0.0202      0.3212±0.0186      0.0716±0.0076                  —                  —      0.4994±0.0118      0.6077±0.0338      0.6248±0.0324      3
PC chain-rule completeness       cmapss:FD004     subset-FD004                              —                  —                  —      0.0014±0.0022   4.231e-05±7e-05                   —                  —                  —      3
PC chain-rule completeness       cmapss:FD002     subset-FD002                              —                  —                  —      0.0030±0.0042   6.383e-05±9e-05                   —                  —                  —      3
PC chain-rule completeness       cmapss:FD001     subset-FD001                              —                  —                  —      0.0208±0.0361   3.314e-04±6e-04                   —                  —                  —      3
```
