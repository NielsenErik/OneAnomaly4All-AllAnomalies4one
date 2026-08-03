# Development Idea and Implementation

## Intro
The core Idea of the project is to develop a multimodal probabilistic circuit able to do anomaly detection on several types of data, e.g. images, text as hallucinations, tabular data and so on, to find a way to have a unified framework for anomaly detection.
It has to be able to compute exact density estimator, and if needed exact marginals, beacuse the point is to maintain reliability and efficiency.
there are 2 main idea, I want to evaluate both for the purpose of novelty.
1. haveing encoders for different data types that crate a latent variable over wich do exact estimations with PC circuits, this is the easier way
2. Avoid encoders to do exact estimation over the input data, therefore ideaddly havting a vtree structure, or a general big mixture of PC that given an input type it "attach" to the correct circuit to do anomaly detection, this idea aims to do strong transfer learnign as well, and it is more complex.

# development
first here I add several references
https://arxiv.org/pdf/2511.11346
https://openreview.net/pdf?id=dLoGOry8sJR
https://arxiv.org/pdf/1807.09306
https://yoojungchoi.github.io/files/ProbCirc20.pdf
https://ojs.aaai.org/index.php/AAAI/article/view/29675
https://openreview.net/pdf?id=vU8EPo44Gj
https://ojs.aaai.org/index.php/AAAI/article/view/34100

It is important, crucial and not avoidable that the implekentation mantain the characteristics of circuits, which are
1. Smoothness (also called completeness) — every sum node's inputs are defined over the same set of variables (the same scope). Together with decomposability, this is what makes computing marginals and partition functions tractable.
2. Decomposability — every product node's inputs are defined over disjoint sets of variables, so their scopes partition the product's scope. This is the other half of what enables tractable marginalization and integration.
3. Determinism (also called selectivity) — for any complete input assignment, at most one child of each sum node outputs a nonzero value. This enables tractable MAP / MPE (most probable explanation) queries.
4. Structured decomposability — a stronger form of decomposability where all product nodes decompose their scopes consistently according to a shared hierarchy of variables called a vtree. This unlocks more advanced operations such as computing probabilities of logical events, circuit multiplication, and certain marginal-MAP queries.

Permit the usage of gaussian mixture and  sum of square PC https://ojs.aaai.org/index.php/AAAI/article/view/34100 for higher expressivsness

Create one singular py file that develop the probabilistc circuits components and class. and one additional .py file that implemet the 2 directions







