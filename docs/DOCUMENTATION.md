# Documentation Exhaustive - Moteur d'Évaluation Formelle par Expression Affine

**Auteur:** Guillaume BERTHELOT  
**Version:** 1.0  
**Date:** 2025

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Fondements Théoriques](#2-fondements-théoriques)
3. [Architecture Logicielle](#3-architecture-logicielle)
4. [Guide d'Installation](#4-guide-dinstallation)
5. [Guide d'Utilisation](#5-guide-dutilisation)
6. [API de Référence](#6-api-de-référence)
7. [Exemples Pratiques](#7-exemples-pratiques)
8. [Tests et Validation](#8-tests-et-validation)
9. [Optimisation et Performance](#9-optimisation-et-performance)
10. [Limitations et Extensions](#10-limitations-et-extensions)
11. [FAQ](#11-faq)
12. [Références](#12-références)

---

## 1. Introduction

### 1.1 Contexte

Les réseaux de neurones profonds sont vulnérables aux **attaques adversariales** : de petites perturbations imperceptibles de l'entrée peuvent conduire à des classifications erronées. Cette vulnérabilité pose des problèmes critiques dans les applications sensibles (voitures autonomes, diagnostic médical, sécurité).

### 1.2 Problématique

Les approches empiriques (tests Monte Carlo, attaques adversariales) ne garantissent pas l'exhaustivité :
- ❌ Impossibilité de tester tous les cas
- ❌ Pas de garantie mathématique
- ❌ Coût computationnel élevé

### 1.3 Solution Proposée

Ce projet implémente une **vérification formelle** par propagation d'expressions affines qui :
- ✅ Garantit mathématiquement les bornes de sortie
- ✅ Couvre tous les cas possibles d'entrées bruitées
- ✅ Identifie les classes certifiées robustes
- ✅ Fournit des métriques quantitatives de robustesse

### 1.4 Avantages de l'Approche

| Critère | Approche Empirique | Approche Formelle (ce projet) |
|---------|-------------------|-------------------------------|
| Garantie | ❌ Non | ✅ Oui (mathématique) |
| Exhaustivité | ❌ Partielle | ✅ Complète |
| Temps | ⚠️ Variable | ✅ Déterministe |
| Précision | ⚠️ Stochastique | ✅ Bornes certifiées |

---

## 2. Fondements Théoriques

### 2.1 Expressions Affines

#### 2.1.1 Définition

Une expression affine représente une combinaison linéaire de variables symboliques :

```
y = a₀ + Σᵢ (aᵢ · xᵢ)    avec xᵢ ∈ [lᵢ, uᵢ]
```

Où :
- `a₀` : constante (terme indépendant)
- `aᵢ` : coefficient de la variable `xᵢ`
- `[lᵢ, uᵢ]` : bornes de la variable `xᵢ`

#### 2.1.2 Calcul des Bornes

Pour une expression affine `y`, les bornes `[min(y), max(y)]` se calculent par :

```
min(y) = a₀ + Σᵢ (aᵢ ≥ 0 ? aᵢ·lᵢ : aᵢ·uᵢ)
max(y) = a₀ + Σᵢ (aᵢ ≥ 0 ? aᵢ·uᵢ : aᵢ·lᵢ)
```

**Exemple :**
```
y = 2 + 3x₁ - x₂  avec x₁ ∈ [0, 1], x₂ ∈ [1, 2]

min(y) = 2 + 3·0 - 1·2 = 0
max(y) = 2 + 3·1 - 1·1 = 4
```

### 2.2 Propagation à Travers les Couches

#### 2.2.1 Couches Linéaires

Pour une couche linéaire `y = Wx + b` :

```python
# Pour chaque neurone de sortie yᵢ
yᵢ = bᵢ + Σⱼ (wᵢⱼ · xⱼ)

# Si xⱼ est une expression affine : xⱼ = a₀⁽ʲ⁾ + Σₖ (aₖ⁽ʲ⁾·εₖ)
# Alors yᵢ devient :
yᵢ = bᵢ + Σⱼ wᵢⱼ·(a₀⁽ʲ⁾ + Σₖ aₖ⁽ʲ⁾·εₖ)
   = (bᵢ + Σⱼ wᵢⱼ·a₀⁽ʲ⁾) + Σₖ (Σⱼ wᵢⱼ·aₖ⁽ʲ⁾)·εₖ
```

**Propriété :** Les opérations linéaires préservent la forme affine.

#### 2.2.2 Couches Convolutionnelles

Pour une convolution 2D :

```
yₒᵤₜ[b,c_out,h,w] = Σc_in Σkh Σkw (w[c_out,c_in,kh,kw] · x[b,c_in,h',w']) + bias[c_out]
```

Où `h' = h·stride + kh - padding` et `w' = w·stride + kw - padding`

**Implémentation :** Chaque pixel de sortie est une combinaison linéaire des pixels d'entrée → préserve la forme affine.

### 2.3 Relaxation des Activations Non-Linéaires

#### 2.3.1 ReLU : Cas d'Étude Principal

Pour `z = ReLU(y) = max(0, y)` avec `y ∈ [l, u]` :

**Cas 1 : Toujours inactif** (`u ≤ 0`)
```
z = 0
```

**Cas 2 : Toujours actif** (`l ≥ 0`)
```
z = y  (préserve l'expression affine)
```

**Cas 3 : Ambiguïté** (`l < 0 < u`)
```
Relaxation linéaire (sur-approximation conservative) :
z ∈ [0, u]
z ≤ (u/(u-l))·(y - l)

Expression affine résultante :
z = 0 + (u/(u-l))·[coefficients de y]
avec nouvelles bornes élargies
```

**Illustration graphique :**

```
    z |
    u |     /----  (borne supérieure relaxée)
      |    /
      |   /
    0 |__/________
      l  0     u   y
```

#### 2.3.2 Autres Activations

**Sigmoid :** `σ(y) = 1/(1+e⁻ʸ)`
```
Approximation linéaire au point milieu m = (l+u)/2 :
σ(y) ≈ σ(m) + σ'(m)·(y - m)
avec σ'(m) = σ(m)·(1 - σ(m))
```

**Tanh :** `tanh(y)`
```
Approximation linéaire similaire :
tanh(y) ≈ tanh(m) + (1 - tanh²(m))·(y - m)
```

### 2.4 MaxPooling : Relaxation Conservative

Pour `z = max(y₁, y₂, ..., yₙ)` :

```
Bornes conservatives :
min(z) = max(min(y₁), min(y₂), ..., min(yₙ))
max(z) = max(max(y₁), max(y₂), ..., max(yₙ))

Expression affine :
On prend l'expression avec max(yᵢ) le plus élevé
et on élargit les bornes pour couvrir tous les cas
```

### 2.5 Soundness (Correction)

**Définition :** Les bornes calculées sont **sound** si et seulement si :

```
∀x ∈ région bruitée, f(x) ∈ [min_formel, max_formel]
```

**Vérification statistique :**
1. Échantillonner N entrées bruitées aléatoirement
2. Calculer f(x) pour chaque échantillon
3. Vérifier que tous les f(x) ∈ bornes formelles
4. Si violation → implémentation incorrecte

---

## 3. Architecture Logicielle

### 3.1 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────┐
│                   Interface Utilisateur                  │
│  (CLI: affine_eval.py, API Python, Scripts de test)    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 Moteur Principal                         │
│  evaluate_model() - Orchestration du workflow           │
└────┬──────────────┬──────────────┬─────────────────┬───┘
     │              │              │                 │
     ▼              ▼              ▼                 ▼
┌─────────┐  ┌──────────┐  ┌─────────────┐  ┌────────────┐
│  ONNX   │  │ Affine   │  │   Bound     │  │  Result    │
│ Parser  │  │ Engine   │  │ Propagator  │  │ Aggregator │
└─────────┘  └──────────┘  └──────┬──────┘  └────────────┘
                                   │
                         ┌─────────▼──────────┐
                         │   NonLinear        │
                         │   Relaxer          │
                         └────────────────────┘
```

### 3.2 Modules Principaux

#### 3.2.1 `affine_engine.py`

**Rôle :** Représentation et manipulation des expressions affines

**Classes :**

```python
class AffineExpression:
    """
    Représente : y = a₀ + Σᵢ(aᵢ·xᵢ)
    
    Attributs:
        constant: float - a₀
        coefficients: Dict[int, float] - {i: aᵢ}
        bounds: Dict[int, Tuple[float, float]] - {i: (lᵢ, uᵢ)}
    
    Méthodes:
        get_bounds() -> (float, float)
        __add__, __mul__ : Opérations arithmétiques
    """

class AffineExpressionEngine:
    """
    Moteur de transformation des expressions
    
    Méthodes statiques:
        create_input_expressions(data, noise) -> List[AffineExpression]
        linear_layer(exprs, W, b) -> List[AffineExpression]
        conv2d_layer(...) -> Tuple[List[AffineExpression], shape]
        maxpool2d_layer(...) -> Tuple[List[AffineExpression], shape]
        avgpool2d_layer(...) -> Tuple[List[AffineExpression], shape]
    """
```

**Complexité :**
- `create_input_expressions`: O(n) où n = nombre de pixels
- `linear_layer`: O(m·k·s) où m=sorties, k=entrées, s=symboles moyens
- `conv2d_layer`: O(C_out·H_out·W_out·C_in·K_h·K_w·s)

#### 3.2.2 `onnx_parser.py`

**Rôle :** Extraction de l'architecture et des poids depuis ONNX

**Classe :**

```python
class ONNXParser:
    """
    Parse un modèle ONNX
    
    Méthodes:
        parse() -> List[Dict[str, Any]]
            Retourne la liste des couches avec leurs paramètres
        
        get_input_shape() -> tuple
        get_output_names() -> List[str]
    """
```

**Format de sortie :**

```python
{
    'type': 'Conv',  # Type d'opération ONNX
    'name': 'conv1',
    'inputs': ['input'],
    'outputs': ['conv1_out'],
    'attributes': {
        'strides': [1, 1],
        'pads': [1, 1, 1, 1],
        'dilations': [1, 1]
    },
    'weights': np.ndarray,  # (C_out, C_in, K_h, K_w)
    'bias': np.ndarray      # (C_out,)
}
```

#### 3.2.3 `relaxer.py`

**Rôle :** Implémentation des relaxations pour activations non-linéaires

**Classe :**

```python
class NonLinearRelaxer:
    """
    Relaxations des fonctions non-linéaires
    
    Méthodes statiques:
        relu_relaxation(expr, type='linear') -> AffineExpression
        sigmoid_relaxation(expr) -> AffineExpression
        tanh_relaxation(expr) -> AffineExpression
    """
```

**Algorithme ReLU Relaxation :**

```python
def relu_relaxation(expr, relaxation_type='linear'):
    l, u = expr.get_bounds()
    
    if u <= 0:
        return zero_expression()
    elif l >= 0:
        return expr  # Identité
    else:
        # Relaxation linéaire
        slope = u / (u - l)
        return scale_expression(expr, slope, offset=-l*slope)
```

---

## 4. Guide d'Installation

## 5. Guide d'Utilisation

## 6. API de Référence

## 7. Exemples Pratiques

## 8. Tests et Validation

## 9. Optimisation et Performance

## 10. Limitations et Extensions

## 11. FAQ

### 11.1 Questions Générales

**Q : Quelle est la différence entre évaluation formelle et tests empiriques ?**

**R :** 
- **Tests empiriques** (Monte Carlo) : Échantillonnent aléatoirement des cas et vérifient la robustesse sur ces échantillons. Pas de garantie sur les cas non testés.
- **Évaluation formelle** : Calcule mathématiquement les bornes exactes couvrant **tous** les cas possibles. Garantie exhaustive.

**Q : Pourquoi mes bornes sont-elles si larges ?**

**R :** Plusieurs raisons possibles :
1. **Niveau de bruit élevé** : Plus ε est grand, plus les bornes s'élargissent
2. **Relaxations conservatives** : Les relaxations ReLU/MaxPool sur-approximent
3. **Propagation de l'incertitude** : L'incertitude se cumule à travers les couches
4. **Complexité du réseau** : Réseaux profonds → bornes plus larges

**Solution :** Utiliser un niveau de bruit plus faible ou optimiser le réseau pour la robustesse.

**Q : Mon modèle n'est pas supporté, que faire ?**

**R :** Vérifiez que :
1. Le modèle est exporté en ONNX opset 11+
2. Toutes les couches utilisées sont supportées (voir section 3.2)
3. Pas de couches dynamiques (Dropout en mode training, etc.)

Pour ajouter une nouvelle couche, créez une méthode dans `affine_engine.py` et `bound_propagator.py`.

### 11.2 Questions Techniques

**Q : Comment interpréter les résultats de soundness ?**

**R :** Le rapport de soundness vérifie que les bornes formelles sont correctes :
- **is_sound = True** : ✅ Implémentation correcte, toutes les valeurs observées sont dans les bornes
- **is_sound = False** : ❌ Bug dans l'implémentation, certaines valeurs dépassent les bornes
- **violation_rate** : % de violations (doit être 0%)
- **marges de sécurité** : Distance entre valeurs observées et bornes (plus c'est large, mieux c'est)

**Q : Que signifie "classe robuste certifiée" ?**

**R :** Une classe est **robuste certifiée** si sa borne inférieure est strictement supérieure aux bornes supérieures de toutes les autres classes :

```
classe_robuste si: min(score(classe)) > max(score(autres_classes))
```

Cela garantit que **toute** perturbation dans la boule de bruit donnera cette classe.

**Q : Pourquoi l'évaluation formelle est-elle plus lente que l'inférence standard ?**

**R :** L'évaluation formelle :
- Manipule des expressions symboliques (pas juste des nombres)
- Maintient des coefficients pour chaque variable de bruit
- Calcule des bornes à chaque étape

**Complexité :**
- Inférence standard : O(n) où n = nombre d'opérations
- Évaluation formelle : O(n · s²) où s = nombre moyen de symboles

**Q : Comment réduire le temps d'exécution ?**

**R :** Plusieurs stratégies :
1. **Réduire le niveau de bruit** : Moins de complexité symbolique
2. **Simplifier les expressions** : Supprimer les coefficients négligeables
3. **Utiliser des approximations plus grossières** : Trade-off précision/vitesse
4. **Activer le cache** : Réutiliser les calculs de bornes
5. **Future : GPU** : Parallélisation massive

### 11.3 Questions de Développement

**Q : Comment ajouter une nouvelle couche (ex: LayerNorm) ?**

**R :** Suivez ces étapes :

1. **Ajouter la méthode dans `affine_engine.py` :**
```python
@staticmethod
def layernorm_layer(expressions: List[AffineExpression],
                   normalized_shape: tuple,
                   weight: np.ndarray,
                   bias: np.ndarray,
                   eps: float = 1e-5) -> List[AffineExpression]:
    """
    Implémente LayerNorm
    Note: LayerNorm est non-linéaire, nécessite une relaxation
    """
    # Votre implémentation ici
    pass
```

2. **Ajouter la propagation dans `bound_propagator.py` :**
```python
elif layer_type == 'LayerNorm':
    expressions = self._propagate_layernorm(expressions, layer)
```

3. **Ajouter la méthode privée :**
```python
def _propagate_layernorm(self, expressions, layer):
    attrs = layer.get('attributes', {})
    # Extraire les paramètres et appeler l'engine
    return self.engine.layernorm_layer(expressions, ...)
```

4. **Ajouter des tests dans `tests/` :**
```python
def test_layernorm_propagation(self):
    # Tester la propagation LayerNorm
    pass
```

**Q : Comment débugger une violation de soundness ?**

**R :** Procédure systématique :

1. **Identifier la couche problématique** :
```python
# Activer le mode verbeux
propagator = BoundPropagator(enable_reporting=True)
# Examiner intermediate_bounds pour chaque couche
```

2. **Vérifier les bornes couche par couche** :
```python
for layer_bounds in results['intermediate_bounds']:
    print(f"Couche {layer_bounds['layer']}: {layer_bounds['bounds']}")
```

3. **Tester une couche isolément** :
```python
# Créer des expressions de test
test_expr = AffineExpression(...)
result = engine.linear_layer([test_expr], weights, bias)
# Vérifier manuellement les bornes
```

4. **Comparer avec une implémentation de référence** :
```python
# Ex: ERAN, AI2, ou implémentation manuelle
```

**Q : Comment contribuer au projet ?**

**R :** Bienvenue ! Voici comment :

1. **Fork le dépôt**
2. **Créer une branche** : `git checkout -b feature/ma-feature`
3. **Développer avec tests** : Ajouter des tests unitaires
4. **Vérifier la qualité** :
```bash
python -m unittest discover tests/
black modules/ tests/  # Formatter le code
flake8 modules/ tests/  # Vérifier le style
```
5. **Commit et Push** :
```bash
git commit -m "feat: Ajout de LayerNorm"
git push origin feature/ma-feature
```
6. **Créer une Pull Request**

**Conventions :**
- Messages de commit : `feat:`, `fix:`, `docs:`, `test:`
- Docstrings : Format Google Style
- Tests : Coverage > 80%

### 11.4 Questions de Débogage

**Q : Erreur "ValueError: input_shape doit être spécifié" ?**

**R :** Le propagateur a perdu la trace de la forme des tenseurs. Solutions :
1. Vérifier que `current_shape` est correctement mis à jour à chaque couche
2. Passer explicitement `input_shape` à `propagate()`
3. Vérifier que les opérations de reshape sont correctement implémentées

**Q : Erreur "IndexError: index out of bounds" dans Conv2D ?**

**R :** Problème de padding ou de stride. Vérifications :
```python
# Calculer les dimensions de sortie
out_h = (in_h + 2*pad_h - dil_h*(kernel_h-1) - 1) // stride_h + 1
out_w = (in_w + 2*pad_w - dil_w*(kernel_w-1) - 1) // stride_w + 1

# Vérifier que out_h, out_w > 0
assert out_h > 0 and out_w > 0
```

**Q : Les bornes sont négatives alors qu'elles devraient être positives ?**

**R :** Vérifier :
1. **Après ReLU** : Les bornes doivent être ≥ 0
2. **Relaxation correcte** : `relu_relaxation()` gère les 3 cas
3. **Propagation des bornes** : `get_bounds()` calcule correctement min/max

Debug :
```python
# Avant ReLU
l, u = expr_before_relu.get_bounds()
print(f"Avant ReLU: [{l}, {u}]")

# Après ReLU
expr_after = relaxer.relu_relaxation(expr_before_relu)
l2, u2 = expr_after.get_bounds()
print(f"Après ReLU: [{l2}, {u2}]")
assert l2 >= 0, "Borne inférieure négative après ReLU!"
```

---

## 12. Références

### 12.1 Publications Scientifiques

**Vérification Formelle de Réseaux de Neurones :**

1. **DeepPoly** (POPL 2019)
   - *Singh, G., Gehr, T., Püschel, M., & Vechev, M.*
   - "An Abstract Domain for Certifying Neural Networks"
   - https://dl.acm.org/doi/10.1145/3290354

2. **AI2** (NeurIPS 2018)
   - *Gehr, T., Mirman, M., Drachsler-Cohen, D., et al.*
   - "AI2: Safety and Robustness Certification of Neural Networks with Abstract Interpretation"
   - https://proceedings.neurips.cc/paper/2018/hash/2ecd2bd94734e5dd392d8a896ac6bb18-Abstract.html

3. **CROWN** (NeurIPS 2018)
   - *Zhang, H., Weng, T. W., Chen, P. Y., Hsieh, C. J., & Daniel, L.*
   - "Efficient Neural Network Robustness Certification with General Activation Functions"
   - https://arxiv.org/abs/1811.00866

4. **Interval Bound Propagation** (ICLR 2019)
   - *Gowal, S., Dvijotham, K., Stanforth, R., et al.*
   - "Scalable Verified Training for Provably Robust Image Classification"
   - https://arxiv.org/abs/1811.12774

**Robustesse Adversariale :**

5. **Certified Defenses** (ICML 2018)
   - *Cohen, J., Rosenfeld, E., & Kolter, Z.*
   - "Certified Adversarial Robustness via Randomized Smoothing"
   - https://arxiv.org/abs/1902.02918

6. **Fast-Lin** (ICML 2018)
   - *Weng, T. W., Zhang, H., Chen, H., et al.*
   - "Towards Fast Computation of Certified Robustness for ReLU Networks"
   - https://arxiv.org/abs/1804.09699

7. **MILP Verification** (CAV 2017)
   - *Tjeng, V., Xiao, K., & Tedrake, R.*
   - "Evaluating Robustness of Neural Networks with Mixed Integer Programming"
   - https://arxiv.org/abs/1711.07356

**Analyse Abstraite :**

8. **Abstract Interpretation** (1977)
   - *Cousot, P., & Cousot, R.*
   - "Abstract Interpretation: A Unified Lattice Model for Static Analysis"
   - Fondements théoriques de l'analyse abstraite

### 12.2 Outils et Frameworks

**Outils de Vérification Existants :**

- **ERAN** (ETH Zurich)
  - https://github.com/eth-sri/eran
  - Supporte DeepPoly, DeepZono, RefineZono
  
- **auto_LiRPA** (Robust ML Lab)
  - https://github.com/Verified-Intelligence/auto_LiRPA
  - Automatic Linear Relaxation based Perturbation Analysis

- **CROWN** (IBM Research)
  - https://github.com/IBM/CROWN-Robustness-Certification
  - Certification avec fonctions d'activation générales

- **α,β-CROWN** (Competition Winner)
  - https://github.com/Verified-Intelligence/alpha-beta-CROWN
  - Gagnant VNN-COMP 2021, 2022, 2023

- **Marabou** (Stanford)
  - https://github.com/NeuralNetworkVerification/Marabou
  - SMT-based neural network verification

**Bibliothèques Python :**

- **ONNX** : https://github.com/onnx/onnx
- **ONNXRuntime** : https://github.com/microsoft/onnxruntime
- **PyTorch** : https://pytorch.org/
- **NumPy** : https://numpy.org/

### 12.3 Datasets

- **MNIST** : http://yann.lecun.com/exdb/mnist/
- **Fashion-MNIST** : https://github.com/zalandoresearch/fashion-mnist
- **CIFAR-10/100** : https://www.cs.toronto.edu/~kriz/cifar.html
- **ImageNet** : https://www.image-net.org/

### 12.4 Benchmarks de Vérification

- **VNN-COMP** (International Verification of Neural Networks Competition)
  - https://sites.google.com/view/vnn2023
  - Benchmarks standardisés, compétition annuelle

- **ACAS Xu** (Airborne Collision Avoidance System)
  - Benchmark classique pour vérification de réseaux de neurones
  - https://github.com/guykatzz/ReluplexCav2017

### 12.5 Cours et Tutoriels

**Cours en Ligne :**

1. **"Reliable and Interpretable Artificial Intelligence"** (ETH Zurich)
   - https://www.sri.inf.ethz.ch/teaching/riai2023
   - Cours complet sur la vérification formelle

2. **"Deep Learning Security"** (UC Berkeley)
   - https://dl-security.github.io/
   - Focus sur robustesse adversariale

3. **"Formal Methods for ML"** (CMU)
   - Fondamentaux de vérification formelle

**Tutoriels :**

- **PyTorch → ONNX Export** : https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
- **Abstract Interpretation Tutorial** : https://www.di.ens.fr/~cousot/AI/IntroAbsInt.html

### 12.6 Articles de Blog et Ressources

1. **"A Practical Guide to Neural Network Verification"**
   - Blog post détaillé sur les méthodes de vérification

2. **"Understanding Adversarial Examples"**
   - https://adversarial-ml-tutorial.org/

3. **"Certified Robustness Explained"**
   - Introduction accessible aux concepts de certification

### 12.7 Communauté et Support

**Forums et Discussions :**

- **r/MachineLearning** (Reddit)
- **ML Security Slack**
- **ONNX Community** : https://onnx.ai/community.html

**Conférences Pertinentes :**

- **NeurIPS** : Neural Information Processing Systems
- **ICML** : International Conference on Machine Learning
- **ICLR** : International Conference on Learning Representations
- **CAV** : Computer Aided Verification
- **S&P** : IEEE Symposium on Security and Privacy

### 12.8 Citations Suggérées

Si vous utilisez ce projet dans vos recherches, veuillez citer :

```bibtex
@misc{berthelot2024affine,
  title={Moteur d'Évaluation Formelle par Expression Affine},
  author={Berthelot, Guillaume},
  year={2024},
  howpublished={\url{https://github.com/username/AbstClaud}},
  note={Outil de vérification formelle pour réseaux de neurones}
}
```

---

## Annexes

### Annexe A : Glossaire

**Termes Techniques :**

- **Expression Affine** : Combinaison linéaire de variables symboliques avec bornes
- **Relaxation** : Sur-approximation d'une fonction non-linéaire par une fonction plus simple
- **Soundness** : Propriété garantissant que les bornes calculées contiennent toutes les valeurs possibles
- **Completeness** : Propriété garantissant que les bornes sont les plus serrées possibles (ce projet ne garantit pas la completeness)
- **Perturbation L∞** : Bruit borné pixel par pixel : |x' - x|∞ ≤ ε
- **Perturbation L2** : Bruit borné en norme euclidienne : ||x' - x||₂ ≤ ε
- **Certification** : Preuve mathématique de robustesse
- **Attaque Adversariale** : Perturbation optimisée pour tromper le réseau

**Acronymes :**

- **ONNX** : Open Neural Network Exchange
- **CNN** : Convolutional Neural Network
- **ReLU** : Rectified Linear Unit
- **FC** : Fully Connected (layer)
- **API** : Application Programming Interface
- **CLI** : Command Line Interface
- **GPU** : Graphics Processing Unit

### Annexe B : Format des Résultats JSON

Structure complète du fichier `results.json` :

```json
{
  "bounds_per_class": {
    "class_0": [lower, upper],
    "class_1": [lower, upper],
    ...
  },
  "robust_class": "class_5" | null,
  "intermediate_bounds": [
    {
      "layer": "conv1",
      "type": "Conv",
      "bounds": [[l1, u1], [l2, u2], ...],
      "shape": [1, 16, 28, 28]
    },
    ...
  ],
  "execution_time": 3.14,
  "noise_level": 0.05,
  "activation_relaxation": "linear",
  "detailed_report": {
    "total_layers": 12,
    "total_execution_time": 3.14,
    "total_expressions_processed": 15680,
    "total_symbols": 2352000,
    "avg_symbols_per_layer": 196000,
    "layer_types": {
      "Conv": {"count": 2, "total_time": 2.5, ...},
      ...
    },
    "timestamp": "2024-01-15T10:30:00"
  }
}
```

### Annexe C : Table de Complexité

| Opération | Entrée | Sortie | Complexité Temporelle | Complexité Spatiale |
|-----------|--------|--------|----------------------|---------------------|
| Linear | (n, s) | (m, s) | O(n · m · s) | O(m · s) |
| Conv2D | (C_in·H·W, s) | (C_out·H'·W', s) | O(C_in·C_out·K²·H'·W'·s) | O(C_out·H'·W'·s) |
| MaxPool | (C·H·W, s) | (C·H'·W', s) | O(C·H'·W'·K²·s) | O(C·H'·W'·s) |
| ReLU | (n, s) | (n, s) | O(n · s) | O(n · s) |

Où : s = nombre moyen de symboles par expression

### Annexe D : Checklist de Déploiement

**Avant déploiement en production :**

- [ ] Tests unitaires passent (100%)
- [ ] Tests de soundness OK (violation_rate = 0%)
- [ ] Performance acceptable (< 60s pour modèles cibles)
- [ ] Documentation à jour
- [ ] Logging configuré
- [ ] Gestion d'erreurs robuste
- [ ] Validation des entrées utilisateur
- [ ] Limites de ressources (timeout, mémoire)
- [ ] Monitoring mis en place
- [ ] Backups configurés

### Annexe E : Historique des Versions

**Version 1.0.0** (2024-01-15)
- ✨ Première version stable
- ✨ Support Conv2D, MaxPool, Linear, ReLU
- ✨ Soundness checker intégré
- ✨ Rapports détaillés
- ✨ Tests complets

**Version 0.9.0** (2024-01-01)
- 🔧 Beta release
- 🔧 Implémentation des couches de base

**Version 0.5.0** (2023-12-15)
- 🚧 Alpha release
- 🚧 Proof of concept

---

## Contact et Support

**Auteur :** Guillaume BERTHELOT

**Issues :** https://github.com/username/AbstClaud/issues

**Discussions :** https://github.com/username/AbstClaud/discussions

**Email :** [Votre email]

---

**Dernière mise à jour :** 2024-01-15

**Version de la documentation :** 1.0

---

*Cette documentation est maintenue activement. Pour toute question, correction ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.*
