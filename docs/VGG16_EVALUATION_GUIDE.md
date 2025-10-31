# Guide d'Évaluation Formelle de VGG16

## Vue d'ensemble

VGG16 est un réseau de neurones convolutionnel profond avec :
- **Entrée** : 224×224×3 (150,528 pixels)
- **Architecture** : 13 couches Conv + 5 MaxPool + 3 FC
- **Sortie** : 1000 classes (ImageNet)

## Défis pour l'Évaluation Formelle

### 1. Complexité Symbolique

**Problème** : Chaque pixel d'entrée devient un symbole
