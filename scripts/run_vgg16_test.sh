#!/bin/bash

echo "=============================================="
echo "  Test VGG16 - Évaluation Formelle"
echo "=============================================="
echo ""

# Activer l'environnement virtuel si nécessaire
# source venv/bin/activate

# Créer les dossiers nécessaires
mkdir -p models results data logs

# Vérifier la disponibilité de Python
if ! command -v python &> /dev/null; then
    echo "❌ Python n'est pas installé ou n'est pas dans le PATH"
    exit 1
fi

echo "🐍 Version Python : $(python --version)"
echo ""

# Lancer les tests avec gestion des erreurs
echo "[1/1] Exécution des tests VGG16..."
python -m unittest tests.test_vgg16_formal -v 2>&1 | tee logs/vgg16_test_$(date +%Y%m%d_%H%M%S).log

# Capturer le code de sortie
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Tests terminés avec succès"
else
    echo "⚠️  Tests terminés avec des erreurs (code: $EXIT_CODE)"
fi

echo "  - Logs sauvegardés dans logs/"
echo "  - Résultats dans results/"

exit $EXIT_CODE
