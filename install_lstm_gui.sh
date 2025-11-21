#!/bin/bash
# Script de instalación para la interfaz LSTM GUI

echo "=================================================="
echo "Instalación de dependencias para LSTM GUI"
echo "=================================================="
echo ""

# Verificar que estamos en el entorno virtual
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  ADVERTENCIA: No se detectó un entorno virtual activo"
    echo "   Ejecuta primero: source .venv/bin/activate"
    echo ""
    read -p "¿Continuar de todos modos? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📦 Instalando PyTorch..."
pip install torch torchvision torchaudio

echo ""
echo "📦 Instalando dependencias adicionales..."
cd store_sim/src/PartB
pip install -r requirements_lstm_gui.txt

echo ""
echo "✅ Instalación completada!"
echo ""
echo "Para ejecutar la aplicación:"
echo "  python run_lstm_gui.py"
echo ""
