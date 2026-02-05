#!/bin/bash

echo "==================================================="
echo "  INICIANDO SETUP DE PROMPT ENGINEERING BENCHMARK"
echo "==================================================="

# 1. Crear entorno virtual
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi

# 2. Activar e instalar
echo "Instalando dependencias..."
source venv/bin/activate
pip install -r requirements.txt

# 3. Verificar Ollama
if ! command -v ollama &> /dev/null; then
    echo "Ollama no encontrado. Instalando..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama ya está instalado."
fi

# 4. Descargar modelo
echo "Descargando modelo Llama 3.1..."
ollama pull llama3.1

echo "==================================================="
echo "SETUP COMPLETADO"
echo "==================================================="