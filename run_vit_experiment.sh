#!/bin/bash

echo "=== Запуск эксперимента Vision Transformer vs CNN ==="
echo "Автор: Студент"
echo "Дата: 2025-06-05"
echo ""

# Проверка наличия Python
if ! command -v python &> /dev/null; then
    echo "❌ Python не найден. Установите Python и попробуйте снова."
    exit 1
fi

# Проверка наличия требуемых файлов
if [ ! -f "vision_transformer_comparison.py" ]; then
    echo "❌ Файл vision_transformer_comparison.py не найден!"
    exit 1
fi

if [ ! -f "requirements_vit.txt" ]; then
    echo "❌ Файл requirements_vit.txt не найден!"
    exit 1
fi

echo "✅ Все необходимые файлы найдены"
echo ""

# Установка зависимостей (опционально)
read -p "Установить зависимости? (y/n): " install_deps
if [ "$install_deps" = "y" ] || [ "$install_deps" = "Y" ]; then
    echo "📦 Устанавливаем зависимости..."
    pip install -r requirements_vit.txt
    echo ""
fi

# Запуск основного эксперимента
echo "🚀 Запускаем эксперимент..."
echo "⚠️  Это может занять около 10-15 минут в зависимости от вашего оборудования"
echo ""

python vision_transformer_comparison.py

echo ""
echo "✅ Эксперимент завершен!"
echo "📊 Проверьте созданные файлы:"
echo "   - training_comparison.png - графики обучения"
echo "   - confusion_matrices.png - матрицы ошибок"
echo ""
echo "📓 Для интерактивного анализа запустите:"
echo "   jupyter notebook vision_transformer_notebook.ipynb" 