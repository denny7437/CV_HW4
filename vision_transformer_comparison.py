#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Домашнее задание 4: Vision Transformers vs CNN
Сравнение производительности ViT и CNN на датасете CIFAR-10

Автор: Студент
Дата: 2025-06-05
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Настройка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# Настройка воспроизводимости результатов
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# =============================================================================
# 1. ПОДГОТОВКА ДАННЫХ
# =============================================================================

def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    """
    Загрузка и подготовка датасета CIFAR-10
    
    Args:
        batch_size (int): размер батча
        num_workers (int): количество процессов для загрузки данных
    
    Returns:
        train_loader, test_loader, classes
    """
    # Трансформации для обучающей выборки (с аугментацией)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Случайная обрезка с padding
        transforms.RandomHorizontalFlip(),     # Случайное горизонтальное отражение
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Трансформации для тестовой выборки (без аугментации)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Загрузка датасета
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Создание DataLoader'ов
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Классы CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes

# =============================================================================
# 2. РЕАЛИЗАЦИЯ VISION TRANSFORMER (ViT)
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Модуль для разбиения изображения на патчи и их эмбеддинга
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Сверточный слой для создания патч-эмбеддингов
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        x = self.proj(x)  # [batch_size, embed_dim, n_patches^0.5, n_patches^0.5]
        x = x.flatten(2)  # [batch_size, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [batch_size, n_patches, embed_dim]
        return x

class MultiHeadAttention(nn.Module):
    """
    Модуль многоголового внимания (Multi-Head Attention)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim должно быть кратно num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Вычисление Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Вычисление внимания
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Применение внимания к значениям
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.proj(out)
        
        return out

class TransformerBlock(nn.Module):
    """
    Блок трансформера (Multi-Head Attention + MLP)
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP блок
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Остаточное соединение с нормализацией (Pre-LN)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) для классификации изображений
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Патч эмбеддинги
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Класс токен (CLS token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Позиционные эмбеддинги
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Dropout
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer блоки
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Финальная нормализация и классификационная голова
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Патч эмбеддинги
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]
        
        # Добавление CLS токена
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, num_patches + 1, embed_dim]
        
        # Добавление позиционных эмбеддингов
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Прохождение через transformer блоки
        for block in self.blocks:
            x = block(x)
        
        # Нормализация и классификация по CLS токену
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Берем только CLS токен
        out = self.head(cls_token_final)
        
        return out

# =============================================================================
# 3. РЕАЛИЗАЦИЯ CNN ДЛЯ СРАВНЕНИЯ
# =============================================================================

class ConvBlock(nn.Module):
    """
    Базовый сверточный блок с нормализацией и активацией
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CNN_Classifier(nn.Module):
    """
    CNN модель для сравнения с ViT
    Архитектура схожа с простой версией ResNet
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Первый блок
        self.conv1 = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(64, 64)
        self.maxpool1 = nn.MaxPool2d(2)  # 32x32 -> 16x16
        
        # Второй блок
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.maxpool2 = nn.MaxPool2d(2)  # 16x16 -> 8x8
        
        # Третий блок
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.maxpool3 = nn.MaxPool2d(2)  # 8x8 -> 4x4
        
        # Четвертый блок
        self.conv7 = ConvBlock(256, 512)
        self.conv8 = ConvBlock(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 4x4 -> 1x1
        
        # Классификационная голова
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        
        return x

# =============================================================================
# 4. ФУНКЦИИ ОБУЧЕНИЯ И ОЦЕНКИ
# =============================================================================

def train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs=50):
    """
    Обучение модели
    
    Args:
        model: модель для обучения
        train_loader: загрузчик обучающих данных
        criterion: функция потерь
        optimizer: оптимизатор
        scheduler: планировщик learning rate
        device: устройство для вычислений
        epochs: количество эпох
    
    Returns:
        history: история обучения
    """
    model.train()
    history = {
        'train_loss': [],
        'train_acc': [],
        'time_per_epoch': []
    }
    
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Вывод прогресса каждые 100 батчей
            if batch_idx % 100 == 0:
                print(f'Эпоха {epoch+1}/{epochs}, Батч {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        # Обновление learning rate
        scheduler.step()
        
        # Сохранение метрик
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        epoch_time = time.time() - start_time
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['time_per_epoch'].append(epoch_time)
        
        print(f'Эпоха {epoch+1}/{epochs} завершена. Loss: {epoch_loss:.4f}, '
              f'Acc: {epoch_acc:.2f}%, Время: {epoch_time:.2f}с')
        print('-' * 60)
    
    return history

def evaluate_model(model, test_loader, device, classes):
    """
    Оценка модели на тестовой выборке
    
    Args:
        model: обученная модель
        test_loader: загрузчик тестовых данных
        device: устройство для вычислений
        classes: список классов
    
    Returns:
        accuracy, predictions, targets
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    print(f'\nТочность на тестовой выборке: {accuracy:.2f}%')
    
    # Подробный отчет по классам
    print('\nОтчет по классам:')
    print(classification_report(all_targets, all_predictions, target_names=classes))
    
    return accuracy, all_predictions, all_targets

def plot_training_history(vit_history, cnn_history):
    """
    Визуализация истории обучения
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # График потерь
    axes[0].plot(vit_history['train_loss'], label='ViT', linewidth=2)
    axes[0].plot(cnn_history['train_loss'], label='CNN', linewidth=2)
    axes[0].set_title('Потери во время обучения', fontsize=14)
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График точности
    axes[1].plot(vit_history['train_acc'], label='ViT', linewidth=2)
    axes[1].plot(cnn_history['train_acc'], label='CNN', linewidth=2)
    axes[1].set_title('Точность во время обучения', fontsize=14)
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Точность (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # График времени на эпоху
    axes[2].plot(vit_history['time_per_epoch'], label='ViT', linewidth=2)
    axes[2].plot(cnn_history['time_per_epoch'], label='CNN', linewidth=2)
    axes[2].set_title('Время на эпоху', fontsize=14)
    axes[2].set_xlabel('Эпоха')
    axes[2].set_ylabel('Время (сек)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(vit_targets, vit_predictions, cnn_targets, cnn_predictions, classes):
    """
    Визуализация матриц ошибок
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Матрица ошибок для ViT
    vit_cm = confusion_matrix(vit_targets, vit_predictions)
    sns.heatmap(vit_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title('Vision Transformer (ViT)', fontsize=14)
    axes[0].set_xlabel('Предсказанный класс')
    axes[0].set_ylabel('Истинный класс')
    
    # Матрица ошибок для CNN
    cnn_cm = confusion_matrix(cnn_targets, cnn_predictions)
    sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title('Convolutional Neural Network (CNN)', fontsize=14)
    axes[1].set_xlabel('Предсказанный класс')
    axes[1].set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def count_parameters(model):
    """
    Подсчет количества параметров модели
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# 5. ОСНОВНАЯ ФУНКЦИЯ ЭКСПЕРИМЕНТА
# =============================================================================

def main():
    """
    Основная функция для запуска эксперимента
    """
    print("="*80)
    print("СРАВНЕНИЕ VISION TRANSFORMER И CNN НА CIFAR-10")
    print("="*80)
    
    # Параметры обучения
    batch_size = 128
    epochs = 30
    learning_rate = 0.001
    
    # Загрузка данных
    print("Загрузка данных CIFAR-10...")
    train_loader, test_loader, classes = get_cifar10_dataloaders(batch_size)
    print(f"Обучающая выборка: {len(train_loader.dataset)} изображений")
    print(f"Тестовая выборка: {len(test_loader.dataset)} изображений")
    print(f"Классы: {classes}")
    
    # Создание моделей
    print("\nСоздание моделей...")
    
    # ViT модель (адаптированная для CIFAR-10)
    vit_model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        dropout=0.1
    ).to(device)
    
    # CNN модель
    cnn_model = CNN_Classifier(num_classes=10).to(device)
    
    # Подсчет параметров
    vit_params = count_parameters(vit_model)
    cnn_params = count_parameters(cnn_model)
    
    print(f"Параметры ViT: {vit_params:,}")
    print(f"Параметры CNN: {cnn_params:,}")
    
    # Настройка оптимизаторов и функций потерь
    vit_optimizer = optim.AdamW(vit_model.parameters(), lr=learning_rate, weight_decay=0.05)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    vit_scheduler = optim.lr_scheduler.CosineAnnealingLR(vit_optimizer, T_max=epochs)
    cnn_scheduler = optim.lr_scheduler.StepLR(cnn_optimizer, step_size=15, gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()
    
    # Обучение ViT
    print(f"\n{'='*20} ОБУЧЕНИЕ VISION TRANSFORMER {'='*20}")
    vit_start_time = time.time()
    vit_history = train_model(vit_model, train_loader, criterion, vit_optimizer, 
                             vit_scheduler, device, epochs)
    vit_training_time = time.time() - vit_start_time
    
    # Обучение CNN
    print(f"\n{'='*20} ОБУЧЕНИЕ CNN {'='*20}")
    cnn_start_time = time.time()
    cnn_history = train_model(cnn_model, train_loader, criterion, cnn_optimizer,
                             cnn_scheduler, device, epochs)
    cnn_training_time = time.time() - cnn_start_time
    
    # Оценка моделей
    print(f"\n{'='*20} ОЦЕНКА МОДЕЛЕЙ {'='*20}")
    
    print("\nОценка Vision Transformer:")
    vit_accuracy, vit_predictions, vit_targets = evaluate_model(vit_model, test_loader, device, classes)
    
    print("\nОценка CNN:")
    cnn_accuracy, cnn_predictions, cnn_targets = evaluate_model(cnn_model, test_loader, device, classes)
    
    # Сводка результатов
    print(f"\n{'='*20} СВОДКА РЕЗУЛЬТАТОВ {'='*20}")
    print(f"{'Модель':<20} {'Параметры':<15} {'Время обучения':<20} {'Точность':<15}")
    print("-" * 70)
    print(f"{'Vision Transformer':<20} {vit_params:,<15} {vit_training_time:.2f}с {'':<7} {vit_accuracy:.2f}%")
    print(f"{'CNN':<20} {cnn_params:,<15} {cnn_training_time:.2f}с {'':<7} {cnn_accuracy:.2f}%")
    
    # Визуализация результатов
    print(f"\n{'='*20} ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ {'='*20}")
    
    # График обучения
    plot_training_history(vit_history, cnn_history)
    
    # Матрицы ошибок
    plot_confusion_matrices(vit_targets, vit_predictions, cnn_targets, cnn_predictions, classes)
    
    # Анализ результатов
    print(f"\n{'='*20} АНАЛИЗ РЕЗУЛЬТАТОВ {'='*20}")
    
    print("ПРЕИМУЩЕСТВА И НЕДОСТАТКИ:")
    print("\nVision Transformer (ViT):")
    print("+ Глобальное внимание ко всем частям изображения")
    print("+ Хорошо масштабируется с увеличением данных")
    print("+ Меньше индуктивных предположений")
    print("- Требует больше данных для обучения")
    print("- Больше параметров")
    print("- Медленнее в обучении на малых датасетах")
    
    print("\nConvolutional Neural Network (CNN):")
    print("+ Эффективно использует пространственные корреляции")
    print("+ Меньше параметров")
    print("+ Быстрее обучается на небольших датасетах")
    print("+ Встроенная трансляционная инвариантность")
    print("- Ограниченное рецептивное поле")
    print("- Сильные индуктивные предположения")
    
    print(f"\n{'='*20} ЭКСПЕРИМЕНТ ЗАВЕРШЕН {'='*20}")

if __name__ == "__main__":
    main() 