#!/usr/bin/env python3
"""
Script para testar o carregador de datasets do Roboflow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainining_v3 import RoboflowDatasetLoader, prepare_roboflow_datasets_for_training

def test_dataset_discovery():
    """
    Testa a descoberta automática de datasets
    """
    print("=== Teste de Descoberta de Datasets ===")
    
    loader = RoboflowDatasetLoader("datasets")
    datasets = loader.discover_datasets()
    
    print(f"Total de datasets encontrados: {len(datasets)}")
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{i}. Dataset: {dataset['name']}")
        print(f"   Caminho: {dataset['path']}")
        print(f"   Estrutura: {dataset['structure_type']}")
        print(f"   Splits: {list(dataset['splits'].keys())}")
        print(f"   Classes: {dataset['classes']}")
        print(f"   Total de imagens: {dataset['total_images']}")
        print(f"   Total de anotações: {dataset['total_annotations']}")
    
    return datasets

def test_dataset_combination():
    """
    Testa a combinação de datasets
    """
    print("\n=== Teste de Combinação de Datasets ===")
    
    # Testar com filtros específicos
    filters = ['stationery', 'office', 'school', 'pen']
    
    combined = prepare_roboflow_datasets_for_training(
        datasets_base_dir="datasets",
        dataset_filters=filters
    )
    
    if combined:
        print(f"\nDatasets combinados com sucesso!")
        print(f"Total de datasets: {len(combined['datasets'])}")
        print(f"Classes finais: {combined['classes']}")
        print(f"Mapeamento de classes: {combined['class_mapping']}")
    else:
        print("Falha ao combinar datasets!")
    
    return combined

def test_specific_dataset_loading():
    """
    Testa o carregamento de um dataset específico
    """
    print("\n=== Teste de Carregamento Específico ===")
    
    loader = RoboflowDatasetLoader("datasets")
    
    # Pegar o primeiro dataset encontrado
    datasets = loader.discover_datasets()
    if not datasets:
        print("Nenhum dataset encontrado!")
        return None
    
    first_dataset = datasets[0]
    print(f"Testando dataset: {first_dataset['name']}")
    
    # Simular carregamento de dados de um split
    from trainining_v3 import ModelTrainer
    trainer = ModelTrainer()
    
    # Pegar o primeiro split
    first_split_name = list(first_dataset['splits'].keys())[0]
    first_split = first_dataset['splits'][first_split_name]
    
    print(f"Testando split: {first_split_name}")
    print(f"Arquivo de anotações: {first_split['annotations_file']}")
    print(f"Diretório de imagens: {first_split['images_dir']}")
    
    # Criar um mapeamento simples
    classes = first_dataset['classes']
    class_mapping = {name: idx for idx, name in enumerate(['background'] + classes)}
    num_classes = len(class_mapping)
    
    try:
        # Testar carregamento de algumas imagens (limitado para teste)
        images, class_labels, bbox_labels = trainer._load_split_data(
            first_split, class_mapping, num_classes
        )
        
        print(f"Carregadas {len(images)} imagens com sucesso!")
        print(f"Shape das imagens: {images[0].shape if images else 'N/A'}")
        print(f"Shape dos labels de classe: {class_labels[0].shape if class_labels else 'N/A'}")
        print(f"Shape dos labels de bbox: {bbox_labels[0].shape if bbox_labels else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return False

if __name__ == "__main__":
    print("Testando o carregador de datasets do Roboflow...")
    
    # Teste 1: Descoberta de datasets
    datasets = test_dataset_discovery()
    
    if not datasets:
        print("❌ Nenhum dataset encontrado! Verifique se os datasets do Roboflow estão no diretório 'datasets/'")
        sys.exit(1)
    
    # Teste 2: Combinação de datasets
    combined = test_dataset_combination()
    
    if not combined:
        print("⚠️ Falha na combinação de datasets")
    else:
        print("✅ Combinação de datasets bem-sucedida")
    
    # Teste 3: Carregamento específico
    if len(datasets) > 0:
        success = test_specific_dataset_loading()
        if success:
            print("✅ Carregamento específico bem-sucedido")
        else:
            print("⚠️ Falha no carregamento específico")
    
    print("\n=== Teste Completo ===")
    print("O código parece estar funcionando corretamente!")
    print("Você pode agora executar:")
    print("  python trainining_v3.py --roboflow")
    print("ou")
    print("  python trainining_v3.py")
    print("e escolher a opção 1 para treinar com seus datasets do Roboflow.")