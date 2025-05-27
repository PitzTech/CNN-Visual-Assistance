import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape, Conv2D, Dropout
from tensorflow.keras.models import Model
import time
import pygame

# Configure GPU usage - Optimized for RTX 4080 Super
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) available: {len(gpus)}")
        print(f"GPU names: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU available, using CPU")

class ObjectDetectionSystem:
    """
    Sistema de detecção de objetos em ambientes educacionais usando MobileNetSSD
    para inclusão de pessoas com deficiência visual.
    """

    def __init__(self):
        # Configurações
        self.confidence_threshold = 0.6
        self.input_size = (300, 300)  # Tamanho padrão para MobileNetSSD

        # Classes de objetos a serem detectados
        self.classroom_objects = [
            'background', 'mesa', 'cadeira', 'quadro', 'mochila', 'porta',
            'janela', 'computador', 'projetor', 'estante', 'lixeira'
        ]

        self.school_supplies = [
            'background', 'caderno', 'lápis', 'caneta', 'borracha', 'régua',
            'livro', 'calculadora', 'apontador', 'tesoura', 'cola'
        ]

        # Inicializar modelos
        self.classroom_model = self._build_detection_model(len(self.classroom_objects))
        self.supplies_model = self._build_detection_model(len(self.school_supplies))

        # Inicializar sistema de áudio para feedback
        self._setup_audio()

        print("Sistema de Identificação de Objetos Educacionais inicializado com sucesso!")

    def _build_detection_model(self, num_classes):
        """
        Cria o modelo MobileNetSSD baseado em MobileNetV2 com TensorFlow/Keras
        """
        # Base MobileNetV2 sem camadas de classificação
        base_model = MobileNetV2(
            input_shape=(300, 300, 3),
            include_top=False,
            weights='imagenet'
        )

        # Congelar camadas da base
        for layer in base_model.layers:
            layer.trainable = False

        # Adicionar camadas SSD para detecção de objetos
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)

        # Saídas: classificação e bounding boxes
        class_output = Dense(num_classes, activation='softmax', name='class_output')(x)

        # Para cada classe, prever 4 coordenadas (x, y, width, height)
        bbox_output = Dense(num_classes * 4, activation='sigmoid', name='bbox_output')(x)
        bbox_output = Reshape((num_classes, 4), name='bbox_reshape')(bbox_output)

        # Modelo completo
        model = Model(inputs=base_model.input, outputs=[class_output, bbox_output])

        # Compilar o modelo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'class_output': 'categorical_crossentropy',
                'bbox_output': 'mean_squared_error'
            },
            loss_weights={'class_output': 1.0, 'bbox_output': 1.0},
            metrics={'class_output': 'accuracy'}
        )

        return model

    def _setup_audio(self):
        """
        Configura o sistema de áudio para feedback ao usuário
        """
        pygame.init()
        pygame.mixer.init()

    def _speak(self, text):
        """
        Função para fornecer feedback de áudio ao usuário
        Em um sistema real, poderia usar TTS como gTTS ou pyttsx3
        """
        print(f"Feedback de áudio: {text}")
        # Em um sistema real:
        # import pyttsx3
        # engine = pyttsx3.init()
        # engine.say(text)
        # engine.runAndWait()

    def preprocess_image(self, image):
        """
        Pré-processa a imagem para alimentar o modelo
        """
        image = cv2.resize(image, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_input(np.expand_dims(image, axis=0))
        return image

    def detect_objects(self, image, mode="classroom"):
        """
        Detecta objetos na imagem baseado no modo selecionado
        """
        processed_image = self.preprocess_image(image)

        if mode == "classroom":
            model = self.classroom_model
            classes = self.classroom_objects
        else:  # mode == "supplies"
            model = self.supplies_model
            classes = self.school_supplies

        # Realizar previsão
        class_probs, bbox_coords = model.predict(processed_image)

        # Processar resultados
        detected_objects = []

        for i in range(len(classes)):
            if i > 0 and class_probs[0][i] > self.confidence_threshold:  # Ignorar background (índice 0)
                # Extrair coordenadas da bounding box
                y_min, x_min, y_max, x_max = bbox_coords[0][i]

                # Converter para coordenadas de pixel
                x_min = int(x_min * image.shape[1])
                y_min = int(y_min * image.shape[0])
                x_max = int(x_max * image.shape[1])
                y_max = int(y_max * image.shape[0])

                detected_objects.append({
                    'class': classes[i],
                    'confidence': float(class_probs[0][i]),
                    'bbox': (x_min, y_min, x_max, y_max)
                })

        return detected_objects

    def provide_feedback(self, detected_objects):
        """
        Fornece feedback sobre os objetos detectados
        """
        if not detected_objects:
            self._speak("Nenhum objeto detectado.")
            return

        # Ordenar objetos por confiança
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)

        # Feedback sobre os 3 objetos mais confiáveis
        top_objects = detected_objects[:3]

        feedback = "Objetos detectados: "
        for i, obj in enumerate(top_objects):
            if i > 0:
                feedback += ", "
            feedback += f"{obj['class']} com {int(obj['confidence'] * 100)}% de certeza"

        self._speak(feedback)

        # Fornecer dicas sobre posição dos objetos principais
        main_object = top_objects[0]
        x_center = (main_object['bbox'][0] + main_object['bbox'][2]) / 2
        y_center = (main_object['bbox'][1] + main_object['bbox'][3]) / 2

        # Determinar a localização relativa (simplificada)
        h, w = self.input_size
        position = ""

        if x_center < w/3:
            position += "à esquerda"
        elif x_center > 2*w/3:
            position += "à direita"
        else:
            position += "no centro"

        if y_center < h/3:
            position += " e na parte superior"
        elif y_center > 2*h/3:
            position += " e na parte inferior"
        else:
            position += " e no meio"

        self._speak(f"O principal objeto ({main_object['class']}) está {position} da imagem.")

    def run_camera_detection(self, mode="classroom"):
        """
        Executa a detecção em tempo real usando a câmera
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro ao abrir a câmera.")
            return

        print(f"Modo de detecção: {'Sala de Aula' if mode == 'classroom' else 'Materiais Escolares'}")
        print("Pressione 'q' para sair, 'c' para capturar e analisar, 'm' para mudar o modo")

        current_mode = mode
        last_feedback_time = 0
        feedback_cooldown = 3  # segundos entre feedbacks automáticos

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mostra o modo atual na tela
            mode_text = "Modo: Sala de Aula" if current_mode == "classroom" else "Modo: Materiais Escolares"
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Exibir instruções
            cv2.putText(frame, "q: sair | c: capturar | m: mudar modo",
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Exibir o frame
            cv2.imshow('Detecção de Objetos Educacionais', frame)

            # Verificar entrada do usuário
            key = cv2.waitKey(1) & 0xFF

            # Capturar e analisar frame atual
            if key == ord('c'):
                detected_objects = self.detect_objects(frame, current_mode)

                # Desenhar bounding boxes
                for obj in detected_objects:
                    x_min, y_min, x_max, y_max = obj['bbox']
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    label = f"{obj['class']}: {int(obj['confidence'] * 100)}%"
                    cv2.putText(frame, label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Mostrar frame com detecções
                cv2.imshow('Detecção de Objetos Educacionais', frame)

                # Fornecer feedback sobre os objetos detectados
                self.provide_feedback(detected_objects)
                last_feedback_time = time.time()

            # Alterar entre os modos
            elif key == ord('m'):
                current_mode = "supplies" if current_mode == "classroom" else "classroom"
                mode_name = "Materiais Escolares" if current_mode == "supplies" else "Sala de Aula"
                self._speak(f"Modo alterado para {mode_name}")

            # Sair do loop
            elif key == ord('q'):
                break

            # Feedback automático a cada 'feedback_cooldown' segundos
            current_time = time.time()
            if current_time - last_feedback_time > feedback_cooldown:
                detected_objects = self.detect_objects(frame, current_mode)
                # Apenas com feedback de áudio, sem desenhar na tela
                if detected_objects:
                    self.provide_feedback(detected_objects)
                    last_feedback_time = current_time

        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()

# Script para treinar o modelo (pseudocódigo - necessita dados reais)
def train_model(model, dataset_path, classes, epochs=10):
    """
    Treina o modelo com um conjunto de dados específico
    (Pseudocódigo - em um cenário real, você precisaria implementar o pipeline de treinamento completo)
    """
    # Aqui você implementaria:
    # 1. Carregamento e processamento do dataset
    # 2. Data augmentation
    # 3. Treinamento do modelo
    # 4. Validação e avaliação
    print(f"Treinando modelo para {len(classes)} classes...")
    # model.fit(...) - Treinamento real aqui
    print("Treinamento concluído!")
    return model

# Função principal
def main():
    # Criar o sistema de detecção
    detection_system = ObjectDetectionSystem()

    # Em um cenário real, você treinaria os modelos com datasets específicos
    # classroom_dataset_path = "path/to/classroom/dataset"
    # supplies_dataset_path = "path/to/supplies/dataset"
    # detection_system.classroom_model = train_model(detection_system.classroom_model,
    #                                               classroom_dataset_path,
    #                                               detection_system.classroom_objects)
    # detection_system.supplies_model = train_model(detection_system.supplies_model,
    #                                              supplies_dataset_path,
    #                                              detection_system.school_supplies)

    # Iniciar o sistema com a câmera
    print("\nIniciando sistema de detecção de objetos educacionais...")
    print("Este sistema auxilia pessoas com deficiência visual a identificar:")
    print("1. Objetos em sala de aula (mesas, cadeiras, quadro, etc.)")
    print("2. Materiais escolares (cadernos, lápis, canetas, etc.)")
    print("\nO sistema fornecerá feedback por áudio sobre os objetos detectados.")

    # Iniciar com o modo "sala de aula" por padrão
    detection_system.run_camera_detection(mode="classroom")

if __name__ == "__main__":
    main()
