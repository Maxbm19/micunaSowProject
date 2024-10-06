Se tiene codigo relacionado al entrenamiento de un modelo de Machine learning supervisado para predecir si el agricultor debe o no sembrar un producto para tener una buena produccion.

Para ello, una vez desarrollado el modelo se lo puso en produccion con Flask como backend para hacer inferencia. La cual se conecta con el frontend:

![image](https://github.com/user-attachments/assets/e4f6330e-b352-446c-8971-241d7bd7cb52)

Donde se selecciona un punto en el mapa y el producto a predecir.

El backend se conecta con las APIs del clima para generar la X que utlizara para la inferencia, se recupera el modelo ya entrenado y bajo solicites GET se calcula la probabilidad
y se develven datos al Frontend:

![image](https://github.com/user-attachments/assets/657e037a-4c10-4c41-9665-58c6037eafd4)

Inferencia en el backend:

![image](https://github.com/user-attachments/assets/8cee2e8c-6abb-4ed0-9eae-b637415124bc)
