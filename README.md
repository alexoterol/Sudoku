## Programa Sudoku
# Resumen
Este programa fue creado en son de aprender diferentes librerías y el funcionamiento del Machine Learning en un problema NP como lo es el juego Sudoku <br/>
<br/>
Para ello se utilizó: Keras, PyQt5 y numpy.<br />
Utilizando los datos en Kaggle: https://www.kaggle.com/datasets/rohanrao/sudoku/data <br/>
Se construyó y entrenó un modelo con los 9 millones de datos dentro de Kaggle y así tener respuestas correctas. <br/>
El programa tiene tres modos: <br/>
Jugar -------------- El usuario tiene la oportunidad de jugar sudoku <br/>
Solución Manual ---- El programa soluciona el sudoku generado con uan función recursiva <br/>
Solución NN -------- El programa soluciona el sudoku generado usando el modelo y los pesos que tuvimos de resultado en el entrenamiento <br/>

# Observaciones:
Para utilizar el modo Solución NN se deben de adaptar los datos de entrada, ya que recibe un string del modo: <br/>
''' <br/>
  0 0 0 7 0 0 0 9 6 <br/>
  0 0 3 0 6 9 1 7 8 <br/>
  0 0 7 2 0 0 5 0 0 <br/>
  0 7 5 0 0 0 0 0 0 <br/>
  9 0 1 0 0 0 3 0 0 <br/>
  0 0 0 0 0 0 0 0 0 <br/>
  0 0 9 0 0 0 0 0 1 <br/>
  3 1 8 0 2 0 4 0 7 <br/>
  2 4 0 0 0 5 0 0 0 <br/>
'''<br/>
Y regresa un valor del tipo string:<br/>
"184753296523469178697281543875312964961547382432698715759834621318926457246175839"
