# Hyper-Opt
Arquivo para geração do dataset de validação, treino e teste contido na pasta data_generator_COSMOS
Arquivo principal neural_network.py com os métodos de busca e o código da rede neural.
Pasta /dataset contém arquivos de exemplo.
Acesso ao dataset COSMOS completo https://drive.google.com/file/d/13BGv152uKLPLvYUqJiA4Yxj32g97JQdr/view?usp=sharing

## Execução

python neural_network.py arg1 arg2
arg1 = gs (grid search), rs (random search) ou bo (bayesian optimizaion)
arg2 = cosmos, rectangles, mnits (file dependecy)

Exemplo:
python neural_network.py rs cosmos
--Executar Random Search no dataset cosmos

python neural_network.py gs rectangles
--Executar Grid Search no dataset de retangulos

## Dependencies
###python 3.6.8 
##### Packages
numpy 1.16.3  
pandas 0.24.2  
scikit-learn 0.21.1  
scipy 1.2.1  
tensorflow 1.13.1 (to CPU process)  
tensorflow-gpu 1.13.1 (to GPU process)  

### R 3.5.0 ou superior
#### Packages
Todas as dependencias podem ser instaladas através da função pre_install_libraries() em Preprocessamento.R
Todas as dependencias são carregadas utilizando a função pre_load_libraries()

## Hyperparameters
LAYER1 = Neurons number in the first hidden layer  
LAYER2 = Neurons number in the second hidden layer  
LAYER3 = Neurons number in the third hidden layer  
LR = learning rate  
BETA = beta parameter to regularization. 0 to ignore  
