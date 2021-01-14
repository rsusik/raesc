import main
from importlib import reload
reload(main)
from main import *
from IPython import display
import sys

if __name__ == "__main__":


  starttime = timer()

  model_mapping = {
    'rae': Models.RAE,
    'raes': Models.RAES,
    'raesc': Models.RAESC
  }

  models = list(map(lambda x: model_mapping[x], sys.argv[1].split(',')))
  n_features = list(map(lambda x: int(x), sys.argv[2].split(',')))
  n_hidden_dim_delimiter = list(map(lambda x: float(x), sys.argv[3].split(',')))
  
  print(models)
  print(n_features)
  print(n_hidden_dim_delimiter)

  tmp = "_".join([sys.argv[1], sys.argv[2], sys.argv[3]]).replace(",", "_")

  output_filename = f'results_{tmp}.pickle'
  
  #with tf.device('/cpu:0'):
  training = train(
      models=models, 
      n_epochs=100,
      n_hidden_dim_delimiter=n_hidden_dim_delimiter,
      n_batch_size=100,
      n_learning_rate=[0.001],
      n_features=n_features,
      n_samples=5000,
      output_filename=output_filename
  )

  print(timer()-starttime)
  print(timer())
