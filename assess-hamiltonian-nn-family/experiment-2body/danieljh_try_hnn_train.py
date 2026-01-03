import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float32)
import argparse
import numpy as np
import matplotlib.pyplot as plt

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

# from nn_models import MLP
# from hnn import HNN
# from data import get_dataset
# from utils import L2_loss, to_pickle, from_pickle
"""
Adapted to the current directory structures (by Jae Hoon (Daniel) Lee).
Explictly added ../hamiltonian_nn/ path.
"""
sys.path.append(PARENT_DIR)
HNN_PATH = os.path.join(PARENT_DIR, 'hamiltonian_nn')
if os.path.exists(HNN_PATH):
    sys.path.append(HNN_PATH)
else:
    print(f"Warning: {HNN_PATH} not found.")
from hamiltonian_nn.nn_models import MLP
from hamiltonian_nn.hnn import HNN
from data import get_dataset
from hamiltonian_nn.utils import L2_loss, to_pickle, from_pickle

# --- try setting seed on the top of the script ---
seed_value = 0
torch.manual_seed(seed_value)
np.random.seed(seed_value)
# import random
# random.seed(seed_value) # python 기본 random 모듈도 추가

def save_model_weights(model, directory, filename):
    try:
        if not filename.lower().endswith(('.pth', '.pt', '.tar')):
            filename += '.pth'
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        full_path = os.path.join(directory, filename)
        torch.save(model.state_dict(), full_path)
        
        print(f"Model's weight has successfully been saved: {full_path}")
    
    except Exception as e:
        print(f"Error occurred while saving the model: {e}")
      

def get_hnn_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2*4, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=200, type=int, help='batch_size')
    parser.add_argument('--input_noise', default=0.0, type=int, help='std of noise added to inputs')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=10000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='2body', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def get_hnn_model(device):
    from hamiltonian_nn.nn_models import MLP
    args = get_hnn_args()
    output_dim = 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type) 
    return model

def hnn_train(args=None):
  device = 'cpu'
  torch.set_grad_enabled(True)
  
  args = get_hnn_args()
  args.verbose = True
  
  model = get_hnn_model(device)

  ''' Moved to the top '''
  # set random seed
  # torch.manual_seed(args.seed)
  # np.random.seed(args.seed)

  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0)

  # arrange data
  data = get_dataset(args.name, args.save_dir, verbose=True)

   # Try torch.tensor() with device added, instead of torch.Tensor().
  x = torch.tensor( data['coords'], requires_grad=True, dtype=torch.float32, device=device)
  test_x = torch.tensor( data['test_coords'], requires_grad=True, dtype=torch.float32, device=device)
  dxdt = torch.tensor(data['dcoords'], dtype=torch.float32, device=device) 
  test_dxdt = torch.tensor(data['test_dcoords'], dtype=torch.float32, device=device) 

  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):

    # train step
    ixs = torch.randperm(x.shape[0])[:args.batch_size]
    dxdt_hat = model.time_derivative(x[ixs])
    noise = args.input_noise * torch.randn(*x[ixs].shape) # add noise, maybe
    dxdt_hat += noise
    loss = L2_loss(dxdt[ixs], dxdt_hat)
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
    optim.step() ; optim.zero_grad()

    # run test data
    test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
    test_dxdt_hat = model.time_derivative(test_x[test_ixs])
    noise = args.input_noise * torch.randn(*test_x[test_ixs].shape) # add noise, maybe
    test_dxdt_hat += noise
    test_loss = L2_loss(test_dxdt[test_ixs], test_dxdt_hat)

    # logging
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())
    if args.verbose and step % args.print_every == 0:
      print("step {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
          .format(step, loss.item(), test_loss.item(), grad@grad, grad.std()))

  train_dxdt_hat = model.time_derivative(x)
  train_dist = (dxdt - train_dxdt_hat)**2
  test_dxdt_hat = model.time_derivative(test_x)
  test_dist = (test_dxdt - test_dxdt_hat)**2
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))
  return model, stats

if __name__ == "__main__":
    args = get_hnn_args()
    hnn_model, stats = hnn_train()

    save_dir = THIS_DIR + "/weights"
    save_model_weights(hnn_model, save_dir,"danieljh_hnn_2body_cpu_trained_directly_on_terminal.pth")
