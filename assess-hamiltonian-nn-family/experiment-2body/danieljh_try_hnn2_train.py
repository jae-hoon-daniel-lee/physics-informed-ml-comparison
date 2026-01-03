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


class HNN2(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                    baseline=False, assume_canonical_coords=True):
        super(HNN2, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.field_type = field_type

        # --- Modification 1 for GPU support: use register_buffer() ---
        # self.M = self.permutation_tensor(input_dim) # original code
        M_tensor = self.permutation_tensor(input_dim)
        self.register_buffer('M_buffer', M_tensor) # register M tensor to buffer
        
    def forward(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1,1)

    '''def rk4_time_derivative(self, x, dt):
        # rk4 needs also be modified to support GPU, or torchdiffeq needs to be used.
        return rk4(fun=self.time_derivative, y0=x, t=0, dt=dt)'''

    def time_derivative(self, x, t=None, separate_fields=False):
        '''NEURAL ODE-STLE VECTOR FIELD'''
        if self.baseline:
            return self.differentiable_model(x)

        '''NEURAL HAMILTONIAN-STLE VECTOR FIELD'''
        F1, F2 = self.forward(x) # traditional forward pass

        # --- Modification 2 for GPU support: when creating tensors, x.device is used ---
        conservative_field = torch.zeros_like(x, device=x.device) # device=x.device
        solenoidal_field = torch.zeros_like(x, device=x.device) # device=x.device

        if self.field_type != 'solenoidal':
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0] # gradients for conservative field
            # Create eye on the same device whre self.M_buffer lives.
            eye = torch.eye(*self.M_buffer.shape, device=self.M_buffer.device)
            conservative_field = dF1 @ eye

        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0] # gradients for solenoidal field
            # Use self.M_buffer.
            solenoidal_field = dF2 @ self.M_buffer.t()

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    def permutation_tensor(self, n):
        # Returns a torch tensor and is processed at __init__.
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            M = torch.ones(n,n)
            M *= 1 - torch.eye(n)
            M[::2] *= -1
            M[:,::2] *= -1
            for i in range(n):
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M


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
        print(f"Error occured while saving the model: {e}")


def get_hnn2_args():
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
    parser.add_argument('--device', default='cpu', type=str, help='Which context? CPU or GPU')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def get_hnn2_model(args):
    from hamiltonian_nn.nn_models import MLP
    output_dim = 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN2(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type) 
    model.to(args.device)
    return model

def hnn2_train():
  torch.set_grad_enabled(True)
  
  args = get_hnn2_args()
  args.verbose = True
  print('args.device:', args.device)
  if args.device != 'cpu':
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('args.device:', args.device)

  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  model = get_hnn2_model(args)

  device = args.device

  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0)

  # arrange data
  data = get_dataset(args.name, args.save_dir, verbose=True)

   # Try torch.tensor() with device (e.g., 'cuda'), instead of torch.Tensor().
  x = torch.tensor( data['coords'], requires_grad=True, dtype=torch.float32, device=device)
  test_x = torch.tensor( data['test_coords'], requires_grad=True, dtype=torch.float32, device=device)
  dxdt = torch.tensor(data['dcoords'], dtype=torch.float32, device=device) 
  test_dxdt = torch.tensor(data['test_dcoords'], dtype=torch.float32, device=device) 

  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):

    # train step
    ixs = torch.randperm(x.shape[0])[:args.batch_size] # original code
    ixs_dev = ixs.to(device) # Assigne this to the same device (e.g., 'cuda').
    dxdt_hat = model.time_derivative(x[ixs_dev])
    noise = args.input_noise * torch.randn(*x[ixs].shape) # add noise, maybe
    dxdt_hat += noise.to(device) # This one is also casted onto the same device (e.g., 'cuda')
    loss = L2_loss(dxdt[ixs_dev], dxdt_hat)
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
    optim.step() ; optim.zero_grad()

    # run test data
    test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
    test_ixs_dev = test_ixs.to(device) # Assigne this to the same device (e.g., 'cuda').
    test_dxdt_hat = model.time_derivative(test_x[test_ixs_dev])
    noise = args.input_noise * torch.randn(*test_x[test_ixs].shape) # add noise, maybe
    test_dxdt_hat += noise.to(device) # This one is also casted onto the same device (e.g., 'cuda')
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
    args = get_hnn2_args()
    hnn2_model, stats = hnn2_train()

    save_dir = THIS_DIR + "/weights"
    if args.device == 'cpu':
      save_model_weights(hnn2_model, save_dir,"danieljh_hnn2_2body_cpu_trained_separately_on_terminal.pth")
    else:
      save_model_weights(hnn2_model, save_dir,"danieljh_hnn2_2body_gpu_trained_separately_on_terminal.pth")