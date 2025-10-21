"""Gray-Scott PINN module (clean reconstruction after corruption).

Implements:
 - Data loading & scaling
 - GrayScott PINN with correct PDE residual sign
 - Training with data, physics, IC, optional periodic BC losses
 - Optional diffusion parameter learning (Du,Dv)
 - Comprehensive error tracking based on ERROR folder patterns
 - History tracking & simple saving helpers
"""
import os, time, pickle, sys
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import error tracking and data loading from same package
from shared.error_tracking import ErrorTracker
from shared.data_loader import load_tuple_data

# -------------------- Utility --------------------
class SimpleScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.fitted = False
    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        s = X.std(axis=0)
        s[s==0] = 1.0
        self.scale_ = s
        self.fitted = True
        return self
    def transform(self, X):
        if not self.fitted: raise RuntimeError("Scaler not fitted")
        X = np.asarray(X)
        return (X - self.mean_) / self.scale_
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    def inverse_transform(self, X):
        if not self.fitted: raise RuntimeError("Scaler not fitted")
        X = np.asarray(X)
        return X * self.scale_ + self.mean_

# -------------------- Network --------------------
class GrayScottNeuralNetwork(nn.Module):
    def __init__(self, num_hidden=128, num_layers=4, output_dim=2):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(3,num_hidden), nn.Tanh()]
        for _ in range(num_layers-1):
            layers += [nn.Linear(num_hidden,num_hidden), nn.Tanh()]
        layers.append(nn.Linear(num_hidden, output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    def forward(self,x): return self.net(x)

# -------------------- PINN Model --------------------
class GrayScottPINN(nn.Module):
    def __init__(self, num_hidden=128, num_layers=4, learn_diffusion=False, true_params=None,
                 initial_Du: float | None = None, initial_Dv: float | None = None):
        super().__init__()
        self.learn_diffusion = learn_diffusion
        self.true_params = true_params or {}
        self.network = GrayScottNeuralNetwork(num_hidden, num_layers, 2)
    # Note: Input normalization chain rule already yields physical derivatives w.r.t. original coords;
    # no additional PDE rescaling is mathematically required.
        if learn_diffusion:
            # Choose initialization priority: provided initial_* > true_params > defaults
            Du0 = initial_Du if initial_Du is not None else self.true_params.get('Du', self.true_params.get('D_u', 0.2))
            Dv0 = initial_Dv if initial_Dv is not None else self.true_params.get('Dv', self.true_params.get('D_v', 0.08))
            Du0 = float(Du0); Dv0 = float(Dv0)
            if Du0 <= 0 or Dv0 <= 0:
                raise ValueError("Initial diffusion coefficients must be positive.")
            self.log_Du = nn.Parameter(torch.log(torch.tensor(Du0)))
            self.log_Dv = nn.Parameter(torch.log(torch.tensor(Dv0)))
        else:
            self.register_buffer('Du_fixed', torch.tensor(float(self.true_params.get('Du', self.true_params.get('D_u',0.2)))))
            self.register_buffer('Dv_fixed', torch.tensor(float(self.true_params.get('Dv', self.true_params.get('D_v',0.08)))))
        self.register_buffer('f', torch.tensor(float(self.true_params.get('f',0.018))))
        self.register_buffer('k', torch.tensor(float(self.true_params.get('k',0.051))))
    @property
    def Du(self): return torch.exp(self.log_Du) if self.learn_diffusion else self.Du_fixed
    @property
    def Dv(self): return torch.exp(self.log_Dv) if self.learn_diffusion else self.Dv_fixed
    def forward(self,x): return self.network(x)
    def get_parameters(self):
        return { 'Du': self.Du.item(), 'Dv': self.Dv.item(), 'f': self.f.item(), 'k': self.k.item() }
    def physics_loss(self, x,y,t,input_scaler,output_scaler):
        """Compute PDE residual loss using only model autograd (no FD blending)."""
        coords = torch.cat([x,y,t],dim=1).requires_grad_(True)
        in_mean = torch.as_tensor(input_scaler.mean_, dtype=torch.float32, device=coords.device)
        in_scale= torch.as_tensor(input_scaler.scale_, dtype=torch.float32, device=coords.device)
        coords_n = (coords - in_mean)/in_scale
        out_n = self(coords_n)
        out_mean = torch.as_tensor(output_scaler.mean_, dtype=torch.float32, device=coords.device)
        out_scale= torch.as_tensor(output_scaler.scale_, dtype=torch.float32, device=coords.device)
        out = out_n * out_scale + out_mean
        u,v = out[:, :1], out[:,1:2]
        grads_u = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x,u_y,u_t = grads_u[:,0:1], grads_u[:,1:2], grads_u[:,2:3]
        grads_v = torch.autograd.grad(v, coords, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_x,v_y,v_t = grads_v[:,0:1], grads_v[:,1:2], grads_v[:,2:3]
        u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:,0:1]
        u_yy = torch.autograd.grad(u_y, coords, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:,1:2]
        v_xx = torch.autograd.grad(v_x, coords, torch.ones_like(v_x), create_graph=True, retain_graph=True)[0][:,0:1]
        v_yy = torch.autograd.grad(v_y, coords, torch.ones_like(v_y), create_graph=True, retain_graph=True)[0][:,1:2]
        lap_u = u_xx + u_yy
        lap_v = v_xx + v_yy
        Du,Dv,f,k = self.Du, self.Dv, self.f, self.k
        f_u = u_t - Du*lap_u + u*v.pow(2) - f*(1-u)
        f_v = v_t - Dv*lap_v - u*v.pow(2) + (f+k)*v
        return f_u.pow(2).mean() + f_v.pow(2).mean()

# -------------------- Trainer --------------------
class PINNTrainer:
    def __init__(self, data_dir: str, params: Dict[str,Any]):
        self.params = params
        self.device = 'cuda' if params.get('use_gpu',True) and torch.cuda.is_available() else 'cpu'
        seed = params.get('seed',42)
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        self.data_dir = data_dir
        self.scalers: Dict[str,SimpleScaler] = {}
        self._load_and_prepare_data()
        self._create_model()
        
        # Initialize comprehensive error tracking
        self.error_tracker = ErrorTracker(
            true_params=self.true_params,
            save_dir=params.get('save_dir', 'results')
        )
    def _load_and_prepare_data(self):
        d = load_tuple_data(self.data_dir)
        self.true_params = d['parameters']
        tuples = d['data_tuples']
        X_all = tuples[:,2:5]; y_all = tuples[:,0:2]
        total = X_all.shape[0]
        n_train = min(self.params.get('num_training_points', total), total)
        idx = np.random.choice(total, size=n_train, replace=False)
        self.X_data = X_all[idx]; self.y_data = y_all[idx]
        self.X_data_total = X_all; self.y_data_total = y_all
        in_scaler = SimpleScaler().fit(self.X_data)
        out_scaler= SimpleScaler().fit(self.y_data)
        self.X_data_norm = in_scaler.transform(self.X_data)
        self.y_data_norm = out_scaler.transform(self.y_data)
        self.X_data_total_norm = in_scaler.transform(self.X_data_total)
        self.y_data_total_norm = out_scaler.transform(self.y_data_total)
        self.scalers={'input': in_scaler, 'output': out_scaler}
        self.n_physics_per_epoch = self.params.get('num_physics_points', 1000)
        self.domain_ranges = {
            'x': (self.X_data[:,0].min(), self.X_data[:,0].max()),
            'y': (self.X_data[:,1].min(), self.X_data[:,1].max()),
            't': (self.X_data[:,2].min(), self.X_data[:,2].max()),
        }
        # Derive (optional) full initial condition grid
        self._load_full_initial_condition()
    def _load_full_initial_condition(self):
        """Derive initial condition grid from data_tuples at earliest time (if complete grid)."""
        self.X_ic_full = None; self.y_ic_full = None
        tuples_full = load_tuple_data(self.data_dir)['data_tuples']
        tvals = tuples_full[:,4]
        t0 = tvals.min()
        mask = np.isclose(tvals, t0)
        ic_rows = tuples_full[mask]
        if ic_rows.size == 0:
            return
        x_unique = np.unique(ic_rows[:,2]); y_unique = np.unique(ic_rows[:,3])
        if ic_rows.shape[0] == x_unique.size * y_unique.size:
            Xg, Yg = np.meshgrid(x_unique, y_unique)
            coords_grid = np.column_stack([Xg.ravel(), Yg.ravel()])
            uv_map = {(row[2], row[3]): (row[0], row[1]) for row in ic_rows}
            uv_arr = np.array([uv_map[(xv,yv)] for xv,yv in coords_grid])
            self.X_ic_full = np.column_stack([coords_grid[:,0], coords_grid[:,1], np.full(coords_grid.shape[0], t0)])
            self.y_ic_full = uv_arr
    def _create_model(self):
        self.model = GrayScottPINN(
            num_hidden=self.params['num_hidden'],
            num_layers=self.params['num_layers'],
            learn_diffusion=self.params.get('learn_diffusion',False),
            true_params=self.true_params,
            initial_Du=self.params.get('init_Du', None),
            initial_Dv=self.params.get('init_Dv', None)
        ).to(self.device)
    def _compute_solution_error(self):
        self.model.eval()
        with torch.no_grad():
            X = torch.as_tensor(self.X_data_total_norm, dtype=torch.float32, device=self.device)
            y_true_norm = torch.as_tensor(self.y_data_total_norm, dtype=torch.float32, device=self.device)
            y_pred_norm = self.model(X)
            mse_norm = torch.mean((y_pred_norm - y_true_norm)**2).item()
            out_scaler = self.scalers['output']
            y_pred = y_pred_norm.cpu().numpy()*out_scaler.scale_ + out_scaler.mean_
            y_true = self.y_data_total
            u_err = np.mean((y_pred[:,0]-y_true[:,0])**2)
            v_err = np.mean((y_pred[:,1]-y_true[:,1])**2)
            mse_phys = u_err + v_err
            u_rel = np.sqrt(u_err)/(np.std(y_true[:,0])+1e-8)
            v_rel = np.sqrt(v_err)/(np.std(y_true[:,1])+1e-8)
        self.model.train()
        return {'mse_normalized': mse_norm,'mse_physical': mse_phys,'u_mse': u_err,'v_mse': v_err,'u_relative_error': u_rel,'v_relative_error': v_rel}
    def train(self):
        crit = nn.MSELoss()
        net_params = [p for n,p in self.model.named_parameters() if n.startswith('network')]
        diff_params= [p for n,p in self.model.named_parameters() if n in ['log_Du','log_Dv']]
        # Optimizer param groups: apply weight decay ONLY to network weights, not diffusion parameters
        groups=[{'params': net_params, 'lr': self.params['learning_rate'], 'weight_decay': self.params.get('weight_decay',0.0)}]
        if diff_params and self.params.get('learn_diffusion',False):
            groups.append({'params': diff_params, 'lr': self.params.get('diffusion_learning_rate', self.params['learning_rate']*0.1), 'weight_decay': 0.0})
        optimizer = optim.Adam(groups)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=100)
        self.training_history={k:[] for k in ['train_loss','data_loss','physics_loss','ic_loss','tuple_data_loss','bc_loss','learning_rate']}
        self.training_history['parameters']={'Du':[],'Dv':[]}; self.training_history['solution_error']=[]
        X_data = torch.as_tensor(self.X_data_norm, dtype=torch.float32, device=self.device)
        y_data = torch.as_tensor(self.y_data_norm, dtype=torch.float32, device=self.device)
        ic_coords = torch.as_tensor(self.X_ic_full, dtype=torch.float32, device=self.device) if self.X_ic_full is not None else None
        ic_targets= torch.as_tensor(self.y_ic_full, dtype=torch.float32, device=self.device) if self.y_ic_full is not None else None
        lambda_physics = self.params.get('lambda_physics',1.0)
        lambda_ic = self.params.get('lambda_ic',0.0)
        lambda_bc = self.params.get('lambda_bc',0.0)
        phys_space = self.params.get('compute_data_loss_physical',False)
        start=time.time()
        # Warmup freeze for diffusion parameters
        warmup = self.params.get('diffusion_warmup_epochs', 0)
        disable_phys_warmup = self.params.get('disable_physics_during_warmup', True)
        diffusion_learn = self.params.get('learn_diffusion', False) and len(diff_params)>0
        if diffusion_learn and warmup>0:
            for p in diff_params: p.requires_grad = False
            print(f"[INFO] Freezing diffusion parameters for first {warmup} epochs (warmup).")
        for epoch in range(self.params['epochs']):
            if diffusion_learn and warmup>0 and epoch == warmup:
                mid_epochs = self.params.get('lbfgs_mid_epochs',0)
                keep_mid_frozen = self.params.get('lbfgs_mid_keep_diffusion_frozen', False)
                run_mid = mid_epochs > 0
                if run_mid and keep_mid_frozen:
                    print(f"[INFO] MID-LBFGS ({mid_epochs} epochs) with diffusion FROZEN at epoch {epoch}.")
                    self._run_lbfgs(crit, mid_epochs, tag='MID-LBFGS-FROZEN')
                # Unfreeze if configured (always, regardless of keep_mid_frozen)
                if self.params.get('unfreeze_diffusion_after_warmup', True):
                    for p in diff_params: p.requires_grad = True
                    print(f"[INFO] Unfreezing diffusion parameters at epoch {epoch}.")
                    # If we want a mid-LBFGS pass with diffusion trainable (when not explicitly kept frozen)
                    if run_mid and not keep_mid_frozen:
                        print(f"[INFO] MID-LBFGS ({mid_epochs} epochs) after unfreezing diffusion.")
                        self._run_lbfgs(crit, mid_epochs, tag='MID-LBFGS')
                else:
                    # User requested to keep diffusion frozen for entire training
                    print(f"[INFO] Diffusion parameters remain frozen after warmup (unfreeze disabled).")
            optimizer.zero_grad()
            if ic_coords is not None and lambda_ic>0:
                in_mean = torch.as_tensor(self.scalers['input'].mean_, dtype=torch.float32, device=self.device)
                in_scale= torch.as_tensor(self.scalers['input'].scale_, dtype=torch.float32, device=self.device)
                ic_norm = (ic_coords - in_mean)/in_scale
                ic_pred_n = self.model(ic_norm)
                out_mean= torch.as_tensor(self.scalers['output'].mean_, dtype=torch.float32, device=self.device)
                out_scale= torch.as_tensor(self.scalers['output'].scale_, dtype=torch.float32, device=self.device)
                ic_pred = ic_pred_n*out_scale + out_mean
                ic_loss = crit(ic_pred, ic_targets)
            else:
                ic_loss = torch.zeros(1, device=self.device)
            pred_n = self.model(X_data)
            if phys_space:
                out_mean= torch.as_tensor(self.scalers['output'].mean_, dtype=torch.float32, device=self.device)
                out_scale= torch.as_tensor(self.scalers['output'].scale_, dtype=torch.float32, device=self.device)
                pred = pred_n*out_scale + out_mean
                y_phys = y_data*out_scale + out_mean
                tuple_data_loss = crit(pred, y_phys)
            else:
                tuple_data_loss = crit(pred_n, y_data)
            data_loss = ic_loss + tuple_data_loss
            xr,yr,tr = self.domain_ranges['x'], self.domain_ranges['y'], self.domain_ranges['t']
            if diffusion_learn and warmup>0 and disable_phys_warmup and epoch < warmup:
                physics_loss = torch.zeros(1, device=self.device)
            else:
                b = self.n_physics_per_epoch
                x_phys = torch.rand(b,1,device=self.device)*(xr[1]-xr[0])+xr[0]
                y_phys = torch.rand(b,1,device=self.device)*(yr[1]-yr[0])+yr[0]
                t_phys = torch.rand(b,1,device=self.device)*(tr[1]-tr[0])+tr[0]
                physics_loss = self.model.physics_loss(
                    x_phys,y_phys,t_phys,
                    self.scalers['input'],self.scalers['output']
                )
            if lambda_bc>0:
                n_bc=min(128,b)
                t_bc = torch.rand(n_bc,1,device=self.device)*(tr[1]-tr[0])+tr[0]
                yb = torch.rand(n_bc,1,device=self.device)*(yr[1]-yr[0])+yr[0]
                xl = torch.full_like(yb, xr[0]); xr_ = torch.full_like(yb, xr[1])
                in_mean = torch.as_tensor(self.scalers['input'].mean_, dtype=torch.float32, device=self.device)
                in_scale= torch.as_tensor(self.scalers['input'].scale_, dtype=torch.float32, device=self.device)
                left_n = (torch.cat([xl,yb,t_bc],1)-in_mean)/in_scale
                right_n= (torch.cat([xr_,yb,t_bc],1)-in_mean)/in_scale
                bc_lr = crit(self.model(left_n), self.model(right_n))
                xb = torch.rand(n_bc,1,device=self.device)*(xr[1]-xr[0])+xr[0]
                ybottom = torch.full_like(xb, yr[0]); ytop = torch.full_like(xb, yr[1])
                bot_n = (torch.cat([xb,ybottom,t_bc],1)-in_mean)/in_scale
                top_n = (torch.cat([xb,ytop,t_bc],1)-in_mean)/in_scale
                bc_bt = crit(self.model(bot_n), self.model(top_n))
                bc_loss = 0.5*(bc_lr+bc_bt)
            else:
                bc_loss = torch.zeros(1,device=self.device)
            total = lambda_ic*ic_loss + tuple_data_loss + lambda_physics*physics_loss + lambda_bc*bc_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
            optimizer.step()
            if epoch> self.params['epochs']//3:
                scheduler.step(total.item())
            lr = optimizer.param_groups[0]['lr']
            self.training_history['train_loss'].append(total.item())
            self.training_history['data_loss'].append(data_loss.item())
            self.training_history['physics_loss'].append(physics_loss.item())
            self.training_history['ic_loss'].append(ic_loss.item())
            self.training_history['tuple_data_loss'].append(tuple_data_loss.item())
            self.training_history['bc_loss'].append(bc_loss.item())
            self.training_history['learning_rate'].append(lr)
            params_now = self.model.get_parameters()
            self.training_history['parameters']['Du'].append(params_now['Du'])
            self.training_history['parameters']['Dv'].append(params_now['Dv'])
            
            # Legacy error computation for backward compatibility
            err = self._compute_solution_error()
            self.training_history['solution_error'].append({'epoch': epoch,'metrics': err})
            
            # Comprehensive error tracking using ErrorTracker
            with torch.no_grad():
                X = torch.as_tensor(self.X_data_total_norm, dtype=torch.float32, device=self.device)
                y_true_norm = torch.as_tensor(self.y_data_total_norm, dtype=torch.float32, device=self.device)
                y_pred_norm = self.model(X)
                
                # Prepare loss components
                losses = {
                    'physics_loss': physics_loss.item(),
                    'data_loss': data_loss.item(),
                    'ic_loss': ic_loss.item(),
                    'bc_loss': bc_loss.item(),
                    'total_loss': total.item()
                }
                
                # Update comprehensive error tracking
                self.error_tracker.update_errors(
                    epoch=epoch,
                    y_pred=y_pred_norm,
                    y_true=y_true_norm,
                    learned_params=params_now,
                    scalers=self.scalers,
                    losses=losses
                )
            
            # Print comprehensive error summary periodically
            if epoch % 50 == 0 or epoch == self.params['epochs']-1:
                el = time.time()-start
                # Convert tensors to Python floats for safe formatting
                if self.params.get('learn_diffusion', False):
                    Du_val = self.model.Du.item()
                    Dv_val = self.model.Dv.item()
                    print(
                        f"Epoch {epoch:04d} | Total {total.item():.3e} "
                        f"Data {data_loss.item():.3e} Phys {physics_loss.item():.3e} "
                        f"IC {ic_loss.item():.3e} BC {bc_loss.item():.3e} LR {lr:.1e} "
                        f"Du {Du_val:.4e} Dv {Dv_val:.4e} t {el:.1f}s"
                    )
                else:
                    print(
                        f"Epoch {epoch:04d} | Total {total.item():.3e} "
                        f"Data {data_loss.item():.3e} Phys {physics_loss.item():.3e} "
                        f"IC {ic_loss.item():.3e} BC {bc_loss.item():.3e} LR {lr:.1e} t {el:.1f}s"
                    )
                
                # Print detailed error summary every 100 epochs
                if epoch % 100 == 0:
                    self.error_tracker.print_error_summary(epoch)
        print('Training complete.')
        if self.params.get('lbfgs_epochs',0) > 0:
            self._run_lbfgs(crit, self.params.get('lbfgs_epochs',0), tag='FINAL-LBFGS')

    def _run_lbfgs(self, crit, epochs: int, tag: str = 'LBFGS'):
        if epochs <=0:
            return
        print(f"Starting {tag} fine-tune for {epochs} epochs (full-batch style)...")
        xr,yr,tr = self.domain_ranges['x'], self.domain_ranges['y'], self.domain_ranges['t']
        b = self.n_physics_per_epoch
        rng = torch.Generator(device=self.device)
        rng.manual_seed(self.params.get('seed',42) + (1234 if 'FINAL' in tag else 5678))
        x_phys = torch.rand(b,1, generator=rng, device=self.device)*(xr[1]-xr[0])+xr[0]
        y_phys = torch.rand(b,1, generator=rng, device=self.device)*(yr[1]-yr[0])+yr[0]
        t_phys = torch.rand(b,1, generator=rng, device=self.device)*(tr[1]-tr[0])+tr[0]
        X_data = torch.as_tensor(self.X_data_norm, dtype=torch.float32, device=self.device)
        y_data = torch.as_tensor(self.y_data_norm, dtype=torch.float32, device=self.device)
        ic_coords = torch.as_tensor(self.X_ic_full, dtype=torch.float32, device=self.device) if self.X_ic_full is not None else None
        ic_targets= torch.as_tensor(self.y_ic_full, dtype=torch.float32, device=self.device) if self.X_ic_full is not None else None
        lambda_physics = self.params.get('lambda_physics',1.0)
        lambda_ic = self.params.get('lambda_ic',0.0)
        lambda_bc = self.params.get('lambda_bc',0.0)
        phys_space = self.params.get('compute_data_loss_physical',False)
        params_for_lbfgs = [p for p in self.model.parameters() if p.requires_grad]
        lbfgs = optim.LBFGS(params_for_lbfgs, lr=self.params.get('lbfgs_lr',1.0), max_iter=20, history_size=50, line_search_fn='strong_wolfe')
        start=time.time()
        for ep in range(epochs):
            def closure():
                lbfgs.zero_grad()
                if ic_coords is not None and lambda_ic>0:
                    in_mean = torch.as_tensor(self.scalers['input'].mean_, dtype=torch.float32, device=self.device)
                    in_scale= torch.as_tensor(self.scalers['input'].scale_, dtype=torch.float32, device=self.device)
                    ic_norm = (ic_coords - in_mean)/in_scale
                    ic_pred_n = self.model(ic_norm)
                    out_mean= torch.as_tensor(self.scalers['output'].mean_, dtype=torch.float32, device=self.device)
                    out_scale= torch.as_tensor(self.scalers['output'].scale_, dtype=torch.float32, device=self.device)
                    ic_pred = ic_pred_n*out_scale + out_mean
                    ic_loss = crit(ic_pred, ic_targets)
                else:
                    ic_loss = torch.zeros(1, device=self.device)
                pred_n = self.model(X_data)
                if phys_space:
                    out_mean= torch.as_tensor(self.scalers['output'].mean_, dtype=torch.float32, device=self.device)
                    out_scale= torch.as_tensor(self.scalers['output'].scale_, dtype=torch.float32, device=self.device)
                    pred = pred_n*out_scale + out_mean
                    y_data_phys = y_data*out_scale + out_mean
                    tuple_data_loss = crit(pred, y_data_phys)
                else:
                    tuple_data_loss = crit(pred_n, y_data)
                physics_loss = self.model.physics_loss(
                    x_phys,y_phys,t_phys,
                    self.scalers['input'],self.scalers['output']
                )
                if lambda_bc>0:
                    n_bc=min(128,b)
                    t_bc = torch.rand(n_bc,1,device=self.device)*(tr[1]-tr[0])+tr[0]
                    yb = torch.rand(n_bc,1,device=self.device)*(yr[1]-yr[0])+yr[0]
                    xl = torch.full_like(yb, xr[0]); xr_ = torch.full_like(yb, xr[1])
                    in_mean = torch.as_tensor(self.scalers['input'].mean_, dtype=torch.float32, device=self.device)
                    in_scale= torch.as_tensor(self.scalers['input'].scale_, dtype=torch.float32, device=self.device)
                    left_n = (torch.cat([xl,yb,t_bc],1)-in_mean)/in_scale
                    right_n= (torch.cat([xr_,yb,t_bc],1)-in_mean)/in_scale
                    bc_lr = crit(self.model(left_n), self.model(right_n))
                    xb = torch.rand(n_bc,1,device=self.device)*(xr[1]-xr[0])+xr[0]
                    ybottom = torch.full_like(xb, yr[0]); ytop = torch.full_like(xb, yr[1])
                    bot_n = (torch.cat([xb,ybottom,t_bc],1)-in_mean)/in_scale
                    top_n = (torch.cat([xb,ytop,t_bc],1)-in_mean)/in_scale
                    bc_bt = crit(self.model(bot_n), self.model(top_n))
                    bc_loss = 0.5*(bc_lr+bc_bt)
                else:
                    bc_loss = torch.zeros(1,device=self.device)
                total = lambda_ic*ic_loss + tuple_data_loss + lambda_physics*physics_loss + lambda_bc*bc_loss
                total.backward()
                return total
            loss = lbfgs.step(closure)
            if ep % 5 ==0 or ep==epochs-1:
                el = time.time()-start
                if self.params.get('learn_diffusion', False):
                    print(f"{tag} {ep:03d}/{epochs} loss {loss.item():.3e} Du {self.model.Du.item():.4e} Dv {self.model.Dv.item():.4e} t {el:.1f}s")
                else:
                    print(f"{tag} {ep:03d}/{epochs} loss {loss.item():.3e} t {el:.1f}s")
        print(f"{tag} fine-tune complete.")

    def save(self, save_dir: str):
        if self.params.get('timestamped_save', False):
            import datetime
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = f"{save_dir}_{ts}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir,'model_state.pth'))
        with open(os.path.join(save_dir,'scalers.pkl'),'wb') as f: pickle.dump(self.scalers,f)
        with open(os.path.join(save_dir,'training_history.pkl'),'wb') as f: pickle.dump(self.training_history,f)
        # Persist experiment parameters / metadata for later reload & plotting
        try:
            metadata = {
                'params': self.params,
                'true_params': self.true_params,
                'timestamp': time.time(),
                'learn_diffusion': self.params.get('learn_diffusion', False),
                'architecture': {
                    'num_hidden': self.params.get('num_hidden'),
                    'num_layers': self.params.get('num_layers')
                }
            }
            with open(os.path.join(save_dir,'metadata.pkl'),'wb') as f: pickle.dump(metadata,f)
        except Exception as e:
            print(f"Warning: could not save metadata.pkl ({e})")
        def np_save(name, arr): np.save(os.path.join(save_dir,name), np.array(arr))
        np_save('train_loss.npy', self.training_history['train_loss'])
        np_save('physics_loss.npy', self.training_history['physics_loss'])
        np_save('data_loss.npy', self.training_history['data_loss'])
        np_save('Du_evolution.npy', self.training_history['parameters']['Du'])
        np_save('Dv_evolution.npy', self.training_history['parameters']['Dv'])
        np_save('learning_rate.npy', self.training_history['learning_rate'])
        if self.training_history['solution_error']:
            mse_phys = [e['metrics']['mse_physical'] for e in self.training_history['solution_error']]
            np_save('solution_mse_physical.npy', mse_phys)
            
        # Save comprehensive error tracking based on ERROR folder patterns
        print("Saving comprehensive error tracking...")
        try:
            # Import datetime for filename generation
            from datetime import datetime
            
            # Save error tracking CSV (following ERROR folder naming convention)
            error_csv_path = self.error_tracker.save_errors_to_csv(
                filename=f"error_tracking_pinn_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
            )
            print(f"✓ Comprehensive error tracking saved to: {error_csv_path}")
            
            # Save final error summary
            final_errors = self.error_tracker.get_latest_errors()
            with open(os.path.join(save_dir, 'final_error_summary.txt'), 'w') as f:
                f.write("=== Final Error Summary ===\n")
                for key, value in final_errors.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.6e}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            print("✓ Final error summary saved")
            
        except Exception as e:
            print(f"Warning: Could not save comprehensive error tracking: {e}")
    
    # Add method to get error tracker for external access
    def get_error_tracker(self):
        """Return the error tracker for external analysis."""
        return self.error_tracker
    # Plotting removed (FD data no longer used).

    # ---------------- Plotting Utilities ----------------

# -------------------- High-level API --------------------
def train_pinn(data_dir: str, params: Dict[str,Any], experiment_name='gray_scott_pinn', base_save_dir='results'):
    trainer = PINNTrainer(data_dir, params)
    trainer.train()
    save_dir = os.path.join(base_save_dir, experiment_name)
    trainer.save(save_dir)
    # If requested and not already plotted inside save (timestamp variant), ensure plot
    if params.get('plot_after_training', False) and not params.get('timestamped_save', False):
        try:
            trainer.plot_solution_time_slices(save_dir=save_dir,
                                              num_times=params.get('num_plot_time_slices',3),
                                              specific_times=params.get('plot_times', None))
        except Exception as e:
            print(f"Post-training plotting failed: {e}")
    return os.path.join(save_dir,'model_state.pth')

