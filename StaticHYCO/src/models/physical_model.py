from flax import nnx
import jax
import jax.numpy as jnp
import jax.scipy as jsp

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.finite_element_method import *


class PhysicalModel(nnx.Module):
    def __init__(self, domain: tuple, N: int, parameters: jnp.ndarray, training: bool, forcing_func: Callable, kappa_func: Callable, eta_func: Callable, rngs: nnx.Rngs):
        self.domain = domain
        self.N = N
        self.parameters = nnx.Param(parameters)
        self.training = training
        self.forcing_func = forcing_func
        self.kappa_func = kappa_func
        self.eta_func = eta_func
        self.Mh_flag = False if eta_func is None else True
        self.mesh = self._create_mesh()
        self.fem = self._assemble_fem(self.mesh)
        self.solution = self._solve_pde(self.mesh, self.fem, self.Mh_flag)
        self.changed = False
        self.training = training
        

    def __call__(self, x: float, y: float) -> float:
        
        self.fem = self._assemble_fem(self.mesh)
        self.solution = self._solve_pde(self.mesh, self.fem, self.Mh_flag)
        self.changed = False
        pts = jnp.array([x, y]).reshape(1, 2)
        # Create the interpolator on the fly so that no non‑array object is stored.
        interp = jsp.interpolate.RegularGridInterpolator(
            (self.mesh["grid_x"].value, self.mesh["grid_y"].value),
            self.solution.value.reshape((self.N + 1, self.N + 1)),
            bounds_error=False,
            fill_value=None,
        )
        return interp(pts)
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def _create_mesh(self) -> dict:
        """
        Create and return a dictionary with the mesh nodes and grid.
        """
        nodes, _, grid_x, grid_y = generate_mesh(self.domain, self.N)
        interior_indices = get_interior_indices(nodes, domain=self.domain)
        return {
            "nodes": nnx.Variable(nodes),
            "grid_x": nnx.Variable(grid_x),
            "grid_y": nnx.Variable(grid_y),
            "interior_indices": nnx.Variable(interior_indices),
        }
    
    def _assemble_fem(self, mesh) -> dict:
        """
        Assemble and return a dictionary with the mesh and FEM matrices.
        The stiffness matrix is built with the (possibly dynamic) coefficient kappa.
        """
        interior_indices = mesh["interior_indices"]
        # Pass the coefficient function for kappa into the stiffness matrix assembly.
        if self.eta_func is not None:
            M = assemble_mass_matrix_2d(self.domain, self.N, eta=self.eta)
            M_in = M[jnp.ix_(interior_indices, interior_indices)]
        else:
            M_in = None
        
        K = assemble_stiffness_matrix_2d(self.domain, self.N, kappa=self.kappa)
        K_in = K[jnp.ix_(interior_indices, interior_indices)]
        F = assemble_load_vector_2d(self.domain, self.N, self.f_func)
        F_in = F[jnp.ix_(interior_indices)]
        
        return {
            "K_in": nnx.Variable(K_in),
            "M_in": nnx.Variable(M_in),
            "F_in": nnx.Variable(F_in),
        }
    
    def _solve_pde(self, mesh: dict, fem: dict, flag: bool) -> jnp.ndarray:
        """
        Solve the PDE using the cached FEM matrices.
        """
        nodes = jax.lax.stop_gradient(mesh["nodes"].value)
        interior_indices = jax.lax.stop_gradient(mesh["interior_indices"].value)
        K_in = fem["K_in"].value
        M_in = fem["M_in"].value
        F_in = fem["F_in"].value
        if flag:
            @jax.jit
            def solve_inner(K_in, M_in, F_in, nodes, interior_indices, flag):
                u_in = jsp.linalg.solve(K_in + M_in, F_in)
                u = jnp.zeros(nodes.shape[0], dtype=jnp.float32)
                u = u.at[interior_indices].set(u_in)
                return u
        else: 
            @jax.jit
            def solve_inner(K_in, M_in, F_in, nodes, interior_indices, flag):
                u_in = jsp.linalg.solve(K_in, F_in)
                u = jnp.zeros(nodes.shape[0], dtype=jnp.float32)

                u = u.at[interior_indices].set(u_in)
                return u


        return nnx.Variable(solve_inner(K_in, M_in, F_in, nodes, interior_indices, flag))
    
    def kappa(self, x, y):
        """
        Compute the coefficient kappa at position (x,y) based on model parameters.
        If no function is provided, default to 1.0.
        """
        if self.kappa_func is not None:
            return self.kappa_func(self.parameters, x, y)
        return 1.0

    def eta(self, x, y):
        """
        Compute the coefficient eta at position (x,y) based on model parameters.
        If no function is provided, default to 0.0.
        For now, eta is not used in the model since it is the Poisson equation.
        """
        if self.eta_func is not None:
            return self.eta_func(self.parameters, x, y)
        return 0.0
    
    def f_func(self, x, y):
        """
        Compute the forcing function at position (x,y) based on model parameters.
        If no function is provided, default to 0.0.
        """
        try:
            return self.forcing_func(self.parameters, x, y)
        except:
            return self.forcing_func(x, y)
        



        
    
