import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.experimental import sparse
from functools import partial
from typing import Callable, Tuple

def generate_mesh(domain: Tuple[float, float], N: int):
    """
    Generates a uniform mesh for a square domain.
    Args:
        domain: Tuple (a, b) defining the interval [a, b] for both x and y.
        N: Number of subdivisions along each axis.
    Returns:
        nodes: (nn, 2) array of node coordinates.
        elements: (ne, 3) array of triangle connectivity.
        x: 1D array of x-coordinates.
        y: 1D array of y-coordinates.
    """
    a, b = domain
    x = jnp.linspace(a, b, N + 1)
    y = jnp.linspace(a, b, N + 1)
    xv, yv = jnp.meshgrid(x, y, indexing='ij')
    nodes = jnp.column_stack((xv.ravel(), yv.ravel()))

    grid = jnp.arange((N + 1) ** 2).reshape(N + 1, N + 1)
    # Lower triangles and upper triangles combined
    t1 = jnp.stack([grid[:-1, :-1].ravel(), grid[1:, :-1].ravel(), grid[1:, 1:].ravel()], axis=1)
    t2 = jnp.stack([grid[:-1, :-1].ravel(), grid[1:, 1:].ravel(), grid[:-1, 1:].ravel()], axis=1)
    elements = jnp.concatenate([t1, t2], axis=0)
    return nodes, elements, x, y

@partial(jax.jit, static_argnames=("eta",))
def local_mass_matrix(coords: jnp.ndarray, area: float, 
                      eta: Callable[[float, float], float] = None) -> jnp.ndarray:
    """
    Computes the local mass matrix for a linear triangular element.
    Uses the formula: (area/12)*[[2,1,1],[1,2,1],[1,1,2]],
    optionally scaled by eta evaluated at the centroid.
    """
    Me = (area / 12.0) * jnp.array([[2, 1, 1],
                                    [1, 2, 1],
                                    [1, 1, 2]], dtype=jnp.float32)
    if eta is not None:
        x_c = jnp.mean(coords[:, 0])
        y_c = jnp.mean(coords[:, 1])
        Me *= eta(x_c, y_c)
    return Me

@partial(jax.jit, static_argnames=("kappa",))
def local_stiffness_matrix(coords: jnp.ndarray, area: float, 
                           kappa: Callable[[float, float], float] = None) -> jnp.ndarray:
    """
    Computes the local stiffness matrix for a linear triangular element.
    Formula: Ke[i,j] = (b_i*b_j + c_i*c_j) / (4*area),
    with b_i = y_{i+1} - y_{i+2} and c_i = x_{i+2} - x_{i+1} (cyclic ordering).
    """
    b = jnp.roll(coords[:, 1], -1) - jnp.roll(coords[:, 1], -2)
    c = jnp.roll(coords[:, 0], -2) - jnp.roll(coords[:, 0], -1)
    Ke = (jnp.outer(b, b) + jnp.outer(c, c)) / (4 * area)
    if kappa is not None:
        x_c = jnp.mean(coords[:, 0])
        y_c = jnp.mean(coords[:, 1])
        Ke *= kappa(x_c, y_c)
    return Ke

def assemble_matrix(domain: Tuple[float, float], N: int, local_matrix_func: Callable, coeff_func: Callable) -> jnp.ndarray:
    """
    Assembles the global matrix using a provided local matrix function.
    Args:
        domain: Tuple (a, b) defining the square domain.
        N: Number of subdivisions along each axis.
        local_matrix_func: Function that computes a 3x3 local matrix from (coords, area).
    Returns:
        Global assembled matrix as a (nn, nn) jnp.ndarray.
    """
    nodes, elements, _, _ = generate_mesh(domain, N)
    nn, ne = nodes.shape[0], elements.shape[0]
    a, b = domain
    h = (b - a) / N
    area = (h * h) / 2.0

    def body_fun(i, M):
        elem = elements[i]
        coords = nodes[elem, :]
        Loc = local_matrix_func(coords, area, coeff_func)
        def inner_body(j, M_inner):
            i_global = elem[j]
            def inner_body2(k, M_inner2):
                j_global = elem[k]
                return M_inner2.at[i_global, j_global].add(Loc[j, k])
            return jax.lax.fori_loop(0, 3, inner_body2, M_inner)
        return jax.lax.fori_loop(0, 3, inner_body, M)

    M = jax.lax.fori_loop(0, ne, body_fun, jnp.zeros((nn, nn), dtype=jnp.float32))
    return M

def assemble_mass_matrix_2d(domain: Tuple[float, float], N: int, eta: Callable) -> jnp.ndarray:
    """Assembles the global mass matrix for 2D problems."""
    return assemble_matrix(domain, N, local_mass_matrix, eta)

def assemble_stiffness_matrix_2d(domain: Tuple[float, float], N: int, kappa: Callable) -> jnp.ndarray:
    """Assembles the global stiffness matrix for 2D problems."""
    return assemble_matrix(domain, N, local_stiffness_matrix, kappa)

def assemble_load_vector_2d(domain: Tuple[float, float], N: int, 
                              f_func: Callable[[float, float], float]) -> jnp.ndarray:
    """
    Assembles the global load vector using vertex-based evaluation.
    Each element contributes (area/3)*f(x, y) at each vertex.
    """
    nodes, elements, _, _ = generate_mesh(domain, N)
    nn, ne = nodes.shape[0], elements.shape[0]
    a, b = domain
    h = (b - a) / N
    area = (h * h) / 2.0

    def body_fun(i, F):
        elem = elements[i]
        coords = nodes[elem, :]
        # Vectorized evaluation of f_func at each vertex of the element.
        fe_local = (area / 3.0) * jax.vmap(lambda x, y: f_func(x, y))(coords[:, 0], coords[:, 1])
        def inner_body(j, F_inner):
            return F_inner.at[elem[j]].add(fe_local[j])
        return jax.lax.fori_loop(0, 3, inner_body, F)

    F = jax.lax.fori_loop(0, ne, body_fun, jnp.zeros(nn, dtype=jnp.float32))
    return F

def get_interior_indices(nodes: jnp.ndarray, domain: Tuple[float, float], tol: float = 1e-6) -> jnp.ndarray:
    """
    Returns the indices of nodes that are interior to the square domain.
    Interior nodes satisfy: a + tol < x < b - tol and a + tol < y < b - tol.
    """
    a, b = domain
    mask = (nodes[:, 0] > (a + tol)) & (nodes[:, 0] < (b - tol)) & \
           (nodes[:, 1] > (a + tol)) & (nodes[:, 1] < (b - tol))
    return jnp.nonzero(mask)[0]

def reduce_system(matrix: jnp.ndarray, load_vector: jnp.ndarray, 
                  nodes: jnp.ndarray, domain: Tuple[float, float], tol: float = 1e-6
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Reduces the global system by applying zero Dirichlet boundary conditions.
    It extracts the interior nodes and returns the submatrix and subvector
    corresponding to the interior degrees of freedom, along with the indices.

    Args:
        matrix: Global assembled matrix (nn x nn).
        load_vector: Global load vector (nn,).
        nodes: Array of node coordinates (nn x 2).
        domain: Tuple (a, b) defining the square domain.
        tol: Tolerance used for identifying interior nodes.

    Returns:
        submatrix: Reduced matrix corresponding to interior nodes.
        subload: Reduced load vector for interior nodes.
        interior_indices: Indices of interior nodes.
    """
    interior_indices = get_interior_indices(nodes, domain, tol)
    submatrix = matrix[jnp.ix_(interior_indices, interior_indices)]
    subload = load_vector[interior_indices]
    return submatrix, subload, interior_indices