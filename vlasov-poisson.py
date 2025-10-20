import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class VlasovPoissonSolver:
    """
    1D Vlasov-Poisson solver using semi-Lagrangian method with JAX

    Equations:
    ∂f/∂t + v·∂f/∂x + E·∂f/∂v = 0  (Vlasov)
    ∂E/∂x = ρ - 1                   (Poisson, assuming neutralizing background)
    ρ = ∫ f dv                       (charge density)
    """

    def __init__(self, nx=64, nv=64, x_max=4*jnp.pi, v_max=6.0):
        self.nx = nx
        self.nv = nv
        self.x_max = x_max
        self.v_max = v_max

        # Spatial grid
        self.x = jnp.linspace(0, x_max, nx, endpoint=False)
        self.dx = x_max / nx

        # Velocity grid
        self.v = jnp.linspace(-v_max, v_max, nv)
        self.dv = 2 * v_max / nv

        # Create meshgrid for initial conditions
        self.X, self.V = jnp.meshgrid(self.x, self.v, indexing='ij')

    def initial_condition(self, mode=1, amplitude=0.05, v_thermal=1.0):
        """Landau damping initial condition"""
        # Maxwellian in velocity
        f0 = jnp.exp(-self.V**2 / (2 * v_thermal**2)) / jnp.sqrt(2 * jnp.pi * v_thermal**2)

        # Spatial perturbation (Landau damping)
        perturbation = 1.0 + amplitude * jnp.cos(2 * jnp.pi * mode * self.X / self.x_max)

        f = f0 * perturbation
        return f

    @partial(jit, static_argnums=(0,))
    def compute_rho(self, f):
        """Compute charge density by integrating over velocity"""
        return jnp.trapezoid(f, dx=self.dv, axis=1)

    @partial(jit, static_argnums=(0,))
    def solve_poisson(self, rho):
        """Solve Poisson equation using FFT (periodic boundary)"""
        # FFT of charge density (subtract mean for periodic BC)
        rho_mean = jnp.mean(rho)
        rho_hat = jnp.fft.fft(rho - rho_mean)

        # Wave numbers
        k = 2 * jnp.pi * jnp.fft.fftfreq(self.nx, d=self.dx)

        # Solve in Fourier space: E_hat = -i * rho_hat / k
        # Handle k=0 mode separately
        E_hat = jnp.where(k != 0, -1j * rho_hat / k, 0.0)

        # Transform back
        E = jnp.real(jnp.fft.ifft(E_hat))
        return E

    @partial(jit, static_argnums=(0,))
    def cubic_spline_interp_periodic(self, x_new, x_old, y_old):
        """
        Cubic spline interpolation with periodic boundary conditions
        Solved using FFT via circulant matrix approach
        """
        n = len(x_old)
        h = x_old[1] - x_old[0]  # Assume uniform grid

        # Step 1: Solve for second derivatives using FFT
        # For periodic cubic splines: M @ d2y = rhs
        # where M is circulant with first row [4, 1, 0, ..., 0, 1]

        # Build right-hand side: 6 * second finite differences
        y_ext = jnp.concatenate([y_old, y_old[:1]])  # Periodic extension
        rhs = 6.0 / h**2 * (jnp.roll(y_old, -1) - 2 * y_old + jnp.roll(y_old, 1))

        # Solve using FFT (circulant matrix eigenvalues)
        # Eigenvalues of circulant matrix [4, 1, 0, ..., 0, 1]
        k = jnp.arange(n)
        lambda_k = 4 + 2 * jnp.cos(2 * jnp.pi * k / n)

        # Solve in Fourier space
        rhs_hat = jnp.fft.fft(rhs)
        d2y_hat = rhs_hat / lambda_k
        d2y = jnp.real(jnp.fft.ifft(d2y_hat))

        # Step 2: Evaluate spline at new points
        # Wrap x_new to [0, x_max)
        x_wrapped = x_new % self.x_max

        # Find interval for each point
        idx = jnp.floor(x_wrapped / h).astype(int) % n

        # Local coordinate in interval [0, h]
        xi = x_wrapped - idx * h

        # Get values and second derivatives at interval endpoints
        y_i = y_old[idx]
        y_ip1 = jnp.roll(y_old, -1)[idx]
        d2y_i = d2y[idx]
        d2y_ip1 = jnp.roll(d2y, -1)[idx]

        # Cubic spline formula
        A = (h - xi) / h
        B = xi / h
        C = (A**3 - A) * h**2 / 6
        D = (B**3 - B) * h**2 / 6

        result = A * y_i + B * y_ip1 + C * d2y_i + D * d2y_ip1

        return result

    @partial(jit, static_argnums=(0,))
    def advect_x(self, f, dt):
        """Advect in x-direction (free streaming) with cubic spline FFT interpolation"""
        # For each velocity, compute departure points
        x_dep = (self.x[:, None] - self.v[None, :] * dt) % self.x_max

        # Cubic spline interpolation for each velocity slice
        def interp_cubic(x_new, f_slice):
            return self.cubic_spline_interp_periodic(x_new, self.x, f_slice)

        # Use vmap to vectorize over velocity dimension
        f_new = jax.vmap(lambda j: interp_cubic(x_dep[:, j], f[:, j]), out_axes=1)(jnp.arange(self.nv))

        return f_new

    @partial(jit, static_argnums=(0,))
    def advect_v(self, f, E, dt):
        """Advect in v-direction (acceleration) with cubic spline FFT interpolation"""
        # For each position, compute departure points in velocity
        v_dep = self.v[None, :] - E[:, None] * dt

        # Clip to velocity boundaries and treat as periodic
        # Map [-v_max, v_max] to periodic domain
        v_dep_wrapped = ((v_dep + self.v_max) % (2 * self.v_max)) - self.v_max

        # Cubic spline interpolation over spatial dimension
        def interp_cubic_v(v_new, f_slice):
            # Temporarily treat velocity as periodic for interpolation
            v_periodic = self.v  # Already uniform grid
            # Map v_new to [0, 2*v_max) for periodic treatment
            v_new_shifted = v_new + self.v_max
            v_grid_shifted = self.v + self.v_max

            return self.cubic_spline_interp_periodic(v_new_shifted, v_grid_shifted, f_slice)

        # Use vmap to vectorize over spatial dimension
        f_new = jax.vmap(lambda i: interp_cubic_v(v_dep_wrapped[i, :], f[i, :]), out_axes=0)(jnp.arange(self.nx))

        return f_new

    @partial(jit, static_argnums=(0,))
    def step(self, f, dt):
        """Single time step using Strang splitting"""
        # Split step: advect x for dt/2, then v for dt, then x for dt/2
        f = self.advect_x(f, dt/2)

        rho = self.compute_rho(f)
        E = self.solve_poisson(rho)

        f = self.advect_v(f, E, dt)
        f = self.advect_x(f, dt/2)

        return f, E, rho

    @partial(jit, static_argnums=(0, 2))
    def step_fn(self, carry, step_idx, dt):
        """Step function for jax.lax.scan"""
        f = carry
        f, E, rho = self.step(f, dt)
        return f, (f, E, rho)

    def simulate(self, T_final=50.0, dt=0.1, save_every=10):
        """Run simulation using jax.lax.scan for efficiency"""
        f = self.initial_condition()

        n_steps = int(T_final / dt)
        n_saves = n_steps // save_every

        # Function to run save_every steps
        def multi_step(carry, _):
            f = carry
            # Run save_every steps
            for _ in range(save_every):
                f, E, rho = self.step(f, dt)
            return f, (f, E, rho)

        # Use scan to iterate
        print(f"Running {n_steps} steps with JAX scan...")
        final_f, outputs = lax.scan(multi_step, f, None, length=n_saves)

        snapshots, E_history, rho_history = outputs
        times = np.arange(n_saves) * save_every * dt

        # Convert to numpy for plotting
        snapshots = np.array(snapshots)
        E_history = np.array(E_history)

        # Print final statistics
        final_energy = np.sum(E_history[-1]**2) * self.dx
        print(f"Simulation complete. Final E energy: {final_energy:.6f}")

        return times, snapshots, E_history

# Example usage
# Create solver
solver = VlasovPoissonSolver(nx=128, nv=128)

# Run simulation
print("Running Vlasov-Poisson simulation...")
times, snapshots, E_history = solver.simulate(T_final=50.0, dt=0.1, save_every=1)

# Plot electric field norm over time
fig, ax = plt.subplots(figsize=(10, 6))
E_norms = [np.sqrt(np.sum(E**2) * solver.dx) for E in E_history]
ax.semilogy(times, E_norms, 'b-', linewidth=2, label='Numerical')

# Add theoretical Landau damping curve
# For mode k = 2π/L_x with L_x = 4π, k = 0.5
# With v_th = 1.0, λ_D = 1.0, k*λ_D = 0.5
k = 2 * np.pi / solver.x_max
v_th = 1.0
k_lambda_D = k * v_th  # Assuming ω_p = 1
# Landau damping rate
gamma = np.sqrt(np.pi / 8) * np.exp(-1/(2*k_lambda_D**2) - 3/2) / (k_lambda_D**3)

# Theoretical curve: E(t) = E(0) * exp(-γ*t)
E0_theory = E_norms[0]
E_theory = E0_theory * np.exp(-gamma * np.array(times))
ax.semilogy(times, E_theory, 'r--', linewidth=2, label=f'Theory (γ={gamma:.4f})')

ax.set_xlabel('Time')
ax.set_ylabel('||E||_2 (L2 norm)')
ax.set_title('Electric Field Norm Evolution (Landau Damping)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('electric_field_norm.png', dpi=150, bbox_inches='tight')
print(f"Theoretical damping rate: γ = {gamma:.4f}")
print("Saved electric field norm to electric_field_norm.png")

plt.show()

