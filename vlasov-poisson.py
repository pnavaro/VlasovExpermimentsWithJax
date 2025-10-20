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
    def cubic_spline_periodic(self, x_new, x_old, y_old):
        """Cubic spline interpolation with periodic boundary conditions"""
        # Extend array for periodicity
        y_ext = jnp.concatenate([y_old[-1:], y_old, y_old[:1]])
        x_ext = jnp.concatenate([x_old[-1:] - self.x_max, x_old, x_old[:1] + self.x_max])

        # Find indices for interpolation
        idx = jnp.searchsorted(x_old, x_new % self.x_max)
        idx = jnp.clip(idx, 0, len(x_old) - 1)

        # Get local neighborhood for cubic interpolation (4 points)
        i = idx
        x0, x1 = x_ext[i], x_ext[i + 1]
        x_1, x2 = x_ext[i - 1 + 1], x_ext[i + 2]  # adjusted indexing
        y0, y1 = y_ext[i], y_ext[i + 1]
        y_1, y2 = y_ext[i - 1 + 1], y_ext[i + 2]

        # Normalize position in interval
        t = (x_new - x0) / (x1 - x0)

        # Catmull-Rom spline coefficients
        a = -0.5 * y_1 + 1.5 * y0 - 1.5 * y1 + 0.5 * y2
        b = y_1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2
        c = -0.5 * y_1 + 0.5 * y1
        d = y0

        return a * t**3 + b * t**2 + c * t + d

    @partial(jit, static_argnums=(0,))
    def advect_x(self, f, dt):
        """Advect in x-direction (free streaming) with cubic spline interpolation"""
        # For each velocity, compute departure points
        x_dep = (self.x[:, None] - self.v[None, :] * dt) % self.x_max

        # Cubic spline interpolation for each velocity slice
        def interp_cubic(x_new, f_slice):
            return self.cubic_spline_periodic(x_new, self.x, f_slice)

        # Use vmap to vectorize over velocity dimension
        f_new = jax.vmap(lambda j: interp_cubic(x_dep[:, j], f[:, j]), out_axes=1)(jnp.arange(self.nv))

        return f_new

    @partial(jit, static_argnums=(0,))
    def cubic_spline_bounded(self, v_new, v_old, y_old):
        """Cubic spline interpolation with clamped boundaries"""
        # Find indices for interpolation
        idx = jnp.searchsorted(v_old, v_new)
        idx = jnp.clip(idx, 1, len(v_old) - 2)

        # Get local neighborhood for cubic interpolation (4 points when possible)
        i = idx
        v0, v1 = v_old[i - 1], v_old[i]
        v_1 = jnp.where(i > 1, v_old[i - 2], v_old[i - 1] - (v_old[i] - v_old[i - 1]))
        v2 = jnp.where(i < len(v_old) - 1, v_old[i + 1], v_old[i] + (v_old[i] - v_old[i - 1]))

        y0, y1 = y_old[i - 1], y_old[i]
        y_1 = jnp.where(i > 1, y_old[i - 2], y_old[i - 1])
        y2 = jnp.where(i < len(v_old) - 1, y_old[i + 1], y_old[i])

        # Normalize position in interval
        t = (v_new - v0) / (v1 - v0 + 1e-10)

        # Catmull-Rom spline coefficients
        a = -0.5 * y_1 + 1.5 * y0 - 1.5 * y1 + 0.5 * y2
        b = y_1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2
        c = -0.5 * y_1 + 0.5 * y1
        d = y0

        return a * t**3 + b * t**2 + c * t + d

    @partial(jit, static_argnums=(0,))
    def advect_v(self, f, E, dt):
        """Advect in v-direction (acceleration) with cubic spline interpolation"""
        # For each position, compute departure points in velocity
        v_dep = self.v[None, :] - E[:, None] * dt

        # Clip to velocity boundaries
        v_dep = jnp.clip(v_dep, -self.v_max, self.v_max)

        # Cubic spline interpolation over spatial dimension
        def interp_cubic_v(v_new, f_slice):
            return self.cubic_spline_bounded(v_new, self.v, f_slice)

        # Use vmap to vectorize over spatial dimension
        f_new = jax.vmap(lambda i: interp_cubic_v(v_dep[i, :], f[i, :]), out_axes=0)(jnp.arange(self.nx))

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
times, snapshots, E_history = solver.simulate(T_final=50.0, dt=0.1, save_every=5)

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

