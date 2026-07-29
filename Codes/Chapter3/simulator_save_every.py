import os
import time
import copy
import pickle
import numpy as np
import scipy
from scipy.integrate import trapezoid
from concurrent.futures import ProcessPoolExecutor, as_completed
import time as _time, os as _os
from numpy.random import SeedSequence

# Goal of the current code : Be able to generate data that is converged and return the necessary time to do so.
# Then, it returns the histoigrams of displacements as pickles, and the treatment code takes care of computing msds and diffusivities.
# Its cool.


# Please note that I developed this code while the theory was also being developed, so as a rule of thumb, there is no kbT, and
# if there is one, it's equal to 1. It is also not perfect, I sometimes relied on old stackexchange posts or gpt for tips on optimisation.
from tm_sinusoid_trap_pack_save_every import trajectory_sin_trap


class SinusoidalTrapSimulator:
    def __init__(
        self,
        mu_11,
        mu_12,
        mu_22,
        kappa,
        alpha,
        l,
        output_dir,
        N_trajectories,
        n_tasks,
        output_subfile,
        save_every=1000,
        # --- global convergence controls ---
        use_convergence=True,
        tol=1e-2,  # global distance tolerance
        check_every=10,  # trajectories per task per batch
        patience=3,  # consecutive under-tol checks to stop
        min_trajectories=200,  # GLOBAL minimum
        max_trajectories=10000,  # GLOBAL safety cap
    ):
        self.mu = np.array(
            [[mu_11, mu_12], [mu_12, mu_22]]
        )  # Mobility matrix construction
        self.kappa = float(kappa)  # Compliance of the optical trap
        self.alpha = float(alpha)  # Amplitude peak-to-peak of the sinusoidal trap
        self.l = float(l)  # Spatial period of the sinusoidal trap
        self.output_dir = (
            f"{output_dir}"  # Output directory where pickles will be saved
        )
        self.N_trajectories = int(
            N_trajectories
        )  # Only for backwards compatibility and quick checks. How many trajectories
        self.n_tasks = int(n_tasks)  # How many processes in parallel
        self.output_subfile = (
            output_subfile  # Subfiles in which the pickles will be saved
        )
        self.save_every = int(
            save_every
        )  # Subsampling trajectories in case they are heavy for the ram

        # convergence config (GLOBAL)
        self.use_convergence = bool(use_convergence)  # T or F - Use convergence ?
        self.tol = float(tol)  # Tolerance of the convergence
        self.check_every = int(
            check_every
        )  # Check the convergence every check_every trajectories
        self.patience = int(
            patience
        )  # How many batches under tol do we wait to declare convergence ?
        self.min_trajectories = int(
            min_trajectories
        )  # Minimal amount of trajectories before computing convergence
        self.max_trajectories = int(max_trajectories)  # Maximal amount of trajectories

        # dynamics precompute
        self.gamma = np.linalg.inv(self.mu)  # Friction matrix gamma construction
        self.gamma_11, self.gamma_12, self.gamma_22 = (
            self.gamma[0, 0],
            self.gamma[1, 0],
            self.gamma[1, 1],
        )  # Friction matrix elements

        self.tau = self.simulation_time_step()  # Simulation time step duration
        self.D_LJ = (
            self.D_eff_Lifson_Jackson()
        )  # uncoupled long-time diffusion coefficient
        self.total_simulation_time = (
            100 * self.tau_star()
        )  # Totl simulation time in seconds
        self.N = int(
            self.total_simulation_time / self.tau
        )  # How manny steps per trajectory ?
        self.N_saved = max(
            1, self.N // self.save_every
        )  # If subsampled : How many saved ?

        self.time_lags = self.time_lag_array(
            n_points=21, lower_bound=int(10 * self.tau_star() / self.tau)
        )  # Array of time lags at which displacement histograms are computed
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_dict = self.init_data_dict()

    # --------- helpers ----------
    def simulation_time_step(self):
        """
        Returns a timestep to ensure simulation work and converge to a proper value.
        Feel free to tweak this, I chose it very small for some last minute tests for reviewer 2.
        """
        return 1e-7

    def tau_star(self):
        """
        Returns the time a which MSD enters long-time regime.
        I also tweaked this for last minute tests.
        """
        return self.l**2 / 0.3

    def D_eff_Lifson_Jackson(self):
        """
        Uncoupled, long-time diffusion coefficient as per computed usin Lifson and Jackson's formula.
        """
        q = np.linspace(0, self.l, 100_000)
        phi = self.alpha * np.sin(2 * np.pi * q / self.l)
        num = trapezoid(np.exp(phi), q)
        den = trapezoid(np.exp(-phi), q)
        return self.l**2 / (num * den * self.gamma_11)

    def time_lag_array(self, n_points, lower_bound):
        """
        Log spaced array of time lags. The array has n_points points (lol),
        and takes lower-bound to avoid calculating points outside of the range of relevance.
        """
        lower_bound = max(1, lower_bound)
        raw_lags = np.unique(
            np.logspace(
                np.log10(lower_bound),
                np.log10(max(lower_bound + 1, self.N - 1)),
                n_points,
            ).astype(int)
        )
        max_valid_lag = (self.N_saved - 1) * self.save_every
        return raw_lags[raw_lags < max_valid_lag]

    def init_data_dict(self):
        """Creates a dictionnary to store the displacement histograms alon with important parameters.
        Cool stuff : since we are stacking histograms, it precomputes the bins and their positions.
        """
        data = {
            "tau": self.tau,
            "N": self.N,
            "tau_star": self.tau_star(),
            "Time_lags": self.time_lags,
        }
        for tl in self.time_lags:
            sigma = np.sqrt(2 * self.D_LJ * self.tau * tl)
            bin_edges, bin_centers, num_bins = self.bins_edges_and_centers(sigma)
            data[f"num_bins_{tl}"] = num_bins
            data[f"bins_dq1_{tl}"] = bin_centers.astype(np.float64)
            data[f"histo_dq1_{tl}"] = np.zeros(num_bins, dtype=np.float64)
        return data

    def output_filename_base(self):
        """
        Computes unique names for output subdirs, one for each set of parameters being iterated through.
        """
        return "-".join(
            f"{key}_{value}"
            for key, value in {
                "N": self.N,
                "tau": self.tau,
                "mu11": self.mu[0, 0],
                "mu12": self.mu[0, 1],
                "mu22": self.mu[1, 1],
                "kappa": self.kappa,
                "alpha": self.alpha,
                "l": self.l,
                "conv_global": int(self.use_convergence),
                "tol": self.tol,
            }.items()
        )

    # --------- histogram helpers ----------
    @staticmethod
    def displacements(q, time_lag, save_every):
        """Displacement calculator."""
        index_lag = time_lag // save_every
        return q[index_lag:] - q[:-index_lag]

    @staticmethod
    def bins_edges_and_centers(sigma):
        """TO FIX :
        Calculates the bin width and positions to ensure hitograms are stacked properly.
        ISSUE : Data_range may be too small or too large depending on the expected augmentation. I had enough free time to tweak manually.
        """
        num_bins = 31
        data_range = (-12 * sigma, 12 * sigma)
        bin_edges = np.linspace(data_range[0], data_range[1], num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_edges, bin_centers, num_bins

    @classmethod
    def fill_displacements_pdf(cls, q, bin_edges_by_tl, time_lags, save_every):
        """Returns dict timelag -> counts."""
        out = {}
        for tl in time_lags:
            dq = cls.displacements(q, tl, save_every)
            bin_edges = bin_edges_by_tl[tl]
            hist_values, _ = np.histogram(dq, bins=bin_edges, density=False)
            out[tl] = hist_values.astype(np.float64)
        return out

    @staticmethod
    def _normalize_hist(h_counts):
        """Eh."""
        total = np.sum(h_counts)
        if total <= 0:
            return np.zeros_like(h_counts, dtype=np.float64)
        return h_counts.astype(np.float64) / total

    @staticmethod
    def _tvd(p, q):
        """Total variation distance for convergence checks."""
        return 0.5 * np.sum(np.abs(p - q))

    # --------- PARALLEL BATCH WORKEr ----------
    @staticmethod
    def _seed_rng(task_id, batch_idx, extra=0):
        ss = SeedSequence(
            [
                _os.getpid(),
                int(_time.time() * 1e6),
                task_id,
                batch_idx,
                extra,
            ]  # i didnt invent this
        )
        np.random.seed(ss.generate_state(1)[0] & 0xFFFFFFFF)

    @classmethod
    def _simulate_batch(cls, args):
        """
        Worker: simulate 'batch_size' trajectories and return incremental histograms per time lag.
        Returns (task_id, batch_idx, batch_hist_by_tl, trajectories_done_in_batch, sim_seconds, optional_warn)
        """
        (
            task_id,
            batch_idx,
            batch_size,
            mu_11,
            mu_12,
            mu_21,
            mu_22,
            kappa,
            alpha,
            l,
            tau,
            N,
            time_lags,
            save_every,
            bin_edges_by_tl,
        ) = args

        logs_dir = os.path.join("./", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file_path = os.path.join(logs_dir, f"log_task_{task_id}.txt")

        t0 = time.time()
        warn = None
        try:
            cls._seed_rng(task_id, batch_idx)

            mu = np.array([[mu_11, mu_12], [mu_21, mu_22]])
            gamma = np.linalg.inv(mu)
            g = scipy.linalg.sqrtm(gamma)
            g_11, g_12, g_22 = g[0, 0], g[1, 0], g[1, 1]

            # accumulate hist per TL over the batch
            incr = {
                tl: np.zeros_like(bin_edges_by_tl[tl][:-1], dtype=np.float64)
                for tl in time_lags
            }

            for b in range(batch_size):
                q1, q2 = trajectory_sin_trap(
                    N, g_11, g_12, g_22, tau, kappa, alpha, l, save_every=save_every
                )
                h = cls.fill_displacements_pdf(
                    q1, bin_edges_by_tl, time_lags, save_every
                )
                for tl in time_lags:
                    incr[tl] += h[tl]

            t1 = time.time()
            sim_seconds = t1 - t0

            # light logging (append)
            with open(log_file_path, "a") as log:
                log.write(
                    f"[task {task_id}] batch {batch_idx} size={batch_size} time={sim_seconds:.3f}s\n"
                )

            return (task_id, batch_idx, incr, batch_size, sim_seconds, warn)

        except Exception as e:
            warn = f"Worker {task_id} batch {batch_idx} error: {repr(e)}"
            with open(log_file_path, "a") as log:
                log.write(warn + "\n")
            return (task_id, batch_idx, None, 0, time.time() - t0, warn)

    # --------- GLOBAL COORDINATOR ----------
    def run_all_tasks(self, max_workers=5):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        output_subdir = os.path.join(self.output_dir, self.output_filename_base())
        os.makedirs(output_subdir, exist_ok=True)

        bin_edges_by_tl = {}
        for tl in self.time_lags:
            sigma = np.sqrt(2 * self.D_LJ * self.tau * tl)
            edges, centers, _ = self.bins_edges_and_centers(sigma)
            bin_edges_by_tl[tl] = edges
            self.data_dict[f"bins_dq1_{tl}"] = centers

        # Global accumulators
        global_hist = {
            tl: np.zeros_like(bin_edges_by_tl[tl][:-1], dtype=np.float64)
            for tl in self.time_lags
        }
        global_traj = 0
        last_snapshot = None
        checks_under_tol = 0
        converged = False

        # Pack constant worker args head
        const = (
            self.mu[0, 0],
            self.mu[0, 1],
            self.mu[1, 0],
            self.mu[1, 1],
            self.kappa,
            self.alpha,
            self.l,
            self.tau,
            self.N,
            self.time_lags,
            self.save_every,
            bin_edges_by_tl,
        )

        # Coordinator timing
        t_all_start = time.time()

        batch_idx = 0
        # Kinda unnecessary
        max_workers = min(max_workers, self.n_tasks) if max_workers else self.n_tasks

        while True:
            # stop if fixed-count mode
            if not self.use_convergence and global_traj >= self.N_trajectories:
                converged = True
                break
            # stop if global caps hit
            if self.use_convergence and global_traj >= self.max_trajectories:
                break

            # Submit one micro-batch per task this round
            to_submit = []
            batch_size = (
                self.check_every
                if self.use_convergence
                else max(1, (self.N_trajectories - global_traj) // max(1, self.n_tasks))
            )
            if batch_size <= 0:
                batch_size = 1

            for task_id in range(self.n_tasks):
                args = (task_id, batch_idx, batch_size, *const)
                to_submit.append(args)

            # Run in parallel
            batch_idx += 1
            contributed = 0
            wall_this_round = 0.0
            warns = []

            t_round = time.time()
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(SinusoidalTrapSimulator._simulate_batch, a)
                    for a in to_submit
                ]
                for fut in as_completed(futures):
                    task_id, bidx, incr, done, sim_seconds, warn = fut.result()
                    wall_this_round = max(wall_this_round, sim_seconds)
                    contributed += done
                    if warn:
                        warns.append(warn)
                    if incr is not None:
                        for tl in self.time_lags:
                            global_hist[tl] += incr[tl]

            global_traj += contributed
            t_round2 = time.time()

            # Global convergence check
            if self.use_convergence and global_traj >= self.min_trajectories:
                if last_snapshot is None:
                    last_snapshot = {
                        tl: self._normalize_hist(global_hist[tl])
                        for tl in self.time_lags
                    }
                    checks_under_tol = 0
                else:
                    max_tvd = 0.0
                    for tl in self.time_lags:
                        p_now = self._normalize_hist(global_hist[tl])
                        max_tvd = max(max_tvd, self._tvd(p_now, last_snapshot[tl]))
                    # update streak
                    if max_tvd <= self.tol:
                        checks_under_tol += 1
                    else:
                        checks_under_tol = 0
                    last_snapshot = {
                        tl: self._normalize_hist(global_hist[tl])
                        for tl in self.time_lags
                    }
                    print(
                        f"[coord] round={batch_idx} total={global_traj} maxTVD={max_tvd:.4g} "
                        f"under_tol={checks_under_tol}/{self.patience} tol={self.tol}"
                    )

                    if checks_under_tol >= self.patience:
                        converged = True
                        break

        t_all_end = time.time()
        total_wall = t_all_end - t_all_start

        # Fill data_dict with combined results + meta and save
        out = copy.deepcopy(self.data_dict)
        for tl in self.time_lags:
            out[f"histo_dq1_{tl}"] = global_hist[tl]

        out.update(
            {
                "Global_trajectories": global_traj,
                "Converged_global": bool(converged),
                "convergence_tol_TVD": self.tol if self.use_convergence else None,
                "convergence_check_every": (
                    self.check_every if self.use_convergence else None
                ),
                "convergence_patience": self.patience if self.use_convergence else None,
                "min_trajectories_global": (
                    self.min_trajectories if self.use_convergence else None
                ),
                "max_trajectories_global": (
                    self.max_trajectories if self.use_convergence else None
                ),
                "n_tasks": self.n_tasks,
                "batch_size": self.check_every if self.use_convergence else None,
                "simulation_wall_seconds": total_wall,
            }
        )

        out_file = os.path.join(output_subdir, "combined_global.pickle")
        with open(out_file, "wb") as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
