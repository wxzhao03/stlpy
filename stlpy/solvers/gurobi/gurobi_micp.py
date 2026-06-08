from ..base import STLSolver
from ...STL import LinearPredicate, NonlinearPredicate
import numpy as np

import gurobipy as gp
from gurobipy import GRB

import time

class GurobiMICPSolver(STLSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`, solve the optimization problem using lexicographic optimization:

    Phase 1: Maximize global robustness
    
    .. math::

        \min & -\\rho_{global} + \sum_{t=0}^T x_t^TQx_t + u_t^TRu_t

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = A x_t + B u_t

        & y_{t} = C x_t + D u_t

        & \\rho_{global} \leq \\rho_{local}[t] \\quad \\forall t

        & \\rho_{global} \geq 0
    
    Phase 2: Fix global robustness, maximize sum of local robustness
    
    .. math::
    
        \min & -\sum_{t=0}^T \\rho_{local}[t] + \sum_{t=0}^T x_t^TQx_t + u_t^TRu_t
        
        \\text{s.t. } & \\rho_{global} = \\rho_{global}^* \\text{ (fixed from Phase 1)}
        
        & \\text{(all other constraints)}

    This two-phase approach ensures deterministic selection of predicates with maximum robustness at each OR node, eliminating random activation behavior.

    With Gurobi using mixed-integer convex programming, this gives a globally optimal solution.
    .. note::

        This class implements the lexicographic optimization approach for STL synthesis,
        based on the algorithm described in:

        Belta C, et al.
        *Formal methods for control synthesis: an optimization perspective*.
        Annual Review of Control, Robotics, and Autonomous Systems, 2019.
        https://dx.doi.org/10.1146/annurev-control-053018-023717.

    :param spec:            An :class:`.STLFormula` describing the specification.
    :param sys:             A :class:`.LinearSystem` describing the system dynamics.
    :param x0:              A ``(n,1)`` numpy matrix describing the initial state.
    :param T:               A positive integer fixing the total number of timesteps :math:`T`.
    :param M:               (optional) A large positive scalar used to rewrite ``min`` and ``max`` as
                            mixed-integer constraints. Default is ``1000``.
    :param robustness_cost: (optional) Boolean flag for adding a linear cost to maximize
                            the robustness measure. Default is ``True``.
    :param presolve:        (optional) A boolean indicating whether to use Gurobi's
                            presolve routines. Default is ``True``.
    :param verbose:         (optional) A boolean indicating whether to print detailed
                            solver info. Default is ``True``.
    :param rho_min:         (optional) Minimum global robustness value. Default is ``0.0``.
    :param obstacle_adjustments: (optional) Dict for obstacle-specific adjustments.
                            Format: {obstacle_name: (direction, weight)}
                            direction: 'closer' or 'farther'
                            weight: positive float (e.g., 0.5)
                            Default is ``None``.
    :param objects:         (optional) Dict of obstacle definitions. Required if obstacle_adjustments is used.
    """

    def __init__(self, spec, sys, x0, T, M=1000, robustness_cost=True, 
            presolve=True, verbose=True, rho_min=0.0, 
            obstacle_adjustments=None, objects=None):
        assert M > 0, "M should be a (large) positive scalar"
        super().__init__(spec, sys, x0, T, verbose)
    
        self.M = float(M)
        self.presolve = presolve
        self.rho_min = rho_min
        self.obstacle_adjustments = obstacle_adjustments # Store obstacle adjustment preferences for trajectory refinement
        self.objects = objects  # Store obstacle definitions for robustness calculations

        # Set up the optimization problem
        self.model = gp.Model("STL_MICP")
        
        # Store the cost function, which will added to self.model right before solving
        self.cost = 0.0

        # Set some model parameters
        if not self.presolve:
            self.model.setParam('Presolve', 0)
        if not self.verbose:
            self.model.setParam('OutputFlag', 0)

        if self.verbose:
            print("Setting up optimization problem...")
            st = time.time()  # for computing setup time

        # Create optimization variables
        self.y = self.model.addMVar((self.sys.p, self.T), lb=-float('inf'), name='y')
        self.x = self.model.addMVar((self.sys.n, self.T), lb=-float('inf'), name='x')
        self.u = self.model.addMVar((self.sys.m, self.T), lb=-float('inf'), name='u')
        
        # Global robustness variable
        self.rho = self.model.addMVar(1, name="rho_global", lb=0.0)
        
        # Local robustness variables for each timestep
        self.rho_local = self.model.addMVar(self.T, name="rho_local", lb=0.0)

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()
        self.AddRobustnessConstraint()

        # Add obstacle-specific cost adjustments if specified
        if robustness_cost:
            self.AddRobustnessCost()
        
        if self.obstacle_adjustments is not None:
            self.AddObstacleAdjustmentCost()

        if self.verbose:
            print(f"Setup complete in {time.time()-st} seconds.")

    def AddControlBounds(self, u_min, u_max):
        """
        Add control input bounds to the optimization problem.
        
        :param u_min: Minimum control input values
        :param u_max: Maximum control input values
        """
        for t in range(self.T):
            self.model.addConstr( u_min <= self.u[:,t] )
            self.model.addConstr( self.u[:,t] <= u_max )

    def AddStateBounds(self, x_min, x_max):
        """
        Add state bounds to the optimization problem.
        
        :param x_min: Minimum state values
        :param x_max: Maximum state values
        """
        for t in range(self.T):
            self.model.addConstr( x_min <= self.x[:,t] )
            self.model.addConstr( self.x[:,t] <= x_max )

    def AddQuadraticCost(self, Q, R):
        """
        Add quadratic cost on state and control to the optimization problem.
        
        .. math::
            \sum_{t=0}^T x_t^TQx_t + u_t^TRu_t
            
        :param Q: State cost matrix (n x n)
        :param R: Control cost matrix (m x m)
        """
        self.cost += self.x[:,0]@Q@self.x[:,0] + self.u[:,0]@R@self.u[:,0]
        for t in range(1,self.T):
            self.cost += self.x[:,t]@Q@self.x[:,t] + self.u[:,t]@R@self.u[:,t]

        # print(type(self.cost))
    
    def AddRobustnessCost(self):
        """
        Add a linear cost to maximize the global robustness measure.
        
        This adds -rho_global to the cost function in Phase 1, which encourages the optimizer to find trajectories with maximum robustness.
        """
        self.cost -= 100*self.rho

    def AddRobustnessConstraint(self):
        """
        Add robustness constraints linking global and local robustness.
        
        The global robustness is constrained to be the minimum of all local robustness values:
        
        .. math::
            \\rho_{global} \leq \\rho_{local}[t] \\quad \\forall t
            
        """
        # Global rho is the minimum of all local rhos
        for t in range(self.T):
            self.model.addConstr(self.rho <= self.rho_local[t])
        
        # Global robustness lower bound
        if self.rho_min == 0:
            self.model.addConstr(self.rho >= self.rho_min)
        else:
            self.model.addConstr(self.rho <= self.rho_min)
        
        # Each local rho must also be non-negative
        for t in range(self.T):
            self.model.addConstr(self.rho_local[t] >= 0.0)

    def AddObstacleAdjustmentCost(self):
        """
        Add cost terms to encourage closer/farther trajectories from specific obstacles.
        
        This method is only called when obstacle_adjustments is not None.
        
        For 'closer': minimize sum(rho_obstacle[t]) → encourages smaller safety margins
        For 'farther': maximize sum(rho_obstacle[t]) → encourages larger safety margins
        
        The cost modification only affects Phase 1 trajectory generation.
        """
        if self.obstacle_adjustments is None or self.objects is None:
            return
        
        if self.verbose:
            print("\nAdding obstacle adjustment costs:")
        
        for obs_name, (direction, weight) in self.obstacle_adjustments.items():
            if obs_name not in self.objects:
                print(f"  Warning: {obs_name} not found in objects, skipping.")
                continue
            
            obs_bounds = self.objects[obs_name]
            
            # the robustness for this specific obstacle
            rho_obs = self.model.addMVar(
                self.T, 
                lb=-float('inf'), 
                name=f"rho_{obs_name}"
            )
            
            # Add constraints that define rho_obs[t] as the robustness to this obstacle
            for t in range(self.T):
                self._add_obstacle_robustness_constraints(rho_obs[t], obs_bounds, t)
            
            # Calculate sum of robustness across all timesteps
            rho_obs_sum = sum(rho_obs[t] for t in range(self.T))
            
            # Add to cost function
            if direction == 'closer':
                # Minimize sum of robustness
                self.cost -= weight * rho_obs_sum
                if self.verbose:
                    print(f"  - {obs_name}: closer (weight={weight})")
            else:  # 'farther'
                # Maximize sum of robustness
                self.cost += weight * rho_obs_sum
                if self.verbose:
                    print(f"  + {obs_name}: farther (weight={weight})")

    def _add_obstacle_robustness_constraints(self, rho_obs_t, obs_bounds, t):
        x_min, x_max, y_min, y_max, z_min, z_max = obs_bounds
        tol = 0.1  
        self.model.addConstr(
            rho_obs_t >= self.x[0, t] - (x_max + tol),
            name=f"rho_obs_right_t{t}"
        )
        
        # Left face: x <= x_min - tol  →  -x >= -(x_min - tol)
        # Robustness: -(x) + (x_min - tol) = (x_min - tol) - x
        self.model.addConstr(
            rho_obs_t >= (x_min - tol) - self.x[0, t],
            name=f"rho_obs_left_t{t}"
        )
        
        # Front face: y >= y_max + tol
        # Robustness: y - (y_max + tol)
        self.model.addConstr(
            rho_obs_t >= self.x[1, t] - (y_max + tol),
            name=f"rho_obs_front_t{t}"
        )
        
        # Back face: y <= y_min - tol  →  -y >= -(y_min - tol)
        # Robustness: (y_min - tol) - y
        self.model.addConstr(
            rho_obs_t >= (y_min - tol) - self.x[1, t],
            name=f"rho_obs_back_t{t}"
        )
        
        # Top face: z >= z_max + tol
        # Robustness: z - (z_max + tol)
        self.model.addConstr(
            rho_obs_t >= self.x[2, t] - (z_max + tol),
            name=f"rho_obs_top_t{t}"
        )
        
        # Bottom face: z <= z_min - tol  →  -z >= -(z_min - tol)
        # Robustness: (z_min - tol) - z
        self.model.addConstr(
            rho_obs_t >= (z_min - tol) - self.x[2, t],
            name=f"rho_obs_bottom_t{t}"
        )

    def Solve(self):
        """
        Solve the optimization problem using two-phase lexicographic optimization.
        
        Phase 1: Maximize global robustness
        Phase 2: Fix global rho to optimal value, maximize sum of local robustness
        
        :return: A tuple (x, u, rho, rho_time_series, total_runtime) where:
                 - x: State trajectory (n x T)
                 - u: Control inputs (m x T)
                 - rho: Global robustness value (scalar)
                 - rho_time_series: Local robustness at each timestep (array of length T)
                 - total_runtime: Total solve time in seconds (Phase 1 + Phase 2)
        """

        # print(f"[Solve debug] self.T={self.T}, self.sys.n={self.sys.n}, self.x shape={self.x.shape}")
        
        # ===== PHASE 1: Maximize global robustness =====
        if self.verbose:
            print("\n" + "="*70)
            print("PHASE 1: Maximizing Global Robustness")
            print("="*70)
        
        # Set the cost function for Phase 1
        self.model.setObjective(self.cost, GRB.MINIMIZE)
        
        # Solve Phase 1
        self.model.optimize()
        # print(f"[Phase 1] Runtime: {self.model.Runtime:.4f}s, Status: {self.model.status}")

        if self.model.status != GRB.OPTIMAL:
            if self.verbose:
                # print(f"\nPhase 1 optimization failed with status {self.model.status}.\n")
                pass
            return (None, None, -np.inf, None, self.model.Runtime)
        
        # Get Phase 1 results
        rho_global_phase1 = self.rho.X[0]
        phase1_runtime = self.model.Runtime
        
        if self.verbose:
            pass
            # print(f"\nPhase 1 Complete!")
            # print(f"  Solve time: {phase1_runtime:.4f} seconds")
            # print(f"  Optimal global robustness: {rho_global_phase1:.4f}")
        
        # ===== PHASE 2: Maximize sum of local robustness =====
        if self.verbose:
            pass
            # print("\n" + "="*70)
            # print("PHASE 2: Maximizing Local Robustness (Deterministic Selection)")
            # print("="*70)
        
        # Fix global robustness to Phase 1 optimal value
        rho_fix_constraint = self.model.addConstr(
            self.rho == rho_global_phase1, 
            name="fix_global_rho"
        )
        
        # Phase 2: Fix trajectory from Phase 1
        x_phase1 = self.x.X.copy()
        u_phase1 = self.u.X.copy()

        for t in range(self.T):
            for i in range(self.sys.n):
                self.model.addConstr(self.x[i,t] == x_phase1[i,t])

            for i in range(self.sys.m):
                self.model.addConstr(self.u[i,t] == u_phase1[i,t])

        # New objective: maximize sum of local robustness only
        local_rho_sum = sum(self.rho_local[t] for t in range(self.T))
        phase2_cost = -1.0 * local_rho_sum

        self.model.setObjective(phase2_cost, GRB.MINIMIZE)

        # Solve Phase 2
        # self.model.setParam('TimeLimit', 5)
        self.model.optimize()
        phase2_runtime = self.model.Runtime

        # print(f"[Phase 2] Runtime: {phase2_runtime:.4f}s, Status: {self.model.status}")
        
        if self.model.status != GRB.OPTIMAL:
            if self.verbose:
                print(f"\nPhase 2 optimization failed with status {self.model.status}.")
                print("Returning Phase 1 results (local robustness may not be optimal).\n")
            
            # Remove the fix constraint and get Phase 1 solution
            self.model.remove(rho_fix_constraint)
            self.model.setObjective(self.cost, GRB.MINIMIZE)
            self.model.optimize()
            
            x = self.x.X
            u = self.u.X
            rho = self.rho.X[0]
            rho_time_series = self.rho_local.X

            
            return (x, u, rho, rho_time_series, phase2_runtime)
        
        # Get final results from Phase 2
        x = self.x.X
        u = self.u.X
        rho = self.rho.X[0]
        rho_time_series = self.rho_local.X
        total_runtime = phase1_runtime + phase2_runtime
        # print(f"[Solve] x.shape={x.shape}, u.shape={u.shape}")
        
        return (x, u, rho, rho_time_series, total_runtime)

    def AddDynamicsConstraints(self):
        """
        Add system dynamics constraints to the optimization problem.
        
        Enforces:
        - Initial condition: x[0] = x0
        - Dynamics: x[t+1] = A*x[t] + B*u[t]
        - Output: y[t] = C*x[t] + D*u[t]
        """
        # Initial condition
        self.model.addConstr( self.x[:,0] == self.x0 )

        # Dynamics
        for t in range(self.T-1):
            self.model.addConstr(
                    self.x[:,t+1] == self.sys.A@self.x[:,t] + self.sys.B@self.u[:,t] )

            self.model.addConstr(
                    self.y[:,t] == self.sys.C@self.x[:,t] + self.sys.D@self.u[:,t] )

        self.model.addConstr(
                self.y[:,self.T-1] == self.sys.C@self.x[:,self.T-1] + self.sys.D@self.u[:,self.T-1] )

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        
        Uses local robustness variables rho_local[t] instead of global rho
        to enable lexicographic optimization and deterministic predicate selection.
        """
        # Recursively traverse the tree defined by the specification
        # to add binary variables and constraints that ensure that
        # rho is the robustness value
        z_spec = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
        self.AddSubformulaConstraints(self.spec, z_spec, 0)
        self.model.addConstr( z_spec == 1 )

    def AddSubformulaConstraints(self, formula, z, t):
        """
        Given an STLFormula (formula) and a binary variable (z),
        add constraints to the optimization problem such that z
        takes value 1 only if the formula is satisfied (at time t).

        If the formula is a predicate, this constraint uses the "big-M"
        formulation

            A[x(t);u(t)] - b + (1-z)M >= rho_local[t],

        which enforces A[x;u] - b >= rho_local[t] if z=1, where (A,b) are the
        linear constraints associated with this predicate.
    
        If the formula is not a predicate, we recursively traverse the
        subformulas associated with this formula, adding new binary
        variables z_i for each subformula and constraining

            z <= z_i  for all i

        if the subformulas are combined with conjunction (i.e. all
        subformulas must hold), or otherwise constraining

            z <= sum(z_i)

        if the subformulas are combined with disjuction (at least one
        subformula must hold).
        """
        # We're at the bottom of the tree, so add the big-M constraints
        if isinstance(formula, LinearPredicate):
            # a.T*y - b + (1-z)*M >= rho_local[t]
            # KEY CHANGE: Use rho_local[t] instead of self.rho
            self.model.addConstr( 
                formula.a.T@self.y[:,t] - formula.b + (1-z)*self.M >= self.rho_local[t] 
            )

            # Force z to be binary
            b = self.model.addMVar(1,vtype=GRB.BINARY)
            self.model.addConstr(z == b)
        
        elif isinstance(formula, NonlinearPredicate):
            raise TypeError("Mixed integer programming does not support nonlinear predicates")

        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            if formula.combination_type == "and":
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                    t_sub = formula.timesteps[i]   # the timestep at which this formula
                                                   # should hold
                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub)
                    self.model.addConstr( z <= z_sub )

            else:  # combination_type == "or":
                z_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                    z_subs.append(z_sub)
                    t_sub = formula.timesteps[i]
                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub)
                self.model.addConstr( z <= sum(z_subs) )