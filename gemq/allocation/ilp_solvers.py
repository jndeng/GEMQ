import pickle
import numpy as np

import gurobipy as gp
from gurobipy import GRB

# test gurobi license
try:
    m = gp.Model()
except gp.GurobiError as e:
    print(e)
    print(
        "Gurobi license not available!\n"
        "Note: A Gurobi license is not required for Mixtral models due to their "
        "small number of experts, but it is required for other models such as "
        "DeepSeekV2-Lite and OLMoE.\n"
        "If you only want to allocate bits for Mixtral models, please comment out this block."
    )


class GEMQSolver:
    """
    Conduct global ILP bit allocation for GEMQ.
    """

    def __init__(
        self,
        layer_re_path,
        x_space=(1, 2, 3),
        extra_constr="",
        start_layer_idx=0,
    ):
        # load weighted layer reconstruction error as coefficients
        with open(layer_re_path, "rb") as file:
            self.coef = pickle.load(file)

        self.x_space = x_space  # available bit-width candidates
        self.num_moe_layers = len(self.coef) # number of effective MoE layers (exclude dense layers)
        self.num_layers = start_layer_idx + self.num_moe_layers
        self.num_experts = len(self.coef[start_layer_idx]) # assume all layers have the same number of experts
        self.num_x = len(x_space)
        print(f"num_layers: {self.num_layers}, num_experts: {self.num_experts}, x_space: {self.x_space}")

        self.extra_constr = extra_constr
        
        self.start_layer_idx = start_layer_idx

    def add_constraints_c2c3(self, m: gp.Model, vars):
        bits = sorted(self.x_space, reverse=True)

        max2 = bits[1]
        for i in range(self.start_layer_idx, self.num_layers):
            m.addConstr(sum(vars[i, j, max2] for j in range(self.num_experts)) >= 1, f"c2")

        max1 = bits[0]
        for i in range(self.start_layer_idx, self.num_layers):
            m.addConstr(sum(vars[i, j, max1] for j in range(self.num_experts)) >= 1, f"c3")

    def build_ilp(self, total_bits):
        """
        Build an ILP model for all experts.

        Args:
            total_bits: total number of allocated bits (bit budget)
        Returns:
            m: Gurobi model
        """
        m = gp.Model("ilp")

        vars = {}
        for i in range(self.start_layer_idx, self.num_layers):
            for j in range(self.num_experts):
                for k in self.x_space:
                    vars[i, j, k] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

        m.setObjective(
            sum(
                self.coef[i][j][k] * vars[i, j, k]
                for i in range(self.start_layer_idx, self.num_layers)
                for j in range(self.num_experts)
                for k in self.x_space
            ),
            GRB.MINIMIZE
        )

        m.addConstr(
            sum(
                vars[i, j, k] * k
                for i in range(self.start_layer_idx, self.num_layers)
                for j in range(self.num_experts)
                for k in self.x_space
            ) <= total_bits,
            name="c0"
        )

        for i in range(self.start_layer_idx, self.num_layers):
            for j in range(self.num_experts):
                m.addConstr(sum(vars[i, j, k] for k in self.x_space) == 1, name=f"c1_{i}_{j}")

        if self.extra_constr == "c2c3":
            self.add_constraints_c2c3(m, vars)
            
        return m

    def solve_all(self, total_bits):
        """
        Solve an ILP problem for all experts in the model.

        Args:
            n: layer/block index
            total_bits: total number of allocated bits
        Returns:
            opt_set: a dictionary with the following structure:
                {
                    start_layer_idx:  {0: <bit>, 1: <bit>, ..., num_expert-1: <bit>},
                    ...
                    num_layers-1:     {0: <bit>, 1: <bit>, ..., num_expert-1: <bit>},
                }
        """
        try:
            m = self.build_ilp(total_bits)
            m.optimize()

            opt_set = {}
            for v in m.getVars():
                if v.X > 1e-6:
                    items = v.VarName.split("_")[1:]
                    i, j, k = map(int, items)
                    if i in opt_set:
                        opt_set[i][j] = k
                    else:
                        opt_set[i] = {j: k}

            print(f"Obj: {m.ObjVal:g}")

            m.dispose()
            gp.disposeDefaultEnv()

        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")
            exit(1)

        except AttributeError:
            print("Encountered an attribute error")
            exit(1)

        return opt_set

