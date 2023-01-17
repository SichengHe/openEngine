import scipy
import openmdao.api as om
import numpy as np
import gasfun

class gas_mixture_thermal(om.ExplicitComponent):

    def __init__(self, alpha, n):

        om.ExplicitComponent.__init__(self)

        self.alpha = alpha
        self.n = n

    def setup(self):

        # Inputs
        self.add_input('T', 298.0, units="K", desc="Temperature")

        # Outputs
        self.add_output('s', 0.0, units="J/K", desc="Entropy")
        self.add_output('s_t', 0.0, units="J/(K * K)", desc="Entropy per temperature")
        self.add_output('h', 0.0, units="J", desc="Enthalpy")
        self.add_output('h_t', 0.0, units="J/K", desc="Enthalpy per temperature")
        self.add_output('cp', 0.0, units="J/(kg * K)", desc="Specific heat")
        self.add_output('r', 0.0, units="J", desc="Gas constant")

        # Every output depends on `a`
        self.declare_partials(of='*', wrt='T', method='cs')

    def compute(self, inputs, outputs):
    
        alpha = self.alpha
        n = self.n

        T = inputs['T']

        s = 0.0
        s_t = 0.0
        h = 0.0
        h_t = 0.0
        cp = 0.0
        r = 0.0

        for i in range(n):

            igas = i + 1

            si, s_ti, hi, h_ti, cpi, ri = gasfun.gasfun(igas, T)

            s = s + si * alpha[i]
            h = h + hi * alpha[i]
            cp = cp + cpi * alpha[i]
            r = r + ri * alpha[i]
            s_t = s_t + s_ti * alpha[i]
            h_t = h_t + h_ti * alpha[i]

        outputs['s'] = s
        outputs['s_t'] = s_t
        outputs['h'] = h
        outputs['h_t'] = h_t
        outputs['cp'] = cp
        outputs['r'] = r

class gas_mixture_thermal_inv(om.ImplicitComponent):

    def __init__(self, alpha, n):

        om.ImplicitComponent.__init__(self)

        self.alpha = alpha
        self.n = n

    def setup(self):
    
        # Inputs
        self.add_input('h', 0.0, units="J", desc="Enthalpy")
        
        # Outputs
        self.add_output('T', 298.0, units="K", desc="Temperature")

        self.declare_partials(of='*', wrt='*', method='cs')

    def apply_nonlinear(self, inputs, outputs, residuals):

        alpha = self.alpha
        n = self.n

        hstar = inputs['h']
        T = outputs['T']
        print("T", T)

        h = 0.0

        for i in range(n):

            igas = i + 1

            hi = gasfun.gasfun(igas, T)[2]

            h = h + hi * alpha[i]

        print("h", h)
        print("hstar", hstar)
        print("h - hstar", h - hstar)

        residuals['T'] = h - hstar



if __name__ == "__main__":

    if 0:

        alpha = [0.781, 0.209, 0.0004, 0.0, 0.00965, 0.0]
        n = 6
        n_air = n - 1
        T = 215.0

        prob = om.Problem()
        prob.model.add_subsystem('gas', gas_mixture_thermal(alpha, n_air),
                                promotes_inputs=['T'])

        prob.setup()
        prob.set_val('T', T)
        fail = prob.run_driver()
        prob.model.list_outputs(val=True, units=True)

        totals = prob.compute_totals(['gas.s', 'gas.s_t', 'gas.h', 'gas.h_t', 'gas.cp', 'gas.r'], ['T'])
        print(totals)

    if 1:
        alpha = [0.781, 0.209, 0.0004, 0.0, 0.00965, 0.0]
        n = 6
        n_air = n - 1
        h = -87237.768456

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('gas', gas_mixture_thermal_inv(alpha, n_air), promotes_inputs=['h'])
        # prob["gas.h"] = h

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.linear_solver = om.ScipyKrylov()

        prob.setup()
        prob.set_val('h', h)
        prob.run_model()

        print(prob.get_val('gas.T'))

    prob = om.Problem()
    model = prob.model

    model.add_subsystem('comp', ImpWithInitial())

    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
    model.linear_solver = om.ScipyKrylov()

    prob.setup()
    prob.run_model()

    print(prob.get_val('comp.x'))


