# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import logging

import sys
sys.path.append( "../../" )
from hippylib import *

logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
set_log_active(False)

def u_boundary(x, on_boundary):
    return on_boundary and ( x[0] < DOLFIN_EPS )

# Poisson's ratio
Nu = 0.25
    
# strain = 1/2 (grad u + grad u^T)
def strain(v):
    return sym(nabla_grad(v))

# stress = 2 mu strain + lambda tr(strain) I
def sigma(v,e):
    return (e/(1+Nu))*strain(v) + ((e*Nu)/((1+Nu)*(1-2*Nu)))*tr(strain(v))*Identity(v.cell().geometric_dimension()) #v.cell().d
 
class Elastic:
    def __init__(self, mesh, Vh, Prior):
        """
        Construct a model by proving
        - the mesh
        - the finite element spaces for the STATE/ADJOINT variable and the PARAMETER variable
        - the Prior information
        """
        self.mesh = mesh
        self.Vh = Vh
        
        # Initialize Expressions
        self.atrue = Expression('(15.0 - 5.0*sin(3.1416*(x[0]/8.0 - 0.5)))')
        self.f = Expression(("0.0","0.0065")) 
        self.u_o = Vector()
        
        self.u_bdr = Expression(("0.0","0.0")) 
        self.u_bdr0 = Expression(("0.0","0.0")) 
        self.bc = DirichletBC(self.Vh[STATE], self.u_bdr, u_boundary)
        self.bc0 = DirichletBC(self.Vh[STATE], self.u_bdr0, u_boundary)
        
        # Assemble constant matrices      
        self.Prior = Prior
        self.Wuu = self.assembleWuu()
        

        self.computeObservation(self.u_o)
                
        self.A = []
        self.At = []
        self.C = []
        self.Raa = []
        self.Wau = []
        
    def generate_vector(self, component="ALL"):
        """
        Return the list x=[u,a,p] where:
        - u is any object that describes the state variable
        - a is a Vector object that describes the parameter variable.
          (Need to support linear algebra operations)
        - p is any object that describes the adjoint variable
        
        If component is STATE, PARAMETER, or ADJOINT return x[component]
        """
        if component == "ALL":
            x = [Vector(), Vector(), Vector()]
            self.Wuu.init_vector(x[STATE],0)
            self.Prior.init_vector(x[PARAMETER],0)
            self.Wuu.init_vector(x[ADJOINT], 0)
        elif component == STATE:
            x = Vector()
            self.Wuu.init_vector(x,0)
        elif component == PARAMETER:
            x = Vector()
            self.Prior.init_vector(x,0)
        elif component == ADJOINT:
            x = Vector()
            self.Wuu.init_vector(x,0)
            
        return x
    
    def init_parameter(self, a):
        """
        Reshape a so that it is compatible with the parameter variable
        """
        self.Prior.init_vector(a,0)       
       
    def assembleA(self,x, assemble_adjoint = False, assemble_rhs = False):
        """
        Assemble the matrices and rhs for the forward/adjoint problems
        """
        trial = TrialFunction(self.Vh[STATE])
        test = TestFunction(self.Vh[STATE])
        c = Function(self.Vh[PARAMETER], x[PARAMETER])
        Avarf = inner(sigma(trial,c), strain(test))*dx #inner(exp(c)*nabla_grad(trial), nabla_grad(test))*dx #NEEDS FIX
        if not assemble_adjoint:
            bform = inner(self.f, test)*dx
            Matrix, rhs = assemble_system(Avarf, bform, self.bc)
        else:
            # Assemble the adjoint of A (i.e. the transpose of A)
            s = Function(self.Vh[STATE], x[STATE])
            obs = Function(self.Vh[STATE], self.u_o)
            bform = inner(obs - s, test)*dx
            Matrix, rhs = assemble_system(adjoint(Avarf), bform, self.bc0)
            
        if assemble_rhs:
            return Matrix, rhs
        else:
            return Matrix
    
    def assembleC(self, x):
        """
        Assemble the derivative of the forward problem with respect to the parameter
        """
        trial = TrialFunction(self.Vh[PARAMETER])
        test = TestFunction(self.Vh[STATE])
        s = Function(Vh[STATE], x[STATE])
        c = Function(Vh[PARAMETER], x[PARAMETER])
        Cvarf = inner(sigma(s,trial), strain(test))*dx #inner(exp(c) * trial * nabla_grad(s), nabla_grad(test)) * dx #NEEDS FIX
        C = assemble(Cvarf)
#        print "||c||", x[PARAMETER].norm("l2"), "||s||", x[STATE].norm("l2"), "||C||", C.norm("linf")
        self.bc0.zero(C)
        return C
       
    def assembleWuu(self):
        """
        Assemble the misfit operator
        """
        trial = TrialFunction(self.Vh[STATE])
        test = TestFunction(self.Vh[STATE])
        varf = inner(trial, test)*dx
        Wuu = assemble(varf)
        Wuu_t = Transpose(Wuu)
        self.bc0.zero(Wuu_t)
        Wuu = Transpose(Wuu_t)
        self.bc0.zero(Wuu)
        return Wuu
    
    def assembleWau(self, x):
        """
        Assemble the derivative of the parameter equation with respect to the state
        """
        trial = TrialFunction(self.Vh[STATE])
        test  = TestFunction(self.Vh[PARAMETER])
        a = Function(self.Vh[ADJOINT], x[ADJOINT])
        c = Function(self.Vh[PARAMETER], x[PARAMETER])
        varf = inner(sigma(trial,test), strain(a))*dx #inner(exp(c)*nabla_grad(trial),nabla_grad(a))*test*dx #NEEDS FIX
        Wau = assemble(varf)
        Wau_t = Transpose(Wau)
        self.bc0.zero(Wau_t)
        Wau = Transpose(Wau_t)
        return Wau
    
    def assembleRaa(self, x):
        """
        Assemble the derivative of the parameter equation with respect to the parameter (Newton method)
        """
        trial = TrialFunction(self.Vh[PARAMETER])
        test  = TestFunction(self.Vh[PARAMETER])
        s = Function(self.Vh[STATE], x[STATE])
        c = Function(self.Vh[PARAMETER], x[PARAMETER])
        a = Function(self.Vh[ADJOINT], x[ADJOINT])
        zer0 = interpolate(Expression("0"), self.Vh[PARAMETER]) 
        varf = inner(nabla_grad(trial),nabla_grad(test))*dx #zer0*dx #inner(nabla_grad(a),exp(c)*nabla_grad(s))*trial*test*dx #NEEDS FIX
        return assemble(varf)

        
    def computeObservation(self, u_o):
        """
        Compute the syntetic observation
        """
        at = interpolate(self.atrue, Vh[PARAMETER])
        x = [self.generate_vector(STATE), at.vector(), None]
        A, b = self.assembleA(x, assemble_rhs = True)
        
        A.init_vector(u_o, 1)
        solve(A, u_o, b)
        
        # Create noisy data, ud
        MAX = u_o.norm("linf")
        noise = 0.01 * MAX * np.random.normal(0, 1, len(u_o.array()))
        u_o.set_local(u_o.array() + noise)
        plot(Function(Vh[STATE], u_o), mode="displacement", title = "Observation")
    
    def cost(self, x):
        """
        Given the list x = [u,a,p] which describes the state, parameter, and
        adjoint variable compute the cost functional as the sum of 
        the misfit functional and the regularization functional.
        
        Return the list [cost functional, regularization functional, misfit functional]
        
        Note: p is not needed to compute the cost functional
        """        
        assert x[STATE] != None
                
        diff = x[STATE] - self.u_o
        Wuudiff = self.Wuu*diff
        misfit = .5 * diff.inner(Wuudiff)
        
        Rx = Vector()
        self.Prior.init_vector(Rx,0)
        self.Prior.R.mult(x[PARAMETER], Rx)
        reg = .5 * x[PARAMETER].inner(Rx)
        
        c = misfit + reg
        
        return c, reg, misfit
    
    def solveFwd(self, out, x, tol=1e-9):
        """
        Solve the forward problem.
        """
        A, b = self.assembleA(x, assemble_rhs = True)
        A.init_vector(out, 1)
        solver = PETScKrylovSolver("cg", amg_method())
        solver.parameters["relative_tolerance"] = tol
        solver.set_operator(A)
        nit = solver.solve(out,b)
        
#        print "FWD", (self.A*out - b).norm("l2")/b.norm("l2"), nit

    
    def solveAdj(self, out, x, tol=1e-9):
        """
        Solve the adjoint problem.
        """
        At, badj = self.assembleA(x, assemble_adjoint = True,assemble_rhs = True)
        At.init_vector(out, 1)
        
        solver = PETScKrylovSolver("cg", amg_method())
        solver.parameters["relative_tolerance"] = tol
        solver.set_operator(At)
        nit = solver.solve(out,badj)
        
#        print "ADJ", (self.At*out - badj).norm("l2")/badj.norm("l2"), nit
    
    def evalGradientParameter(self,x, mg):
        """
        Evaluate the gradient for the variation parameter equation at the point x=[u,a,p].
        Parameters:
        - x = [u,a,p] the point at which to evaluate the gradient.
        - mg the variational gradient (g, atest) being atest a test function in the parameter space
          (Output parameter)
        
        Returns the norm of the gradient in the correct inner product g_norm = sqrt(g,g)
        """ 
        C = self.assembleC(x)

        self.Prior.init_vector(mg,0)
        C.transpmult(x[ADJOINT], mg)
        Rx = Vector()
        self.Prior.init_vector(Rx,0)
        self.Prior.R.mult(x[PARAMETER], Rx)   
        mg.axpy(1., Rx)
        
        g = Vector()
        self.Prior.init_vector(g,1)
        
        self.Prior.Msolver.solve(g, mg)
        g_norm = sqrt( g.inner(mg) )
        
        return g_norm
        
    
    def setPointForHessianEvaluations(self, x):  
        """
        Specify the point x = [u,a,p] at which the Hessian operator (or the Gauss-Newton approximation)
        need to be evaluated.
        """      
        self.A  = self.assembleA(x)
        self.At = self.assembleA(x, assemble_adjoint=True )
        self.C  = self.assembleC(x)
        self.Wau = self.assembleWau(x)
        self.Raa = self.assembleRaa(x)

        
    def solveFwdIncremental(self, sol, rhs, tol):
        """
        Solve the incremental forward problem for a given rhs
        """
        solver = PETScKrylovSolver("cg", amg_method())
        solver.set_operator(self.A)
        solver.parameters["relative_tolerance"] = tol
        self.A.init_vector(sol,1)
        nit = solver.solve(sol,rhs)
#        print "FwdInc", (self.A*sol-rhs).norm("l2")/rhs.norm("l2"), nit
        
    def solveAdjIncremental(self, sol, rhs, tol):
        """
        Solve the incremental adjoint problem for a given rhs
        """
        solver = PETScKrylovSolver("cg", amg_method())
        solver.set_operator(self.At)
        solver.parameters["relative_tolerance"] = tol
        self.At.init_vector(sol,1)
        nit = solver.solve(sol, rhs)
#        print "AdjInc", (self.At*sol-rhs).norm("l2")/rhs.norm("l2"), nit
    
    def applyC(self, da, out):
        self.C.mult(da,out)
    
    def applyCt(self, dp, out):
        self.C.transpmult(dp,out)
    
    def applyWuu(self, du, out, gn_approx=False):
        self.Wuu.mult(du, out)
    
    def applyWua(self, da, out):
        self.Wau.transpmult(da,out)

    
    def applyWau(self, du, out):
        self.Wau.mult(du, out)
    
    def applyR(self, da, out):
        self.Prior.R.mult(da, out)
        
    def Rsolver(self):        
        return self.Prior.Rsolver
    
    def applyRaa(self, da, out):
        self.Raa.mult(da, out)
            
if __name__ == "__main__":
    set_log_active(False)
    sep = "\n"+"#"*80+"\n"
    nx = 100
    ny = 20
    mesh = RectangleMesh(Point(0, 0),Point(8,0.5),nx,ny, "right")
    Vh2 = VectorFunctionSpace(mesh, 'Lagrange', 1)
    Vh1 = FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    print sep, "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()), sep 
 
    Prior = BiLaplacianPrior(Vh[PARAMETER], gamma=5e-1, delta=1e-1)
    model = Elastic(mesh, Vh, Prior)

    print sep, "Test the gradient and the Hessian of the model", sep        
    a0 = interpolate(Expression("x[0] + 2.2"), Vh[PARAMETER]) #sin(x[0])
    modelVerify(model, a0.vector(), 1e-12)

    print sep, "Find the MAP point", sep
    a0 = interpolate(Expression("20"),Vh[PARAMETER])
    solver = ReducedSpaceNewtonCG(model)
    solver.parameters["abs_tolerance"] = 1e-12
    solver.parameters["inner_rel_tolerance"] = 1e-12
    solver.parameters["c_armijo"] = 1e-4
    solver.parameters["GN_iter"] = 6
    
    x = solver.solve(a0.vector())
    
    if solver.converged:
        print "\nConverged in ", solver.it, " iterations."
    else:
        print "\nNot Converged"

    print "Termination reason: ", solver.termination_reasons[solver.reason]
    print "Final gradient norm: ", solver.final_grad_norm
    print "Final cost: ", solver.final_cost

    print sep, "Compute the low rank Gaussian Approximation of the posterior", sep
    model.setPointForHessianEvaluations(x)
    Hmisfit = ReducedHessian(model, solver.parameters["inner_rel_tolerance"], gauss_newton_approx=False, misfit_only=True)
    p = 25
    k = min( 50, Vh[PARAMETER].dim()-p)
    Omega = np.random.randn(x[PARAMETER].array().shape[0], k+p)
    d, U = singlePassG(Hmisfit, Prior.R, Prior.Rsolver, Omega, k)
    
    posterior = GaussianLRPosterior(Prior, d, U)
    posterior.mean = x[PARAMETER]

    post_tr, prior_tr, corr_tr = posterior.trace(method="Estimator", tol=1e-2, min_iter=20, max_iter=200)
    print "Posterior trace {0:5e}; Prior trace {1:5e}; Correction trace {2:5e}".format(post_tr, prior_tr, corr_tr)
    post_pw_variance, pr_pw_variance, corr_pw_variance = posterior.pointwise_variance("Exact")
    
    fid = File("results/pointwise_variance.pvd")
    fid << Function(Vh[PARAMETER], post_pw_variance, name="Posterior")
    fid << Function(Vh[PARAMETER], pr_pw_variance, name="Prior")
    fid << Function(Vh[PARAMETER], corr_pw_variance, name="Correction")
    
    plt.figure()
    plt.plot(range(0,k), d, 'ob')
    plt.yscale('log')
    plt.title("Spectrum of data misfit Hessian")

    print sep, "Generate samples from Prior and Posterior\n","Export generalized Eigenpairs", sep
    fid_prior = File("samples/sample_prior.pvd")
    fid_post  = File("samples/sample_post.pvd")
    nsamples = 20
    noise = Vector()
    posterior.init_vector(noise,"noise")
    noise_size = noise.array().shape[0]
    s_prior = Function(Vh[PARAMETER], name="sample_prior")
    s_post = Function(Vh[PARAMETER], name="sample_post")
    for i in range(nsamples):
        noise.set_local( np.random.randn( noise_size ) )
        posterior.sample(noise, s_prior.vector(), s_post.vector())
        fid_prior << s_prior
        fid_post << s_post
        
    #Save eigenvalues for printing:
    posterior.exportU(Vh[PARAMETER], "hmisfit/evect.pvd")
    np.savetxt("hmisfit/eigevalues.dat", d)

    
    # save and plot
    xx = [Function(Vh[i], x[i]) for i in range(len(Vh))]
    File("results/u_state.pvd") << xx[STATE]
    File("results/E_parameter_inv.pvd") << xx[PARAMETER]
    atrue = interpolate(Expression('(15.0 - 5.0*sin(3.1416*(x[0]/8.0 - 0.5)))'),Vh[PARAMETER])
    File("results/E_parameter_true.pvd") << atrue
    plot(atrue, title = "True Parameter")
    plot(xx[STATE], title = "State",mode="displacement")
    plot(xx[PARAMETER], title = "Parameter")
    plot(xx[ADJOINT], title = "Adjoint")
    interactive()
    
 #   model.setPointForHessianEvaluations(x)
 #   Hmisfit = ReducedHessian(model, solver.parameters["inner_rel_tolerance"], gauss_newton_approx=False, misfit_only=True)
 #   p = 50
 #   k = min( 250, Vh[PARAMETER].dim()-p)
 #   Omega = np.random.randn(x[PARAMETER].array().shape[0], k+p)
 #   d, U = singlePassG(Hmisfit, Prior.R, Prior.Rsolver, Omega, k)
 #   plt.figure()
 #   plt.plot(range(0,k), d, 'b*')
 #   plt.yscale('log')
    

    plt.show()
 #   interactive()    
