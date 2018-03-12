
# crappy - needs to be kept in sync with C++
cdef extern from "collocate.hpp":
	cpdef enum CollocationSolver:
		LU = 0
		LDLT = 1
		QR = 2
		SVD = 3

cdef inline CollocationSolver solver_to_enum(solver):
	if solver is None: return CollocationSolver.LDLT
	if solver == "LU": return CollocationSolver.LU
	if solver == "LDLT": return CollocationSolver.LDLT
	if solver == "QR": return CollocationSolver.QR
	if solver == "SVD": return CollocationSolver.SVD
	raise Exception("Solver {} not understood".format(solver))