
# crappy - needs to be kept in sync with C++
cdef extern from "collocate.hpp":
	cpdef enum CollocationSolver:
		LDLT = 1
		QR = 2
		SVD = 3

cdef inline CollocationSolver solver_to_enum(solver):
	cdef CollocationSolver solver_enum
	if solver == "LDLT": solver_enum = CollocationSolver.LDLT
	elif solver == "QR": solver_enum = CollocationSolver.QR
	elif solver == "SVD": solver_enum = CollocationSolver.SVD
	else: raise Exception("Solver {} not understood".format(solver))
	return solver_enum