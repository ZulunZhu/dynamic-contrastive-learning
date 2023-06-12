from eigency.core cimport *
from libcpp.string cimport string

ctypedef unsigned int uint

#cdef extern from "instantAlg.cpp":
cdef extern from "instantAlg_arxiv.cpp":
	pass

cdef extern from "instantAlg.h" namespace "propagation":
	cdef cppclass Instantgnn:
		Instantgnn() except+
		double initial_operation(string,string,uint,uint,double,double,double,Map[MatrixXd], string &) except +
		void snapshot_lazy(string, double, double, Map[MatrixXd], Map[MatrixXd],Map[MatrixXd], string &) except +
		void snapshot_operation(string, double, double, Map[MatrixXd], string &)
		
