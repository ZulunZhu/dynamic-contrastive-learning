from propagation cimport Instantgnn

cdef class InstantGNN:
	cdef Instantgnn c_instantgnn

	def __cinit__(self):
		self.c_instantgnn=Instantgnn()

	def initial_operation(self,path,dataset,unsigned int m,unsigned int n,rmax,alpha,epsilon,np.ndarray array3, algorithm):
		return self.c_instantgnn.initial_operation(path.encode(),dataset.encode(),m,n,rmax,alpha,epsilon,Map[MatrixXd](array3),algorithm.encode())
	
	def snapshot_lazy(self, upfile, rmax,alpha, np.ndarray array3, np.ndarray array4, np.ndarray array5, algorithm):
		return self.c_instantgnn.snapshot_lazy(upfile.encode(), rmax, alpha, Map[MatrixXd](array3),Map[MatrixXd](array4), Map[MatrixXd](array5), algorithm.encode())
	
	def snapshot_operation(self, upfile, rmax,alpha, np.ndarray array3, algorithm):
		return self.c_instantgnn.snapshot_operation(upfile.encode(), rmax, alpha, Map[MatrixXd](array3),algorithm.encode())
	