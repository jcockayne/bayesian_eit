from eit.kernels import distribution
import sys
if __name__ == '__main__':
	job_url = sys.argv[1]
	n_threads = int(sys.argv[2])
	print("Starting worker with job_url={}, n_threads={}".format(job_url, n_threads))
	worker = distribution.client.MCMCClient("W1", job_url, "jobs", n_threads)
	worker.start()