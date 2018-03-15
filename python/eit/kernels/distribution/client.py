import numpy as np
from .. import PCNTemperingKernel_C
import multiprocessing
import pika
import time


class MCMCClient(multiprocessing.Process):
    def __init__(self, name, job_queue_url, job_queue_name, n_threads):
        super(multiprocessing.Process, self).__init__()
        self.name = name
        self.__job_queue_url__ = job_queue_url
        self.__job_queue_name__ = job_queue_name
        self.channel = None

        self.__kernels__ = {}
        self.__n_threads__ = n_threads


    def message(self, message):
        print('[{}]: {}'.format(self.name, message)) # TODO: logging framework?
    

    def run(self):
        if self.channel is None:
            conn_params = pika.ConnectionParameters(self.__job_queue_url__, blocked_connection_timeout=5.0)
            connection = None
            while connection is None:
                try:
                    self.message('Attempting connection.')

                    connection = pika.BlockingConnection(conn_params)

                    self.message('Connected')
                except:
                    self.message("Connection to {} refused. Retrying.".format(self.__job_queue_url__))
                    time.sleep(1)
            self.channel = connection.channel()
            self.channel.queue_declare(queue=self.__job_queue_name__)

        self.message("Awaiting message...")
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(self.process_message, queue=self.__job_queue_name__)
        self.channel.start_consuming()


    def process_message(self, channel, method_frame, properties, body):
        try:
            request = messages.deserialize(body)
            if type(request) is messages.ClearKernels:
                to_call = self.clear_kernels
            elif type(request) is messages.CreateKernels:
                to_call = self.initialize_kernels_for_run
            elif type(request) is messages.ApplyKernel:
                to_call = self.apply_kernel
            else:
                raise Exception('Message type {} not understood'.format(type(request)))

            args = {k:v for k,v in request.__dict__.items() if not k.startswith('_')}
            result = to_call(**args)
            if result is None: return
            response = messages.serialize(result)
            ch.basic_publish(exchange='', routing_key=properties.reply_to, body=response)
            ch.basic_ack(delivery_tag=method_frame.delivery_tag)
            self.message('Run complete, reply sent.')

        except Exception as ex:
            self.message('Error encountered: {}'.format(ex))
            raise # TODO: ack failed instead?



    def clear_kernels(self, run_id=None):
        if run_id is None:
            self.__kernels__ = {}
        else:
            keys_to_clear = [k for k in self.__kernels__ if k.startswith(run_id)]
            for k in keys_to_clear:
                self.__kernels__.pop(k, None)


    def initialize_kernels_for_run(self, 
        run_id, 
        init_beta,
        prior_mean,
        sqrt_prior_cov,
        grid,
        likelihood_variance,
        stim_pattern,
        all_data,
        temperatures,
        collocate_args,
        proposal_dot_mat
    ):
        kernel_ids = []
        for i_data in range(len(all_data)):
            data_prev = np.empty((0,0)) if i_data == 0 else dall_data[i_data-1]
            data_next = data_dict[i_data]
            
            for j, t in enumerate(temperatures):
                kernel_name = 'Frame_{}->{}_temp={}'.format(i_data-1, i_data, t)
                kernel_id = run_id + ':' + kernel_name
                self.initialize_kernel(
                    kernel_id,
                    beta,
                    prior_mean,
                    sqrt_proposal_cov,
                    grid,
                    likelihood_variance,
                    pattern,
                    data_prev,
                    data_next,
                    t,
                    fun_args,
                    proposal_dot_mat
                )
                kernel_ids.append(kernel_id)

            kernel_id = run_id + ':Frame_{}'.format(i_data)
            self.initialize_kernel(
                kernel_id,
                beta,
                prior_mean,
                sqrt_proposal_cov,
                grid,
                likelihood_variance,
                np.empty((0,0)),
                data_next,
                1.,
                fun_args,
                proposal_dot_mat
            )
            kernel_ids.append(kernel_id)

        # TODO: pass back the list of kernel_ids


    def initialize_kernel(self, 
        kernel_id, 
        beta, 
        prior_mean, 
        sqrt_prior_cov, 
        grid, 
        likelihood_variance, 
        pattern, 
        data_1, 
        data_2, 
        temp,
        collocate_args, 
        proposal_dot_mat
    ):
        if kernel_id in self.__kernels__:
            # TODO: log error, return failure
            pass

        kernel = PCNTemperingKernel_C(
            beta,
            prior_mean,
            sqrt_prior_cov,
            grid,
            likelihood_variance,
            pattern,
            data_1,
            data_2,
            temp,
            collocate_args,
            proposal_dot_mat
        )
        kernel.name = kernel_id
        self.__kernels__[kernel_id] = kernel

    def apply_kernel(self, kernel_id, kappa_0, n_iter, beta=None, bayesian=True):
        if kernel_id not in self.__kernels__:
            # TODO: log error, return failure
            pass

        kernel = self.__kernels__[kernel_id]
        result = kernel.apply(kappa_0, n_iter, n_threads=self.__n_threads__, beta=beta, bayesian=bayesian)

        # TODO: serialize and return the result somehow