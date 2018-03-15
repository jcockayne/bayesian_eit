import pika
from . import messages

class MCMCProxyKernel(object):
	def __init__(self, job_url, job_queue, kernel_name, batch_size):
		self.__job_url__ = job_url
		self.__job_queue__ = job_queue
		self.__kernel_name__ = kernel_name
		self.__batch_size__ = batch_size

		self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()
        

	def apply(self, kappa_0, n_iter, beta=None, bayesian=True):
		# TODO: batch up kappa_0
		# TODO: THIS PATTERN WILL NOT WORK!
		# 		We can't use reply-to because we don't want to await the reply - all messages should go at once,
		#		then we need some kind of mechanism for awaiting and gathering the responses
		#		Perhaps use a message ID and futures?
		kappas = []

		to_send = [messages.ApplyKernel(kappa, n_iter, beta=beta, bayesian=bayesian) for kappa in kappas]
		bodies = [messages.serialize(m) for m in to_send]

		for body in bodies:
			self.channel.basic_publish(
	            exchange='',
	            routing_key=self.__job_queue_name__,
	            body=body,
	            properties=pika.BasicProperties(reply_to='amq.rabbitmq.reply-to')
	        )