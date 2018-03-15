.PHONY: clean build containers

DEBUG=FALSE
PYTHON?=python

ifeq ($(DEBUG),TRUE)
	CMAKE_CUSTOM_FLAGS="-DCMAKE_BUILD_TYPE=Debug"
endif

rebuild:
	$(MAKE) clean
	$(MAKE) build

clean:
	rm -rf cpp/build
	mkdir cpp/build
	cd cpp/build && cmake $(CMAKE_CUSTOM_FLAGS) ..

build:
	cd cpp/build && $(MAKE)
	cd python && $(PYTHON) setup.py clean --all develop

containers:
	docker build -t bayesian_eit:base -f containers/base/dockerfile .
	docker build -t bayesian_eit:worker -f containers/worker/dockerfile .