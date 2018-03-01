.PHONY: clean build

DEBUG=FALSE

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
	cd python && python setup.py clean --all develop
