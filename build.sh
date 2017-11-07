rm -r cpp/build;
mkdir cpp/build;
cd cpp/build;
cmake ..;
make -j8;
cd ../../python;
python setup.py clean --all develop;