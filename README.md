Linux
-----

sudo apt-get install libglm-dev

Build
-----

mkdir Debug
cd Debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

mkdir Release
cd Release
cmake -DCMAKE_BUILD_TYPE=Release ..
make
