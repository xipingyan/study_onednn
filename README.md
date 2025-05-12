# Study oneDNN

# How to build

    $ cd study_onednn
    $ git submodule update --init

    <!-- build onednn -->
    $ cd 3rd/oneDNN
    $ mkdir build && cd build
    $ cmake -DDNNL_GPU_RUNTIME=OCL -DCMAKE_INSTALL_PREFIX=install ..
    $ make -j32 && make install

    $ cd study_onednn
    $ mkdir build && cd build
    $ cmake ..
    $ make -j32

    $ ./study_onednn

# Reference

[``oneDNN:``](https://oneapi-src.github.io/oneDNN/supported_primitives.html) <br>
[``X86:``](https://www.felixcloutier.com/x86/index.html) <br>
