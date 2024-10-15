# Signatures
Implementation of tensor algebra methods and signature volatility model



### To install iisignature

`pip install iisignature`

- Versions of numpy and numba that work: numpy 1.23.0,  numba 0.58.1

- Check that C++ compilers g++, gcc are installed

- Check the dynamic libraries loaded by the `.so` file that is causing the issue to ensure that it is using the correct `libstdc++`.

  ##### Use `ldd` to check the dependencies:

  ```
  ldd /home/mr_giraffe/anaconda3/lib/python3.8/site-packages/iisignature.cpython-38-x86_64-linux-gnu.so
  ```

  libstdc++ should point to the system version, not to the one of Anaconda. Otherwise, it may cause `undefined symbol: _ZSt28__throw_bad_array_new_lengthv` error.
