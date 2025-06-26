# oneDNN source code

oneDNN is submodule.

# How use dnnbench
```
./tests/benchdnn/benchdnn --conv --help

./tests/benchdnn/benchdnn --conv  --mode=c --dir=FWD_I
 --dt=f16 mb1_ic1024oc512_ih16oh16kh1sh1dh0ph0_iw16ow16kw1sw1dw0pw0 --engine=gpu
```
Get use pattern via set macro 'ONEDNN_VERBOSE=2' for openvino app.
 
