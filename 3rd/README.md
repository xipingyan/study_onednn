# oneDNN source code

oneDNN is submodule.

# How use dnnbench
```
./tests/benchdnn/benchdnn --conv --help

./tests/benchdnn/benchdnn --conv  --mode=c --dir=FWD_I
 --dt=f16 mb1_ic1024oc512_ih16oh16kh1sh1dh0ph0_iw16ow16kw1sw1dw0pw0 --engine=gpu
```
Get use pattern via set macro 'ONEDNN_VERBOSE=2' for openvino app.

#### matmul

    * `C` for correctness testing.
    * `P` for performance testing.

    verbose:
```
onednn_verbose,v1,primitive,exec,gpu:0,matmul,jit:gemm:any,undef,src:f16::blocked:ab::f0 wei:s8::blocked:ba::f0 bia:f16::blocked:ab::f0_mask2 dst:f16::blocked:ab::f0,attr-scratchpad:user attr-fpmath:f16:true attr-scales:wei:2:f16 attr-post-ops:binary_add:f16:3:ab,,4160x3420:3420x1280,0.732178
```
    Save verbose to txt file: verbose.txt

    ```
    cd oneDNN/scripts/verbose_converter
    python verbose_converter.py -i ./verbose.txt
    ```

    You will get benchdnn command:

    ```
     ./tests/benchdnn/benchdnn --matmul --reset --allow-enum-tags-only=0 --engine=gpu:0 --runtime_dims_masks= --bia_mask=2 --dt=f16:s8:f16 --bia-dt=f16 --stag=ab --wtag=ba --dtag=ab --strides=:: --attr-post-ops=binary_add:f16:3:ab --attr-scales=wei:per_oc:f16 --attr-zero-points= --attr-scratchpad=user --attr-fpmath=f16:true 4160x3420:3420x1280 --mode=p
    ```
 
