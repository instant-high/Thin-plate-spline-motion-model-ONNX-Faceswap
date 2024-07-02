# Thin-plate-spline-motion-model-ONNX-Faceswap
Thin Plate Spline Motion Model - ONNX. Extended version for FaceSwap - HeadSwap - PartSwap

Minimum re-implementation of my old torch version, using tpsmm onnx for faceswap, headswap or partswap.

Models can be downloaded from faceparsing and thin-plate-spline-onnx repo.

https://github.com/instant-high/Thin-plate-spline-motion-model-ONNX-Faceswap/assets/77229558/3327f430-0f3f-459b-a939-ab3fb10f92e4



.

swap_tpsmm.py --driving "input\obama360.mp4" --source "input\johnson.jpg" --output "result\johnson.mp4" --parser_index 1,2,3,4,5,6,10,11,12,13,17 --audio

https://github.com/instant-high/Thin-plate-spline-motion-model-ONNX-Faceswap/assets/77229558/745b8231-0fe4-4cb5-892b-caf118b0ef6b

.

swap_tpsmm.py --driving "input\obama360.mp4" --source "input\pitt.jpg" --output "result\pitt.mp4" --parser_index 1,2,3,4,5,6,10,11,12,13 --audio --source_segmentation

https://github.com/instant-high/Thin-plate-spline-motion-model-ONNX-Faceswap/assets/77229558/cea3453a-2594-463c-b584-555a4266c636

.

Works best for frontal faces. Play with parser_index to swap specific parts and source_segmentation parameter to get best results.







