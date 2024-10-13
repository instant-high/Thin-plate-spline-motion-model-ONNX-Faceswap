import os
import subprocess
import platform
import cv2
import argparse
import numpy as np
import onnxruntime
onnxruntime.set_default_logger_severity(3)

from tqdm import tqdm
from utils.retinaface import RetinaFace
from utils.face_alignment import get_cropped_head_256

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default='source.jpg')
parser.add_argument("--driving", type=str, default='driving.mp4')
parser.add_argument("--output", type=str, default='result.mp4')
parser.add_argument("--crop_scale", type=float, default=1.25, help="bbox size around the face")
parser.add_argument("--parser_index", default="1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17", type=lambda x: list(map(int, x.split(','))),help='index of swapped parts')
parser.add_argument("--source_segmentation", action="store_true", help="use source segmentation mask")
parser.add_argument("--audio", dest="audio", action="store_true", help="Keep audio")
args = parser.parse_args()

device = 'cuda'

# face detector model:      
detector = RetinaFace("utils/scrfd_2.5g_bnkps.onnx", provider=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"], session_options=None)

# face parser model:
from face_parser.face_parser import FACE_PARSER
facemask = FACE_PARSER(model_path="face_parser/face_parser.onnx",device=device)
parser_index = args.parser_index
assert type(parser_index) == list
    
# kp_detector and tpsmm models:
session_options = onnxruntime.SessionOptions()
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
#providers = ["CPUExecutionProvider"]
#if device == 'cuda':
providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"}),"CPUExecutionProvider"] # EXHAUSTIVE
      
kp_detector = onnxruntime.InferenceSession('tpsmm/kp_detector.onnx', sess_options=session_options, providers=providers)    
tpsm_model = onnxruntime.InferenceSession('tpsmm/tpsmm_rel.onnx', sess_options=session_options, providers=providers)    

def process_image(model, img, size, crop_scale=1.6):
    bboxes, kpss = model.detect(img, det_thresh=0.3)
    assert len(kpss) != 0, "No face detected"
    aimg, mat = get_cropped_head_256(img, kpss[0], size=size, scale=crop_scale)
    return aimg, mat    

def swap():

    # load, crop, align, get kp source face:
    source = cv2.imread(args.source)
    source, mat = process_image(detector, source, 256, crop_scale=args.crop_scale)

    source = source.astype('float32') / 255
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    source = np.transpose(source[np.newaxis].astype(np.float32), (0, 3, 1, 2))        
    ort_inputs = {kp_detector.get_inputs()[0].name: source}
    kp_source = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0]  # 1, 50, 2

    # load video:
    cap = cv2.VideoCapture(args.driving)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # writer:
    if args.audio:
        out = cv2.VideoWriter(('_temp.mp4'),cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w, h))
    else:
        out = cv2.VideoWriter((args.output),cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w, h))

    
    for index in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if ret:
                
            target, matrix = process_image(detector, frame, 256, crop_scale=args.crop_scale)
            
            # only standard animation:
            frame_face = cv2.resize(target, (256, 256))/ 255
            frame_face = np.transpose(frame_face[np.newaxis].astype(np.float32), (0, 3, 1, 2))
            
            ort_inputs = {kp_detector.get_inputs()[0].name: frame_face}
            kp_driving = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0]            
            kp_norm = kp_driving
            
            ort_inputs = {tpsm_model.get_inputs()[0].name: kp_source,tpsm_model.get_inputs()[1].name: source, tpsm_model.get_inputs()[2].name: kp_norm, tpsm_model.get_inputs()[3].name: frame_face}
            animated = tpsm_model.run([tpsm_model.get_outputs()[0].name], ort_inputs)[0]

            animated = np.transpose(animated.squeeze(), (1, 2, 0))
            animated = cv2.cvtColor(animated, cv2.COLOR_RGB2BGR) *255
            
            # face parsing mask - target face or source/animated face
            if args.source_segmentation:
                p_mask = facemask.create_region_mask((animated).astype(np.uint8), parser_index)
            else:
                p_mask = facemask.create_region_mask(target.astype(np.uint8), parser_index)
                
            p_mask = cv2.resize(p_mask,(256,256))
            p_mask = cv2.cvtColor(p_mask, cv2.COLOR_GRAY2RGB)
            
            # mask erosion kernel
            kernel = np.ones((5,5), np.uint8)
            p_mask = cv2.erode(p_mask, kernel, iterations=1)
           
            # add soft border:
            p_mask = cv2.rectangle(p_mask, (5,5), (251,251), (0, 0, 0), 10)
            p_mask = cv2.GaussianBlur(p_mask,(9,9),cv2.BORDER_DEFAULT)
            
            # color correction:
            animatedB = cv2.GaussianBlur(animated,(11,11),0).astype(np.float32)
            targetB = cv2.GaussianBlur(target,(11,11),0).astype(np.float32)
            targetB = cv2.addWeighted(animatedB, 0.5, targetB, 0.5, 0.0) # change values eg: 0.3/0.7 or 0.7/0.3
            animated = animated * targetB / (animatedB)
                            
            # replace target face with animated face.
            animated = p_mask * animated + (1 - p_mask) * (target)
                
            # put all together:
            inverse_matrix = cv2.invertAffineTransform(matrix)
            animated = cv2.warpAffine(animated, inverse_matrix, (w, h))            
            mask = cv2.warpAffine(p_mask, inverse_matrix, (w, h))
            img = (mask * animated + (1 - mask) * (frame)).astype(np.uint8)
            
            # write result video:
            cv2.imshow("Result",img)
            cv2.waitKey(1)
            out.write(img)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
swap()

if args.audio:
    print ("Writing Audio...")
    command = 'ffmpeg.exe -y -vn -i ' + '"' + args.driving + '"' + ' -an -i ' + '_temp.mp4' + ' -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest ' + '"' + args.output + '"'
    subprocess.call(command, shell=platform.system() != 'Windows')
    os.system('cls')
    #print(command) 
    #input("Face swap done. Press Enter to continue...") 
    if os.path.exists('_temp.mp4'):
        os.remove('_temp.mp4')

