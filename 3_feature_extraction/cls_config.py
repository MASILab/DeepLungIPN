config = {}
config['topk'] = 5
config['resample'] = None

config['preload_train'] = True

config['preload_val'] = True

config['padmask'] = False

config['crop_size'] = [96,96,96]
config['scaleLim'] = [0.85,1.15]
config['radiusLim'] = [6,100]
config['jitter_range'] = 0.15
config['isScale'] = True

config['random_sample'] = True
config['T'] = 1
config['topk'] = 5
config['stride'] = 4
config['augtype'] = {'flip':True,'swap':False,'rotate':False,'scale':False}

config['detect_th'] = 0.05
config['conf_th'] = -1
config['nms_th'] = 0.05
config['filling_value'] = 160

config['startepoch'] = 20
config['lr_stage'] = [50,100,140,160]
config['lr'] = [0.01,0.001,0.0001,0.00001]
config['miss_ratio'] = 1
config['miss_thresh'] = 0.03
config['anchors'] = [10,30,60]