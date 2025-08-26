import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<使用教程.md>下方常见错误和解决方案
warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-AIFI-DPB.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/1/1.yaml',
                cache=False,
                imgsz=640,
                epochs=120,
                batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
                workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
                # resume='', # last.pt path
                project='runpolyp/polyp/train',
                name='rt-detr-AIFI-DPB.yaml_train',
                )
    model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-ContextGuidedDown.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/1/1.yaml',
                cache=False,
                imgsz=640,
                epochs=120,
                batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
                workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
                # resume='', # last.pt path
                project='runpolyp/polyp/train',
                name='rt-detr-ContextGuidedDown.yaml_train',
                )
    # model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-PConv-Rep_DPB_PSConv.yaml')
    # # model.load('') # loading pretrain weights
    # model.train(data='dataset/1/1.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=120,
    #             batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
    #             workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
    #             # resume='', # last.pt path
    #             project='runpolyp/polyp/train',
    #             name='rt-detr-PConv-Rep_DPB_PSConv.yaml_train',
    #             )
    # model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-PConv-Rep_DPB_PSFM.yaml')
    # # model.load('') # loading pretrain weights
    # model.train(data='dataset/1/1.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=120,
    #             batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
    #             workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
    #             # resume='', # last.pt path
    #             project='runpolyp/polyp/train',
    #             name='rt-detr-PConv-Rep_DPB_PSFM.yaml_train',
    #             )
    # model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-PConv-Rep_DPB_RetBlockC3.yaml')
    # # model.load('') # loading pretrain weights
    # model.train(data='dataset/1/1.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=120,
    #             batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
    #             workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
    #             # resume='', # last.pt path
    #             project='runpolyp/polyp/train',
    #             name='rt-detr-PConv-Rep_DPB_RetBlockC3.yaml_train',
    #             )
    # model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-PConv-Rep_DPB_SDFM.yaml')
    # # model.load('') # loading pretrain weights
    # model.train(data='dataset/1/1.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=120,
    #             batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
    #             workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
    #             # resume='', # last.pt path
    #             project='runpolyp/polyp/train',
    #             name='rt-detr-PConv-Rep_DPB_SDFM.yaml_train',
    #             )
    # model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-PConv-Rep_DPB_SDI.yaml')
    # # model.load('') # loading pretrain weights
    # model.train(data='dataset/1/1.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=120,
    #             batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
    #             workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
    #             # resume='', # last.pt path
    #             project='runpolyp/polyp/train',
    #             name='rt-detr-PConv-Rep_DPB_SDI.yaml_train',
    #             )
    # model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-PConv-Rep_DPB_WaveletUnPool.yaml')
    # # model.load('') # loading pretrain weights
    # model.train(data='dataset/1/1.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=120,
    #             batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
    #             workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
    #             # resume='', # last.pt path
    #             project='runpolyp/polyp/train',
    #             name='rt-detr-PConv-Rep_DPB_WaveletUnPool.yaml_train',
    #             )
    # model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-PConv-Rep_DPB_WFU.yaml')
    # # model.load('') # loading pretrain weights
    # model.train(data='dataset/1/1.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=120,
    #             batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
    #             workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
    #             # resume='', # last.pt path
    #             project='runpolyp/polyp/train',
    #             name='rt-detr-PConv-Rep_DPB_WFU.yaml_train',
    #             )
    # model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-PConv-Rep_DPB_WTConv2d.yaml')
    # # model.load('') # loading pretrain weights
    # model.train(data='dataset/1/1.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=120,
    #             batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
    #             workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
    #             # resume='', # last.pt path
    #             project='runpolyp/polyp/train',
    #             name='rt-detr-PConv-Rep_DPB_WTConv2d.yaml_train',
    #             )
    # model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-PConv-Rep_DPB_MSGA.yaml')
    # # model.load('') # loading pretrain weights
    # model.train(data='dataset/1/1.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=120,
    #             batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
    #             workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
    #             # resume='', # last.pt path
    #             project='runpolyp/polyp/train',
    #             name='rt-detr-PConv-Rep_DPB_MSGA.yaml_train',
    #             )
    # model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-PConv-Rep_DPB_MPCAFSA.yaml')
    # # model.load('') # loading pretrain weights
    # model.train(data='dataset/1/1.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=120,
    #             batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
    #             workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
    #             # resume='', # last.pt path
    #             project='runpolyp/polyp/train',
    #             name='rt-detr-PConv-Rep_DPB_MPCAFSA.yaml_train',
    #             )
    # model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-PConv-Rep_DPB_CGRFPN.yaml')
    # # model.load('') # loading pretrain weights
    # model.train(data='dataset/1/1.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=120,
    #             batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
    #             workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
    #             # resume='', # last.pt path
    #             project='runpolyp/polyp/train',
    #             name='rt-detr-PConv-Rep_DPB_CGRFPN.yaml_train',
    #             )
    # model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-PConv-Rep_DPB_ContextGuidedBlock_Down.yaml')
    # # model.load('') # loading pretrain weights
    # model.train(data='dataset/1/1.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=120,
    #             batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
    #             workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
    #             # resume='', # last.pt path
    #             project='runpolyp/polyp/train',
    #             name='rt-detr-PConv-Rep_DPB_ContextGuidedBlock_Down.yaml_train',
    #             )
    # model = RTDETR('E:/zhuyunhao/RTDETR-main/ultralytics/cfg/models/rt-detr/rt-detr-PConv-Rep_DPB_ContextGuideFusionModule.yaml')
    # # model.load('') # loading pretrain weights
    # model.train(data='dataset/1/1.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=120,
    #             batch=8, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
    #             workers=8 , # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             device='0', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
    #             # resume='', # last.pt path
    #             project='runpolyp/polyp/train',
    #             name='rt-detr-PConv-Rep_DPB_ContextGuideFusionModule.yaml_train',
    #             )