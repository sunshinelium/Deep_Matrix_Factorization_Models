import argparse
from Model import DMF,neuMF,GMF,MLP
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Options")

parser.add_argument('-dataName', action='store', dest='dataName', default='D:/project/dataset/user_apps_NY_01.rating')
parser.add_argument('-minId', action='store', dest='minId', default=0)
parser.add_argument('-isRate', action='store', dest='isRate', default=0)
parser.add_argument('-lineSpliter', action='store', dest='lineSpliter', default='\t')
parser.add_argument('-model', action='store', dest='model', default='neuMF') #deepCF,DMF,neuMF
parser.add_argument('-negNum', action='store', dest='negNum', default=4, type=int)
parser.add_argument('-userLayer', action='store', dest='userLayer', default=[ 512,64])
parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[ 1024,64])

parser.add_argument('-MLPlayer', action='store', dest='MLPlayer', default=[ 64,32,16,8])
parser.add_argument('-GMFlayer', action='store', dest='GMFlayer', default=32 )
# parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
parser.add_argument('-lr', action='store', dest='lr', default=0.001)
parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=50, type=int)
parser.add_argument('-batchSize', action='store', dest='batchSize', default=256, type=int)
parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=5)
parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./checkPoint/')
parser.add_argument('-topK', action='store', dest='topK', default=10)

args = parser.parse_args()
title = 'neuMF'
plt.figure()
plt.xlabel('epoch')
plt.ylabel('hr@10')
# plt.legend(['GMF','MLP','neuMF'],['red','black','yellow'])
def inner_compare(model):
    plt.title(title)
    for (mlplayer,c) in zip([[256,128,64],[128,64,32,16],[256,128,64,32]],['red','black','yellow']):
        classifier = eval(model)(args,32,mlplayer,c)
        classifier.run()
        plt.legend()
    plt.savefig('./result_image/'+model+'.png')
    plt.show()

def out_compare():
    plt.title(title)
    for model,c in zip(['GMF','MLP','neuMF'],['red','black','yellow']):
        classifier = eval(model)(args,args.GMFlayer,args.MLPlayer,c)
        classifier.run()
        plt.legend()
    plt.savefig('./result_image/'+title+'.png')
    plt.show()
# inner_compare('MLP')
out_compare()