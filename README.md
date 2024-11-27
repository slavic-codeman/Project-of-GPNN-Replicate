# Project-of-GPNN-Replicate
PKU Cognitive Reasoning Project of GPNN
## Original Project
源代码下载: [GPNN](https://github.com/SiyuanQi-zz/gpnn)
## News
GPNN代码的python3.9改写版本已经上传，待校验。

## Tasks
0. 看论文。
1. 研究Github的配置和提交代码，注意不要覆盖主分支。
2. 尝试下载代码，配置环境。
3. 寻找CAD120数据集。
4. 与原GPNN比对，校验GPNN_python3.9_v1中有且仅有版本兼容性的必要修改。

## Data (at present![alt text](00D3224B.png))
### VCOCO数据集: 
从MS-COCO中提取出来的部分数据，需要先下载[MS-COCO](https://cocodataset.org/#download)中的train\val\test的2014版本的images和train\val的annotations文件，然后再由[VOCO](https://github.com/s-gupta/v-coco)的代码进行转换。为了方便，已经上传了一个经过修改且运行成功的v-coco-master文件夹，先看里面的NEW_README.md，再结合原始的README.md进行数据转换操作。

### HICO-DET: 
可以直接搜索网页下载。这里也提供一个[网盘链接](https://disk.pku.edu.cn/link/AAAB41B22C75AA4D549D0D419C6CD2DD9F),密码队名大写，（可能文件夹内部很深，因为是从服务器上直接zip压缩的）。注意这里的数据格式可能与GPNN的要求并不匹配，只是原始的数据。

### CAD120: 
目前没有找到...

## Schedule
接下来约10天的时间内：
### amannier
请对比原GPNN和GPNN_python3.9_v1的代码，确定有且仅有到python3.9的兼容性的必要修改，没有影响代码逻辑，包括冗余，缺失等等。
### Mao YE
请尝试研究原GPNN代码中把HICO数据集加载到dataloader的方法，确定数据的格式，看和下载的HICO数据集有什么区别。
### qyjwty
请尝试研究原GPNN代码中把VCOCO数据集加载到dataloader的方法，确定数据的格式，看和下载的HICO数据集有什么区别。为了从COCO中拿到VCOCO，需要使用v-coco-master的代码。
### Keran Wang
请尝试寻找CAD120数据集，如果能找到，进一步研究原GPNN代码中该数据集的格式和加载到dataloader的方式。

## Attention
1. 如果想要提交自己的代码，建议弄个自己的branch提交，不要直接提交到main。
2. 数据集可能很大，比如MSCOCO，可能需要一个远程服务器，或者本地非C盘运行。
3. 关于数据集的问题，Mao YE， qyjwty，Keran Wang可以根据各自实际情况重新安排。
4. 数据集优先级暂定为HICO>VCOCO>CAD120，因为CAD120还未找到，VCOCO需要COCO的进一步转化，HICO则已经可以直接下载好。争取在12.16报告前完成HICO或者VCOCO的结果复现。