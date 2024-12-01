# Project-of-GPNN-Replicate
PKU Cognitive Reasoning Project of GPNN
## Original Project
源代码下载: [GPNN](https://github.com/SiyuanQi-zz/gpnn)
## Tasks
0. 看论文。
1. 尝试下载代码，配置环境，试着运行一下。
2. bug: 见my_gpnn/vcoco.py第268行的问题
3. *找创新改进点
## Data
原github中tmp文件的drive已经失效，新的链接为[tmp](https://drive.google.com/drive/folders/1vrY7dEautbrScO2kITdFR0kf9Te_GMAJ)
有两个文件夹，都要解压，确保解压的dst相同，这样都会解压到生成的一个tmp文件中。
同时hico和vcoco子文件夹还有zip文件，要进一步解压，注意要进到各自子文件夹里解压保证解压的结果在当前文件夹中，不然跑到外面去。
## Running Code
目前已经获得数据tmp，请把tmp文件夹，my_gpnn，v-coco-master文件夹放在同一个目录下（假设是gpnn_master），然后需要修改以下文件中的一些路径:
###  my_gpnn/vsrl_utils.py
dir_name更换为指向v-coco-master的data文件夹的路径。
###  my_gpnn/config.py
self.project_root和self.vcoco_data_root更换为指向gpnn_master的路径，即所有文件夹所在的这个目录。
### my_gpnn/utils.py
210行root改为自己tmp/vcoco/vcoco_features的路径。
### 运行
my_gpnn中的vcoco.py,cad120.py,hico.py目前都可以直接python运行，如果要修改参数，可以直接去各自main里的参数默认设置改。
关于环境配置，这里的requirements.txt可以提供参考，但并不会用到里面所有的库。该文件是直接由一个python3.10的conda环境中导出的,cuda版本11.7。事实上这个项目也没有用到很多特殊的deep learning相关库，环境是比较通用的。（有import error直接安装对应的库就行）
## Attention
1. 如果想要提交自己的代码，建议弄个自己的branch提交，不要直接提交到main。
