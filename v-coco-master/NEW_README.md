## 修改的代码说明

### 环境
建议python=3.7的conda，依赖包可能需要自己一步步探索缺失什么就安装（原文没有requirements）

### copy_coco.py
用于把train val test的所有图片都汇集到一个文件夹all data中，请在里面改变对应的路径，并运行。

### script_pick_annotations.py
把 coco_annotation_dir改成你自己的annotation文件的路径。然后直接运行。

### 按照README.md的方法构建v-coco

### V-COCO.py
将里面的base图片路径改为你刚刚汇集所有图片的all data文件夹，并运行代码，观察是否有输出，并生成了5张图片。
