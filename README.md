# template-match-banck-card
使用opencv-python实现的基于模板匹配的银行卡号识别项目
# 数据分析与测试过程
back-card-template-match.ipynb是拿到数据后的分析与算法的测试过程，我们在这里将模板经过处理之后切割成了10个部分，存放到了cuted_template中作为预测时的模板
# utils
在utils中
|contours.py|cvTools.py|dilateAndErosion.py|gradX|matchTemplate|
|--|--|--|--|--|
|获取边界框|cv处理工具|膨胀和腐蚀|sobelx梯度|模板匹配|
# 预测
card-ocr.py是预测过程，
可选参数如下
|image|template|chigh|imgshow|
|--|--|--|--|
|银行卡的路径|模板所在的路径(文件夹)|适用银行卡的高度|是否显示(设置为True时显示部分中间过程)|

# 代码解析
详细解释：[opencv案例实战——银行卡模式匹配识别](https://blog.csdn.net/DuLNode/article/details/128531516?spm=1001.2014.3001.5502)
