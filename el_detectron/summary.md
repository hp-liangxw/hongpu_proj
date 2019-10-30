
# **测试总结报表**

## 一、基本信息

|测试版本||
| ---- | ---- |
|测试人|xavier|
|测试时间|2019-09-29 10:03:59|
|报告版本|v1.0|
<br>

|测试配置||
| ---- | ---- |
|是否前处理|True|
|预处理信息|rows: 6        cols: 12|
|模型与缺陷|/data/el/trains_pytorch/yjh_0918_jndanjing_yinlie<br>yinlie: 0.985<br>shixiao: 0.95<br>/data/el/trains_pytorch/yjh_0918_jndanjing_xuhan<br>xuhan: 0.97|
|自动标签|True|
|后处理|False|
|评估|False|
|答案|/data/el/tests_pytorch/test_by_xavier/defect_location.csv|
<br>

## 二、测试结果统计

1. 测试缺陷分布
    
    |缺陷名称|检测个数|
    | --- | --- |
    |xuhan|10|
    |yinlie|7|
    |shixiao|2|
    
    ![](pics/defects_ratio.jpg)

2. 置信度区间分布
    
    ![](pics/hist_yinlie.jpg)
    ![](pics/hist_shixiao.jpg)
    ![](pics/hist_xuhan.jpg)

3. 测试结果缺陷分布热力图
    ![](pics/heatmap_alg_results.jpg)
<br>
    