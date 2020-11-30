# 首届国网北京市电力人工智能数据竞赛-安全帽未佩戴行为目标检测赛道
以下为首届国网北京市电力人工智能数据竞赛-安全帽未佩戴行为目标检测赛道基准模型介绍。其中包含了智源联邦学习框架的简化版本（真实版本后期会以论文的形式发布）、智源整理标注的初赛安全帽数据集和基于YOLOv3模型完成的实验。

## 环境要求
* gevent==1.4.0
* loguru==0.5.3
* Python==3.7.6
* numpy==1.18.1
* Pillow==7.0.0
* torch==1.6.0
* terminaltables==3.1.0
* torchvision==0.6.0
* tqdm==4.42.1

详情请参考`baai-federated-learning-helmet-baseline`下面的`requirements.txt`

## 项目结构
```
.
├── README.md
├── baai-client  # 智源联邦学习客户端
│   ├── api
│   │   ├── __init__.py
│   │   └── my_api.py  # 选手联邦学习服务端需要调用智源联邦学习客户端的函数（训练、测试等）
│   ├── config
│   │   ├── __init__.py
│   │   └── project_conf.py  # 本机端口号和环境配置
│   ├── log
│   │   ├── D_preliminary_contest_helmet_federal_M_yolov3_SE_25_CE_4_ID_1.json  # 最后测试集提交结果
│   │   └── D_preliminary_contest_helmet_federal_M_yolov3_SE_25_CE_4_ID_1.log  # 训练、测试过程完整日志
│   ├── main.py  # 智源联邦学习客户端启动主函数
│   ├── service
│   │   └── federated
│   │       ├── client.py  # 智源联邦学习客户端类
│   │       ├── config
│   │       │   └── preliminary_contest_helmet_federal
│   │       │       ├── create_preliminary_contest_helmet_federal_model.sh  # 构建yolov3模型的bash脚本
│   │       │       ├── preliminary_contest_helmet_federal.data  # 本地初赛安全帽数据集路径信息
│   │       │       └── yolov3_preliminary_contest_helmet_federal.cfg  # 根据bash脚本生成的cfg文件
│   │       ├── models
│   │       │   └── models.py  # yolov3模型相关类
│   │       ├── utils
│   │       │   ├── data.py  # 处理加载数据集相关函数
│   │       │   ├── options.py  # 初始化参数函数
│   │       │   └── utils.py  # 辅助功能函数
│   └── utils
│       ├── __init__.py
│       ├── common_utils.py  # 常用功能函数
│       ├── http_request_utils.py  # 通信功能函数（GET、POST）
│       ├── request_api.py  # 调用通信功能函数api
│       └── result_utils.py  # 通信结果封装类
├── contestant-server  # 选手联邦学习服务端
│   ├── config
│   │   ├── __init__.py
│   │   └── project_conf.py  # 本机端口号和环境配置
│   ├── log
│   │   └── D_preliminary_contest_helmet_federal_M_yolov3_SE_25_CE_4.log  # 训练、测试过程完整日志
│   ├── service
│   │   └── federated
│   │       ├── config
│   │       │   └── preliminary_contest_helmet_federal
│   │       │       ├── create_preliminary_contest_helmet_federal_model.sh  # 构建yolov3模型的bash脚本
│   │       │       └── yolov3_preliminary_contest_helmet_federal.cfg  # 根据bash脚本生成的cfg文件
│   │       ├── models
│   │       │   └── models.py  # yolov3模型相关类
│   │       ├── server.py
│   │       ├── utils
│   │       │   ├── options.py  # 初始化参数函数
│   │       │   └── utils.py  # 辅助功能函数
│   │       └── weights  # 保存yolov3预训练模型
│   └── utils
│       ├── __init__.py
│       ├── common_utils.py  # 常用功能函数
│       ├── http_request_utils.py  # 通信功能函数（GET、POST）
│       ├── request_api.py  # 调用通信功能函数api
│       └── result_utils.py  # 通信结果封装类
├── requirements.txt  # 需要安装的python库
└── sgcc-client  # 国网电力联邦学习客户端（功能与智源联邦学习客户端相同，此处不展开介绍）
    ├── api
    │   ├── __init__.py
    │   └── my_api.py
    ├── config
    │   ├── __init__.py
    │   └── project_conf.py
    ├── log
    │   ├── D_preliminary_contest_helmet_federal_M_yolov3_SE_25_CE_4_ID_2.json
    │   └── D_preliminary_contest_helmet_federal_M_yolov3_SE_25_CE_4_ID_2.log
    ├── main.py
    ├── service
    │   └── federated
    │       ├── client.py
    │       ├── config
    │       │   └── preliminary_contest_helmet_federal
    │       │       ├── create_preliminary_contest_helmet_federal_model.sh
    │       │       ├── preliminary_contest_helmet_federal.data
    │       │       └── yolov3_preliminary_contest_helmet_federal.cfg
    │       ├── models
    │       │   └── models.py
    │       ├── utils
    │       │   ├── data.py
    │       │   ├── options.py
    │       │   └── utils.py
    └── utils
        ├── __init__.py
        ├── common_utils.py
        ├── http_request_utils.py
        ├── request_api.py
        └── result_utils.py
```

## 下载地址
* [初赛安全帽数据集](https://open.baai.ac.cn/data-set-detail/MTI2NTE=/Njk=/true)    
其中智源客户端的数据集包括`preliminary_contest_helmet_federal/annotations`下面的`train1.json`，`val1.json`，`test.json`，`test_image_info.json`和`preliminary_contest_helmet_federal/images`下面的`train1`，`val1`，`test`；
国网电力客户端的数据集包括`preliminary_contest_helmet_federal/annotations`下面的`train2.json`，`val2.json`，`test.json`，`test_image_info.json`和`preliminary_contest_helmet_federal/images`下面的`train2`，`val2`，`test`；  
* [yolov3预训练模型](http://dorc-data.ks3-cn-beijing.ksyun.com/2015682aasdf154asdfe5d5aq961fa6eg/weights_yolov3_pre_model/weights.tar.gz)  
当前主要采用`weights`下面的`darknet53.conv.74`
## 运行方式
### 智源联邦学习客户端
* 进入`baai-client/service/federated/config/preliminary_contest_helmet_federal`目录
  * 修改`preliminary_contest_helmet_federal.data`当中的数据路径
  * 生成`yolov3`模型的`cfg`文件  
  `bash create_preliminary_contest_helmet_federal_model.sh 2`
* 进入`baai-client/config`
  * 修改`project_conf.py`当中的`host`和`port`
* 进入`baai-client/service/federated/utils`目录
  * 配置参数，特别是`data_config`，`model_def`，`server_ip`，`server_port`，`client_ip`，`client_port`

* 启动智源联邦学习客户端
  * 进入`baai-client`，运行以下指令  
  `python main.py`

### 国网电力联邦学习客户端
* 进入`sgcc-client/service/federated/config/preliminary_contest_helmet_federal`目录
  * 修改`preliminary_contest_helmet_federal.data`当中的数据路径
  * 生成`yolov3`模型的`cfg`文件  
  `bash create_preliminary_contest_helmet_federal_model.sh 2`
* 进入`sgcc-client/config`
  * 修改`project_conf.py`当中的`host`和`port`
* 进入`sgcc-client/service/federated/utils`目录
  * 配置参数，特别是`data_config`，`model_def`，`server_ip`，`server_port`，`client_ip`，`client_port`

* 启动国网电力联邦学习客户端
  * 进入`sgcc-client`，运行以下指令  
  `python main.py`

### 选手联邦学习服务端
* 把下载好的[yolov3预训练模型](http://dorc-data.ks3-cn-beijing.ksyun.com/2015682aasdf154asdfe5d5aq961fa6eg/weights_yolov3_pre_model/weights.tar.gz)拷贝到`contestant-server/service/federated/weights`
* 进入`contestant-server/service/federated/config/preliminary_contest_helmet_federal`目录
  * 生成`yolov3`模型的`cfg`文件  
  `bash create_preliminary_contest_helmet_federal_model.sh 2`
* 进入`contestant-server/config`
  * 修改`project_conf.py`当中的`host`和`port`
* 进入`contestant-server/service/federated/utils`目录
  * 配置参数，特别是`pretrained_weights`，`model_def`，`server_ip`，`server_port`，`client_ips`，`client_ports`

* 启动选手联邦学习服务端
  * 进入`contestant-server`，运行以下指令  
    `PYTHONPATH=your/project/path/contestant-server python service/federated/server.py`

## 实验指标 
* 正确率：P (Precision) = TP / (TP + FP)，所有预测出来的正例中有多少是真的正例
* 召回率：R (Recall) = TP / (TP + FN)，所有真实正例中预测出了多少真实正例  
* F1值：F1 Score = 2 * P * R / (P + R)，精确率和召回率的调和均值 
* mAP (mean Average Precision): 目标检测模型的评估指标，参考[目标检测模型的评估指标mAP详解（附代码）](https://zhuanlan.zhihu.com/p/37910324)
* 国网电子指标：马上更新

## 实验结果
基于默认实验参数，初赛安全帽测试集基于YOLOv3模型的结果： 
<div class="table">
<table border="1" cellspacing="0" cellpadding="10" width="100%">
<thead>
<tr class="firstHead">  
<th colspan="1" rowspan="1">Index</th> <th>Class name</th> <th>Precision</th> <th>Recall</th> <th>F1</th> <th>AP (mAP)</th>
 </tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>Wear_helmet</td> <td>0.31408</td></td> <td>0.66526</td> <td>0.42670</td> <td>0.55481</td>
</tr>
<tr>
<td>1</td>
<td>No_helmet</td> <td>0.39388</td> <td>0.82585</td> <td>0.53337</td> <td>0.72258</td>
</tr>
<tr>
<td>2</td>
<td>Total</td> <td>0.35398</td> <td>0.74556</td> <td>0.48004</td> <td>0.63870</td>
</tr>
</tbody>
</table>
</div>