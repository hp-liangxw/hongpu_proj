# 电池缺陷检测服务 web API


## 一、机台发送图片给算法服务器

- 机台发送

    地址： http:#\<ip\>:8000\send_img  
    方法： POST  
    发送内容：
    ~~~python
    content-type: "application/json"

    data = {
        "version":"1.0",  
        "serial_number": "LRXXXXXX",    # 电池序列号（能够唯一确定一块电池）
        "img_name":"xxxx.jpg",          # 图像文件的名称
        "camera_number": 6,             # 摄像机机位号（从1开始编号）
        "img_bytes_b64": b"xxxx",       # 经过base64编码的jpg图像二进制流数据
    }
    # postData 即为发送的输入数据
    postData = json.dumps(data)            
    ~~~

- 服务器返回

    API返回的结果为json字符串，内容为：

    ```python
    {
        "error": ""  # 错误信息，空表示成功
    } 
    ```

## 二、机台查询检测结果

- 机台发送
    
    地址： http:#\<ip\>:8000\query_result\\<电池序列号>  
    方法： GET  

- 服务器返回

    ~~~python
    {
        "version": "1.0", 
        "error": "",                    # 错误信息，空表示成功
        "serial_number": "LRXXXXXX",    # 电池图片序列号
        "defects": {
            # 1号机位拍摄的图片中的缺陷
            "1": [
                {"coord": (x1, y1, x2, y2), "class": "aodian", "prob": 0.98},
                {"coord": (x1, y1, x2, y2), "class": "aodian", "prob": 0.98},
            ],
            # 2号机位拍摄的图片中的缺陷
            # 如果未检测到缺陷时返回空列表
            "2": [],
            # 其他机位图片的检测结果
        }
    }
        
    # 注释： 
    # coord: 缺陷框在图片中的像素位置（图像左上角为原点）
    # class: 缺陷名称
    # prob: 缺陷置信度
    ~~~