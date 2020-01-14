import json
from ab_test.json_coders import HPJsonEncoder
from ab_test.el_data import ELData, VIData, ModuleInfo, ImageInfo, MesInfo

with open('014.jpg', 'rb') as f:
    imagebyte = f.read()

module_info = ModuleInfo(id='', rows=1, cols=12, station_stage='chuanjian',
                         module_type='5BB', monocrystal=True, half_plate=True,
                         shift='2019.01.01-白班', layout='5BB')

img_info = ImageInfo(img_name='014.jpg',  # 图像名称
                     img_class='el',  # 图像类别, 外观：wg; EL：el
                     _img_bytes=imagebyte,  # 图像字节流
                     rows=1,
                     cols=12,  # 图片中电池片列数
                     row_lines=[20, 410],
                     col_lines=[17, 213, 413, 613, 814, 1014, 1211, 1407, 1606, 1805, 2008, 2209, 2404],
                     section_idx=0,  # 图片的分段索引,从0开始
                     section_num=12,  # 图片的分段数量
                     edge_removed=False,  # 图像黑边是否已经切除
                     col_flipped=False)

mes_info = MesInfo(equip_id='CJ01',  # 设备号
                   facility_id='M02',  # 车间号
                   product_line='L01')

el_data = ELData(module_info=module_info,
                 img_info=img_info, mes_info=mes_info)
vi_data = VIData(station_server='192.168.3.194',  # 机台的服务地址（用于接受vi检测结果信号）
                 vi_server='192.168.3.194:8000',  # 算法服务器地址 192.168.1.10:8000
                 confirm_server='192.168.3.194:8000',  # 复检服务器地址 192.168.1.10:8000
                 confirm_mode='all',  # 复检: all:全复检; ng_only:只复检坏件; ok_only:只复检好件;
                 # none: 不复检; confirm_only: 只复检，不作vi检测
                 platform='127.0.0.1:8001',
                 defects='yiwu',  # 待检测缺陷名称列表
                 features=[],  # 扩展特征
                 # 其它式列表
                 ext_info={}
                 )

post_data = {
    'el_data': el_data,
    'vi_data': vi_data
}

with open('test.json', 'w') as f:
    json.dump(post_data, f, cls=HPJsonEncoder)
