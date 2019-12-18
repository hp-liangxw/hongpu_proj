import os
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import numpy as np
import math

# 显示所有列
pd.set_option('display.max_columns', None)

FLASK_APP = Flask(__name__)

base_dir = os.path.abspath(os.path.dirname(__file__))
FLASK_APP.config[
    'SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://hongpu:hongpu8192@%s/vi_plat?charset=utf8' % '172.26.23.254:3306'
FLASK_APP.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
FLASK_APP.config['SECRET_KEY'] = '23i$&d5&sdfi6'
DB = SQLAlchemy(FLASK_APP)


# DB_MIGRATE = Migrate(FLASK_APP, DB)


def get_defect_num(row, defect):
    if len(row) == 1 and 'None' in row:
        return 0
    else:
        n = 0
        for i in row:
            name, _, _ = i.split('+')
            if name == defect:
                n += 1
        return n


def get_defect_height_ratio(x):
    defect_height = x['coord_y2'] - x['coord_y1']

    r = x['row']
    grid_lines_by_row = eval(x['grid_lines_by_row'])
    battery_height = grid_lines_by_row[r] - grid_lines_by_row[r - 1]

    height_ratio = defect_height / battery_height

    return height_ratio


def get_defect_width_ratio(x):
    defect_width = x['coord_x2'] - x['coord_x1']

    c = x['col']
    grid_lines_by_col = eval(x['grid_lines_by_col'])
    if x['cols'] == 12:
        battery_width = grid_lines_by_col[c] - grid_lines_by_col[c - 1]
    else:
        battery_width = (grid_lines_by_col[math.ceil(c / 2)] - grid_lines_by_col[math.ceil(c / 2) - 1]) / 2

    widht_ratio = defect_width / battery_width

    return widht_ratio


def get_defect_diagonal_ratio(x):
    defect_height = x['coord_x2'] - x['coord_x1']
    defect_width = x['coord_x2'] - x['coord_x1']
    defect_diagonal = np.sqrt(defect_height ** 2 + defect_width ** 2)

    r = x['row']
    c = x['col']
    grid_lines_by_row = eval(x['grid_lines_by_row'])
    grid_lines_by_col = eval(x['grid_lines_by_col'])
    battery_height = grid_lines_by_row[r] - grid_lines_by_row[r - 1]
    if x['cols'] == 12:
        battery_width = grid_lines_by_col[c] - grid_lines_by_col[c - 1]
    else:
        battery_width = (grid_lines_by_col[math.ceil(c / 2)] - grid_lines_by_col[math.ceil(c / 2) - 1]) / 2
    battery_diagonal = np.sqrt(battery_height ** 2 + battery_width ** 2)

    return defect_diagonal / battery_diagonal


def module_infos():
    json_data = {
        'stage': 'cengqian', 'start_time': '2019-12-16', 'end_time': '2019-12-17', 'shift': '晚班', 'line': 'L5',
        'optional_fields': ['result', 'total_defects_number',
                            'yinlie_battery_number', 'yinlie_total_number',
                            'xuhan_battery_number', 'xuhan_total_number']
    }

    sql_str = """
        SELECT 
            n.id, n.serial_id, n.creat_time as time, s.stage, n.shift, s.product_line, s.name as station_name, 
            CASE 
                WHEN n.is_ng_ai=0 THEN 'OK' 
                WHEN n.is_ng_ai=1 THEN 'NG' 
            END result, 
            n.name as defect_name, n.row, n.col 
        FROM (
            SELECT m.station_id, m.id, m.serial_id, m.creat_time, m.shift, m.is_ng_ai, a.name, a.row, a.col 
            FROM vi_plat.module m LEFT JOIN vi_plat.defect_ai a ON m.id=a.module_id
        ) n LEFT JOIN vi_plat.station s ON n.station_id=s.id 
        WHERE s.stage='{}' AND n.creat_time BETWEEN '{}' AND '{}' AND INSTR(n.shift, '{}')>0 AND s.product_line='{}'
    """.format(json_data['stage'], json_data['start_time'], json_data['end_time'], json_data['shift'],
               json_data['line'])

    # 读数据库
    df = pd.read_sql(sql_str, DB.get_engine())

    # ----预处理----
    # 唯一标识符
    df['uid'] = df['serial_id'] + "+" + df["time"].astype(str)
    # 组件信息
    df['module_info'] = df['stage'] + '+' + df['shift'] + '+' + df['product_line'] + '+' + df['station_name'] + '+' + \
                        df['result']
    # 缺陷信息
    df["defect_info"] = df["defect_name"].astype(str) + "+" + (df["row"]).astype(str) + "+" + (df["col"]).astype(str)
    # 聚合
    module_grouped = df[['uid', 'module_info', 'defect_info']].groupby(df['uid']).agg(list)
    # 去重
    module_grouped['uid'] = module_grouped['uid'].apply(lambda x: list(set(x))[0])
    module_grouped['module_info'] = module_grouped['module_info'].apply(lambda x: list(set(x))[0])

    # 字段统计
    if 'total_defects_number' in json_data['optional_fields']:
        module_grouped["total_defects_number"] = module_grouped['defect_info'].apply(
            lambda x: 0 if len(x) == 1 and 'None' in x else len(x))
    if 'yinlie_battery_number' in json_data['optional_fields']:
        module_grouped["yinlie_battery_number"] = module_grouped['defect_info'].apply(
            lambda x: get_defect_num(set(x), 'yinlie'))
    if 'yinlie_total_number' in json_data['optional_fields']:
        module_grouped["yinlie_total_number"] = module_grouped['defect_info'].apply(
            lambda x: get_defect_num(x, 'yinlie'))
    if 'xuhan_battery_number' in json_data['optional_fields']:
        module_grouped["xuhan_battery_number"] = module_grouped['defect_info'].apply(
            lambda x: get_defect_num(set(x), 'xuhan'))
    if 'xuhan_total_number' in json_data['optional_fields']:
        module_grouped["xuhan_total_number"] = module_grouped['defect_info'].apply(lambda x: get_defect_num(x, 'xuhan'))

    # 结果输出
    module_grouped['id'] = module_grouped['uid'].apply(lambda x: x.split('+')[0])
    module_grouped['time'] = module_grouped['uid'].apply(lambda x: x.split('+')[1])
    module_grouped['station_name'] = module_grouped['module_info'].apply(lambda x: x.split('+')[-2])
    module_grouped['result'] = module_grouped['module_info'].apply(lambda x: x.split('+')[-1])
    #
    result = []
    for _, row in module_grouped.iterrows():
        print(row.to_dict())
        result.append(row.to_dict())

    # print(module_grouped.columns)
    # for _, row in module_grouped.iterrows():
    #     print(row.module_info)

    # print(module_grouped.tail())
    return module_grouped


def defect_module():
    json_data = {
        'stage': 'cengqian', 'start_time': '2019-12-16', 'end_time': '2019-12-17', 'shift': '晚班',
        'line': 'L5', 'defect_name': 'yinlie', 'pass_recipe': '是',
        'optional_fields': ['defece_ratio', 'defect_width_ratio', 'defect_height_ratio']
    }

    pass_recipe = 1 if json_data['pass_recipe'] == '是' else 0

    sql_str = """
        SELECT
            n.id, n.serial_id, n.creat_time as time, s.stage, n.shift, s.product_line, s.name as station_name, 
            n.grid_lines_by_row, n.grid_lines_by_col, n.rows, n.cols, 
            CASE
                WHEN n.is_ng_ai=0 THEN 'OK'
                WHEN n.is_ng_ai=1 THEN 'NG'
            END result,
            n.name as defect_name, n.row, n.col, n.confidence, n.area_ratio, 
            n.coord_x1, n.coord_y1, n.coord_x2, n.coord_y2, n.pass_recipe
        FROM
            (SELECT
                m.station_id, m.id, m.serial_id, m.creat_time, m.shift, m.is_ng_ai, 
                m.grid_lines_by_row, m.grid_lines_by_col, m.rows, m.cols,
                a.name, a.row, a.col, a.confidence, a.area_ratio, 
                a.coord_x1, a.coord_y1, a.coord_x2, a.coord_y2, a.pass_recipe
            FROM
                vi_plat.module m LEFT JOIN vi_plat.defect_ai a ON m.id=a.module_id
            ) n LEFT JOIN vi_plat.station s ON n.station_id=s.id
        WHERE
            s.stage='{}' AND n.creat_time BETWEEN '{}' AND '{}' AND INSTR(n.shift, '{}')>0 AND s.product_line='{}' 
            AND n.name="{}" AND n.pass_recipe={};
        """.format(json_data['stage'], json_data['start_time'], json_data['end_time'], json_data['shift'],
                   json_data['line'], json_data['defect_name'], pass_recipe)

    # 读数据库
    df = pd.read_sql(sql_str, DB.get_engine())

    # ----预处理----
    # 行列号
    df['defect_position'] = df["row"].astype(str) + "×" + df["col"].astype(str)
    df['battery_height_ratio'] = df.apply(lambda x: get_defect_height_ratio(x), axis=1)
    df['battery_width_ratio'] = df.apply(lambda x: get_defect_width_ratio(x), axis=1)
    df['battery_diagonal_ratio'] = df.apply(lambda x: get_defect_diagonal_ratio(x), axis=1)

    # ----结果输出----
    result = []
    for _, row in df.iterrows():
        print(row.to_dict())
        result.append(row.to_dict())


defect_module()
