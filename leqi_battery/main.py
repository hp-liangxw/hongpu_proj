from utils.xml_handler import LabelHandler

xml_dir = r"D:\Desktop\lj_banpian_quanpian_filter\lj_banpian\xuhan_half_lwb_0806\xml"
LabelHandler.del_empty_xml(xml_dir)
LabelHandler.print_all_labels(xml_dir)