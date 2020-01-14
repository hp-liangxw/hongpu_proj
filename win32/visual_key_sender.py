# from config import config
import sys
import time
import re
import logging
if sys.platform.startswith('win'):
    import win32gui
    import win32api
    import win32clipboard
    import win32con
    import win32com
    import win32com.client


class SelfHandle():
    handle = None
    WScript_shell = None


def send_result(result):
    try:
        logging.info('将结果信号 %s 发送至机台。控制机台流水。', result)
        # if not config.getConfig('camera_station.enable_send_virtualkey_to_equip_client_window', False):
        #     return

        # enable_test = config.getConfig(
        #     'camera_station.enable_equip_client_window_test', False)
        # if result == "OK":
        #     if enable_test:
        #         vk = config.getConfig(
        #             'camera_station.virtual_key_id_for_ok', 120)
        #     else:
        #         vk = win32con.VK_F9
        # else:
        #     if enable_test:
        #         vk = config.getConfig(
        #             'camera_station.virtual_key_id_for_ng', 121)
        #     else:
        #         vk = win32con.VK_F10

        # title_target = "MPS"
        # pattern = '(.*' + title_target + ')'
        # res = find_key(all_window_info(), pattern)
        #
        # for handle in res:
        #     send_message(handle, win32con.VK_F10)
        # logging.info('send virtual key to station client: %s' % str(res))
        # get_WScript_shell().SendKeys('%')

        # close_dialog()

        win32gui.SetForegroundWindow(get_main_window_handle())
        send_close_dialog_message(get_main_window_handle())

        
    except:
        logging.exception("exception on posting key event to specified window")


# def send_text(text):
#     try:
#         if not config.getConfig('camera_station.enable_send_text_to_equip_client_window', False):
#             return
#
#         title_target = config.getConfig(
#             'camera_station.equip_client_window_title', "ASIC")
#         pattern = '(.*' + title_target + ')'
#         res = find_key(all_window_info(), pattern)
#
#         for handle in res:
#             send_text_message(handle, text)
#
#         # get_WScript_shell().SendKeys('%')
#         win32gui.SetForegroundWindow(get_main_window_handle())
#
#     except:
#         logging.exception("exception on sending text to specified window")
#
#
# def close_dialog():
#     try:
#         if not config.getConfig('camera_station.enable_close_dialog_of_equip_client_window', False):
#             return
#
#         title_target = config.getConfig(
#             'camera_station.dialog_of_equip_client_window_title', "Location")
#         pattern = '(.*' + title_target + ')'
#         res = find_key(all_window_info(), pattern)
#
#         for handle in res:
#             send_close_dialog_message(handle)
#         logging.info('send close dialog msg to: %s' % str(res))
#         # get_WScript_shell().SendKeys('%')
#         win32gui.SetForegroundWindow(get_main_window_handle())
#
#     except:
#         logging.exception("exception on closing dialog of specified window")


def all_window_info():
    # 获取活动的窗口，（浏览器多个页面只获取一个）
    hwnd_title = dict()

    def get_all_hwnd(hwnd, mouse):
        if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
            hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})
    win32gui.EnumWindows(get_all_hwnd, 0)
    return hwnd_title


def find_key(data: dict, param: str):
    # 通过正则匹配获取键
    res = []
    for k, v in data.items():
        if re.match(param, v):
            res.append(k)
    return res


def send_message(handle, vk):
    # 向窗口发送消息
    sleep_time = 0.2
    win32gui.SetForegroundWindow(handle)
    time.sleep(sleep_time)
    win32gui.PostMessage(handle, win32con.WM_KEYDOWN, vk)
    win32gui.PostMessage(handle, win32con.WM_KEYUP, vk)
    time.sleep(sleep_time)


# def send_text_message(handle, text):
#     # 向窗口发送文本
#     put_text_to_clipboard(text)
#     sleep_time = config.getConfig(
#         'camera_station.sleep_time_for_equip_client_window', 0.2)
#     win32gui.SetForegroundWindow(handle)
#     time.sleep(sleep_time)
#     win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.WM_KEYDOWN, 0)
#     win32api.keybd_event(86, 0, win32con.WM_KEYDOWN, 0)
#     win32api.keybd_event(86, 0, win32con.KEYEVENTF_KEYUP, 0)
#     win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
#     win32api.keybd_event(win32con.VK_RETURN, 0, win32con.WM_KEYDOWN, 0)
#     win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)
#     time.sleep(sleep_time)
#
#
def send_close_dialog_message(handle):
    sleep_time = 0.2
    win32gui.SetForegroundWindow(handle)
    time.sleep(sleep_time)
    close_dialog_key = 'RETURN'
    win32api.keybd_event(eval('win32con.VK_' + close_dialog_key), 0, win32con.WM_KEYDOWN, 0)
    win32api.keybd_event(eval('win32con.VK_' + close_dialog_key), 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(sleep_time)


def get_main_window_handle():
    pattern = '(.*MPS)'
    # pattern = '(.*微信)'
    res = find_key(all_window_info(), pattern)
    if len(res) > 0:
        handle = res[-1]
    print(handle)
    return handle
#
#
# def get_WScript_shell():
#     if not SelfHandle.WScript_shell:
#         SelfHandle.WScript_shell = win32com.client.Dispatch("WScript.Shell")
#
#     return SelfHandle.WScript_shell
#
#
# def activate_self_window():
#     if not config.getConfig('camera_station.enable_send_virtualkey_to_equip_client_window', False):
#         return
#     try:
#         # get_WScript_shell().SendKeys('%')
#         main_handle = get_main_window_handle()
#         win32gui.SetForegroundWindow(main_handle)
#     except:
#         logging.exception("exception on activating self main window, main window hwnd: %s" % str(main_handle))
#
#
# def put_text_to_clipboard(text):
#     # 写入剪切板
#     win32clipboard.OpenClipboard()
#     win32clipboard.EmptyClipboard()
#     win32clipboard.SetClipboardData(
#         win32con.CF_TEXT, text.encode(encoding='gbk'))
#     win32clipboard.CloseClipboard()
#
#
# def get_text_from_clipboard():
#     # 读取剪切板
#     win32clipboard.OpenClipboard()
#     text = win32clipboard.GetClipboardData(win32con.CF_TEXT)
#     win32clipboard.CloseClipboard()
#     return text


send_result("OK")