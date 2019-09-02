from ftplib import FTP
import os, re, time
import ftplib

CONST_BUFFER_SIZE = 8192
LOCAL_DIR = r'D:\BaiduNetdiskDownload\longi_data\190614_pics'
FTP_START_DIR = '/'
START_DATE = 20190501
START_LINE = 'M2EL-101'
LINE_LIST = ['M2EL-101', 'M2EL-201', 'M2EL-301', 'M2EL-401']
CONDITION_NO = 0


def download(ftp, localfile, ftpfile):
    f = open(localfile, "wb").write
    try:
        ftp.retrbinary("RETR %s" % ftpfile, f, CONST_BUFFER_SIZE)
    except:
        return False
    return True


# with FTP('66.0.94.252') as ftp:
#     ftp.login('ELFTP', 'Longi2016el')
#     ftp.cwd('M1EL-101/20180927/DayFlight/NG/')
#     download(ftp, r'LRP504032180900611044.jpg')
#     ftp.quit()


def main():
    with FTP('66.0.94.252') as ftp:
        ftp.login('ELFTP', 'Longi2016el')
        ftp.cwd(FTP_START_DIR)
        # assert START_LINE in ftp.nlst(), 'START_LINE not in filelist! Check if you have write the wrong dir name.'
        for f in ftp.nlst():
            # 迭代第一层：M*EL-**1
            # if f != START_LINE:  # 如果没有到达START_LINE, 则一直跳过
            #     print('Line %s passed.' % f)
            # else:
            #     CONDITION_NO = 1
            # if CONDITION_NO == 0:
            #     continue
            if f not in LINE_LIST:
                continue

            try:
                ftp.cwd(FTP_START_DIR)
            except:
                continue
            if re.match('.*01', f):
                try:
                    ftp.cwd(f)
                except:
                    continue
                first_dir = ftp.pwd()
                for f in ftp.nlst():
                    try:
                        ftp.cwd(first_dir)
                    except:
                        continue
                    # 迭代第二层：日期
                    if re.match('\d{8}', f) and int(f) >= START_DATE:
                        try:
                            ftp.cwd(f)
                        except:
                            continue
                        second_dir = ftp.pwd()
                        for f in ftp.nlst():
                            try:
                                ftp.cwd(second_dir)
                            except:
                                continue
                            # 迭代第三层：日夜班
                            if re.match('.*Flight', f):
                                # 直接进入NG文件夹
                                try:
                                    ftp.cwd(os.path.join(f, 'NG'))
                                except:
                                    continue
                                third_dir = ftp.pwd()
                                for f in ftp.nlst():
                                    try:
                                        ftp.cwd(third_dir)
                                    except:
                                        continue
                                    pic_path = os.path.join(ftp.pwd(), f)
                                    local_pic_path = os.path.join(LOCAL_DIR, ftp.pwd(), f)
                                    # 若目录不存在则创建
                                    if not os.path.exists(os.path.dirname(local_pic_path)):
                                        print('make dir %s' % os.path.dirname(local_pic_path))
                                        os.makedirs(os.path.dirname(local_pic_path))
                                    # 若文件已经存在则跳过
                                    if not os.path.exists(local_pic_path):
                                        tic = time.time()
                                        download(ftp, local_pic_path, f)
                                        toc = time.time()
                                        print('finish download pic: %s, time costs: %.4f' % (f, toc - tic))

                                    else:
                                        print('file exists, pass: %s' % f)
